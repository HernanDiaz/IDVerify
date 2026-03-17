"""
dataset_sidtd.py — Adaptador del dataset SIDTD para DocVerify.

SIDTD (Synthetic ID and Travel Document dataset, Boned et al. 2024)
  URL: http://datasets.cvc.uab.es/SIDTD/
  Paper: arXiv:2401.01858

Estructura real en disco (templates):
  SIDTD/templates/
  ├── Images/
  │   ├── reals/
  │   │   ├── alb_id_00.jpg
  │   │   ├── alb_id_01.jpg  …  (todos los reals en directorio plano)
  │   │   └── svk_id_99.jpg
  │   └── fakes/
  │       ├── alb_id_00_fake_6_25.jpg
  │       └── …  (todos los fakes en directorio plano)
  └── Annotations/
      ├── reals/
      │   ├── alb_id.json          ← VIA v2, contiene los 100 reals de alb_id
      │   ├── aze_passport.json    ← un JSON por tipo de documento
      │   └── …
      └── fakes/
          ├── alb_id_00_fake_6_25.json
          └── …  (un JSON por imagen fake)

Formato de anotaciones
──────────────────────
• Fake JSON: {"name":…, "ctype":"Inpaint_and_Rewrite"|"Crop_and_Replace",
              "src":"alb_id_00.jpg", "second_src":"None"|"alb_id_XX.jpg",
              "field":"expiry_date", "second_field":"None"|"field_name"}
• Real JSON (VIA v2): {"_via_img_metadata": {"00.jpg<size>":
              {"filename":"00.jpg", "regions":[{"shape_attributes":
              {"name":"rect","x":…,"y":…,"width":…,"height":…},
              "region_attributes":{"field_name":"photo"}}]}}}

Nota: la key VIA es "{number}.jpg" (sin prefijo de clase) y el número
corresponde al sufijo numérico del stem completo (alb_id_00 → "00.jpg").

Reconstrucción de máscaras
──────────────────────────
Para cada fake, el JSON contiene el campo alterado. Cruzando con el VIA JSON
del tipo de documento correspondiente se obtiene el bounding box → máscara.
"""

from __future__ import annotations

import json
import re
import zipfile
from pathlib import Path
from typing import Optional
from urllib.request import urlretrieve

import pandas as pd
from tqdm import tqdm

import config

# ============================================================
# CONSTANTES
# ============================================================

SIDTD_KIND = "templates"
_BASE_URL  = "http://datasets.cvc.uab.es/SIDTD"
_ARCHIVES  = {"templates": f"{_BASE_URL}/templates.zip"}


# ============================================================
# 1. DESCARGA
# ============================================================

def download_sidtd(root: Path = config.SIDTD_ROOT, kind: str = SIDTD_KIND) -> None:
    """Descarga y descomprime SIDTD en `root`. Omite si ya existe."""
    images_dir = root / kind / "Images"
    if images_dir.exists():
        print(f"[SIDTD] Ya existe {images_dir} — descarga omitida.")
        return

    root.mkdir(parents=True, exist_ok=True)
    url      = _ARCHIVES[kind]
    zip_path = root / f"{kind}.zip"

    print(f"[SIDTD] Descargando {url} …")
    urlretrieve(url, zip_path, reporthook=_progress_hook())
    print(f"\n[SIDTD] Descomprimiendo …")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(root)
    zip_path.unlink()
    print(f"[SIDTD] Listo en {root / kind}")


def _progress_hook():
    pbar = [None]
    def hook(block_num, block_size, total_size):
        if pbar[0] is None:
            pbar[0] = tqdm(total=total_size, unit="B", unit_scale=True, desc="  ↓")
        pbar[0].update(min(block_size, max(0, total_size - pbar[0].n)))
        if block_num * block_size >= total_size:
            pbar[0].close()
    return hook


# ============================================================
# 2. PARSING DE ANOTACIONES VIA v2 (reals)
# ============================================================

def _cls_from_stem(stem: str) -> str:
    """
    Extrae el tipo de documento del stem completo.
    Ejemplos:
      "alb_id_00"           → "alb_id"
      "aze_passport_42"     → "aze_passport"
      "rus_internalpassport_07" → "rus_internalpassport"
    """
    return re.sub(r"_\d+$", "", stem)


def _number_from_stem(stem: str) -> str:
    """
    Extrae el número del stem completo.
    "alb_id_00" → "00",  "aze_passport_42" → "42"
    """
    m = re.search(r"_(\d+)$", stem)
    return m.group(1) if m else stem


def _load_via_class_json(via_json_path: Path) -> dict[str, list[dict]]:
    """
    Parsea un JSON VIA v2 de clase (e.g. alb_id.json).

    El VIA tiene formato:
      {"_via_img_metadata": {"00.jpg<size>": {"filename":"00.jpg",
        "regions": [{"shape_attributes":{"name":"rect","x":…}, …}]}}}

    Retorna:
      {"00": [{"field_name": str, "x": int, "y": int, "w": int, "h": int}]}
    El índice usa solo el número (sin extensión) para facilitar el lookup.
    """
    with open(via_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    img_dict = data.get("_via_img_metadata", data)
    result: dict[str, list[dict]] = {}

    for entry in img_dict.values():
        if not isinstance(entry, dict):
            continue
        filename = entry.get("filename", "")
        number   = Path(filename).stem          # "00.jpg" → "00"
        regions  = entry.get("regions", [])
        rects: list[dict] = []

        for r in regions:
            sa = r.get("shape_attributes", {})
            ra = r.get("region_attributes", {})
            if sa.get("name") != "rect":
                continue
            x = int(sa.get("x", 0))
            y = int(sa.get("y", 0))
            w = int(sa.get("width", sa.get("w", 0)))
            h = int(sa.get("height", sa.get("h", 0)))
            if w <= 0 or h <= 0:
                continue
            field_name = str(ra.get("field_name", "")).strip().casefold()
            rects.append({"field_name": field_name, "x": x, "y": y, "w": w, "h": h})

        result[number] = rects

    return result


def _build_real_annot_index(annot_reals_dir: Path) -> dict[str, list[dict]]:
    """
    Construye índice {full_stem → [field_rects]} para todas las imágenes reales.

    Combina el nombre del archivo JSON (tipo de documento) con el número
    de la imagen para reconstruir el stem completo:
      alb_id.json + "00" → "alb_id_00"
    """
    index: dict[str, list[dict]] = {}
    for via_json in sorted(annot_reals_dir.glob("*.json")):
        cls     = via_json.stem          # "alb_id", "aze_passport", …
        entries = _load_via_class_json(via_json)
        for number, rects in entries.items():
            full_stem = f"{cls}_{number}"
            index[full_stem] = rects
    return index


# ============================================================
# 3. PARSING DE METADATA DE FALSIFICACIONES
# ============================================================

def _none_str(val) -> Optional[str]:
    """Convierte el string literal "None" a None de Python."""
    if val is None or str(val).strip().lower() == "none":
        return None
    return str(val).strip()


def _parse_fake_metadata(fake_json_path: Path) -> Optional[dict]:
    """Parsea el JSON de metadata de una imagen falsa."""
    try:
        with open(fake_json_path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        return {
            "name":         raw.get("name"),
            "ctype":        (raw.get("ctype") or raw.get("type_transformation") or "").lower(),
            "src":          _none_str(raw.get("src")),
            "second_src":   _none_str(raw.get("second_src")),
            "field":        _none_str(raw.get("field")),
            "second_field": _none_str(raw.get("second_field")),
        }
    except Exception:
        return None


def _resolve_altered_field(fake_stem: str, meta: dict) -> Optional[str]:
    """
    Determina qué campo fue alterado EN ESTA imagen fake.

    Inpaint_and_Rewrite: siempre meta["field"].
    Crop_and_Replace: hay dos fakes; el campo depende de si esta fake
      es la basada en src o en second_src.
    """
    field  = meta.get("field")
    ctype  = meta.get("ctype", "")

    if "inpaint" in ctype:
        return field

    # Crop_and_Replace: determinar qué documento es "este fake"
    src_stem    = Path(meta["src"]).stem      if meta.get("src")        else None
    second_stem = Path(meta["second_src"]).stem if meta.get("second_src") else None
    # El stem base del fake: quitar sufijo _fake_N_M
    base_stem = re.sub(r"_fake_\d+_\d+$", "", fake_stem)

    if src_stem and base_stem == src_stem:
        return field
    if second_stem and base_stem == second_stem:
        return meta.get("second_field") or field
    # Fallback
    return field


def _get_altered_rects(
    fake_stem: str,
    fake_json_path: Path,
    real_annot_index: dict[str, list[dict]],
) -> list[dict]:
    """
    Devuelve bounding boxes del campo alterado en formato para _mask_from_rects().
    Retorna [] si no se puede determinar.
    """
    meta = _parse_fake_metadata(fake_json_path)
    if meta is None:
        return []

    altered_field = _resolve_altered_field(fake_stem, meta)
    if not altered_field:
        return []
    altered_field = altered_field.strip().casefold()

    # Buscar anotaciones del documento base
    base_stem = re.sub(r"_fake_\d+_\d+$", "", fake_stem)
    rects     = real_annot_index.get(base_stem, [])

    # Match exacto por field_name
    matched = [
        {"x": r["x"], "y": r["y"], "w": r["w"], "h": r["h"]}
        for r in rects if r["field_name"] == altered_field
    ]
    # Fallback: match parcial
    if not matched:
        matched = [
            {"x": r["x"], "y": r["y"], "w": r["w"], "h": r["h"]}
            for r in rects
            if altered_field in r["field_name"] or r["field_name"] in altered_field
        ]
    return matched


# ============================================================
# 4. CONSTRUCCIÓN DEL DATAFRAME PRINCIPAL
# ============================================================

def build_sidtd_dataframe(
    root: Path = config.SIDTD_ROOT,
    kind: str  = SIDTD_KIND,
    subset_name: str = "sidtd",
) -> pd.DataFrame:
    """
    Construye un DataFrame compatible con dataset.build_full_doc_df() para SIDTD.

    Columnas del DataFrame resultante (mismas que FantasyID + extras SIDTD):
        subset, stem, img_path, json_path,
        label, mask_rects_abs, mask_n_rects, mask_area_px,
        n_rect_regions, n_altered_rect_regions,
        sidtd_class, sidtd_attack_type
    """
    kind_dir   = root / kind
    images_dir = kind_dir / "Images"
    annot_dir  = kind_dir / "Annotations"

    assert images_dir.exists(), (
        f"No se encuentra {images_dir}.\n"
        f"Ejecuta: python -c \"from dataset_sidtd import download_sidtd; download_sidtd()\""
    )

    # 1. Índice de anotaciones reales
    print("[SIDTD] Indexando anotaciones VIA v2 …")
    real_annot_index = _build_real_annot_index(annot_dir / "reals")
    print(f"[SIDTD] {len(real_annot_index)} imágenes reales indexadas.")

    rows = []

    # ── Imágenes reales (bonafide) ───────────────────────────
    print("[SIDTD] Indexando reals …")
    for img_path in sorted((images_dir / "reals").glob("*")):
        if img_path.suffix.lower() not in config.IMG_EXTS:
            continue
        stem = img_path.stem                     # "alb_id_00"
        cls  = _cls_from_stem(stem)              # "alb_id"
        rows.append({
            "subset":                 subset_name,
            "stem":                   stem,
            "img_path":               str(img_path),
            "json_path":              None,
            "label":                  0,
            "mask_rects_abs":         "[]",
            "mask_n_rects":           0,
            "mask_area_px":           0,
            "n_rect_regions":         len(real_annot_index.get(stem, [])),
            "n_altered_rect_regions": 0,
            "sidtd_class":            cls,
            "sidtd_attack_type":      None,
        })

    # ── Imágenes falsas (attack) ─────────────────────────────
    print("[SIDTD] Indexando fakes y reconstruyendo máscaras …")
    fake_imgs = sorted([
        p for p in (images_dir / "fakes").glob("*")
        if p.suffix.lower() in config.IMG_EXTS
    ])
    fakes_ann_dir = annot_dir / "fakes"

    for img_path in tqdm(fake_imgs, desc="  fakes"):
        stem      = img_path.stem                # "alb_id_00_fake_6_25"
        cls       = _cls_from_stem(             # "alb_id"
            re.sub(r"_fake_\d+_\d+$", "", stem)
        )
        json_path = fakes_ann_dir / (stem + ".json")
        json_path = json_path if json_path.exists() else None

        meta = _parse_fake_metadata(json_path) if json_path else None
        ctype = meta.get("ctype", "") if meta else ""
        attack_type = (
            "crop_and_replace"    if "crop"    in ctype else
            "inpaint_and_rewrite" if "inpaint" in ctype else
            "unknown"
        )

        altered_rects = (
            _get_altered_rects(stem, json_path, real_annot_index)
            if json_path else []
        )
        mask_area = sum(r["w"] * r["h"] for r in altered_rects)

        rows.append({
            "subset":                 subset_name,
            "stem":                   stem,
            "img_path":               str(img_path),
            "json_path":              str(json_path) if json_path else None,
            "label":                  1,
            "mask_rects_abs":         json.dumps(altered_rects),
            "mask_n_rects":           len(altered_rects),
            "mask_area_px":           mask_area,
            "n_rect_regions":         len(altered_rects),
            "n_altered_rect_regions": len(altered_rects),
            "sidtd_class":            cls,
            "sidtd_attack_type":      attack_type,
        })

    df = pd.DataFrame(rows)
    n_real  = (df["label"] == 0).sum()
    n_fake  = (df["label"] == 1).sum()
    n_masks = (df["mask_n_rects"] > 0).sum()
    print(
        f"[SIDTD] {len(df)} imágenes ({n_real} reales, {n_fake} fakes).\n"
        f"[SIDTD] Fakes con máscara: {n_masks}/{n_fake} "
        f"({100*n_masks/max(n_fake,1):.1f}%)."
    )
    return df


# ============================================================
# 5. DIAGNÓSTICO
# ============================================================

def diagnose(root: Path = config.SIDTD_ROOT) -> None:
    """Resumen del estado del dataset y calidad de las máscaras reconstruidas."""
    df = build_sidtd_dataframe(root)

    print("\n── Distribución por clase ──")
    print(df.groupby(["sidtd_class", "label"]).size().unstack(fill_value=0).to_string())

    print("\n── Tipo de ataque (fakes) ──")
    fakes = df[df["label"] == 1]
    print(fakes["sidtd_attack_type"].value_counts().to_string())

    print("\n── Cobertura de máscaras (fakes) ──")
    print(f"  Con máscara  : {(fakes['mask_n_rects'] > 0).sum()} / {len(fakes)}")
    print(f"  Sin máscara  : {(fakes['mask_n_rects'] == 0).sum()} / {len(fakes)}")
    print(f"  Área media   : {fakes['mask_area_px'].mean():.0f} px²")


if __name__ == "__main__":
    diagnose()
