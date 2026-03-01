"""
dataset.py — Indexación, parsing de anotaciones y construcción de tf.data.Dataset.
"""

import ast
import json
import unicodedata
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import tensorflow as tf
from tqdm import tqdm

import config


# ============================================================
# 1. INDEXACIÓN DE IMÁGENES
# ============================================================

def list_images(split_dir: Path) -> list[Path]:
    """Devuelve todas las imágenes con extensión válida dentro de split_dir."""
    paths = [
        p for p in split_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in config.IMG_EXTS
    ]
    return sorted(paths)


def parse_path_info(img_path: Path, root_dir: Path) -> tuple:
    """
    Extrae metadatos de la ruta relativa al root del dataset.

    Estructura esperada:
      <split>/bonafide/<device>/<file>.jpg
      <split>/attack/<attack_type>/<device>/<file>.jpg

    Devuelve: (split, cls, attack_type, device)
    """
    parts = img_path.relative_to(root_dir).parts
    split = parts[0]   # train | test
    cls   = parts[1]   # bonafide | attack

    if cls == "attack":
        attack_type = parts[2]
        device      = parts[3]
    else:
        attack_type = None
        device      = parts[2]

    return split, cls, attack_type, device


def build_image_dataframe(root_dir: Path) -> pd.DataFrame:
    """
    Recorre root_dir e indexa todas las imágenes de train y test
    en un DataFrame con sus metadatos.
    """
    train_dir = root_dir / "train"
    test_dir  = root_dir / "test"

    assert train_dir.exists(), f"No se encuentra TRAIN_DIR: {train_dir.resolve()}"
    assert test_dir.exists(),  f"No se encuentra TEST_DIR: {test_dir.resolve()}"

    rows = []
    for img_path in list_images(train_dir) + list_images(test_dir):
        split, cls, attack_type, device = parse_path_info(img_path, root_dir)
        rows.append({
            "split":       split,
            "cls_dir":     cls,
            "attack_type": attack_type,
            "device":      device,
            "img_path":    str(img_path),
            "stem":        img_path.stem,
            "ext":         img_path.suffix.lower(),
        })

    df = pd.DataFrame(rows)
    print(f"[OK] Total imágenes indexadas: {len(df)}")
    print(df.groupby(["split", "cls_dir"]).size().to_string())
    return df


# ============================================================
# 2. EMPAREJAMIENTO IMAGEN ↔ JSON
# ============================================================

def find_json_for_image(img_path_str: str) -> Optional[str]:
    """
    Busca el JSON de anotación con el mismo nombre y en la misma
    carpeta que la imagen. Devuelve None si no existe.
    """
    img_p = Path(img_path_str)
    cand  = img_p.with_suffix(".json")
    return str(cand) if cand.exists() else None


def add_json_paths(df: pd.DataFrame) -> pd.DataFrame:
    """Añade la columna json_path al DataFrame y valida que no falte ninguno."""
    df = df.copy()
    df["json_path"] = df["img_path"].apply(find_json_for_image)

    missing = df["json_path"].isna().sum()
    print(f"[OK] JSON encontrados: {df['json_path'].notna().sum()} / {len(df)}")

    if missing > 0:
        print("[ERROR] Imágenes sin JSON asociado:")
        print(df[df["json_path"].isna()][["img_path"]].head(20).to_string())
        raise FileNotFoundError(
            f"Faltan {missing} JSON para imágenes en el dataset."
        )

    return df


# ============================================================
# 3. PARSING DE ANOTACIONES Y GENERACIÓN DE MÁSCARAS
# ============================================================

def _norm_field_name(x) -> Optional[str]:
    """Normaliza nombres de campo Unicode a forma canónica lowercase."""
    if x is None:
        return None
    x = unicodedata.normalize("NFKC", str(x)).strip().casefold()
    return x if x else None


def parse_doc_full_image(
    img_path: str,
    json_path: str,
    stem: str,
    subset: str,
    label: int,
) -> dict:
    """
    Parsea el JSON de anotación de un documento y extrae los rectángulos
    de regiones alteradas (falsificadas).

    Devuelve un diccionario con:
      - mask_rects_abs: JSON string con lista de rects {x,y,w,h} alterados
      - mask_n_rects: número de rectángulos alterados
      - mask_area_px: suma de áreas (puede sobrecontar si hay solapamiento)
      - n_rect_regions: total de regiones rectangulares en el JSON
      - n_altered_rect_regions: total de regiones rectangulares alteradas
    """
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    regions = data.get("regions", []) or []

    altered_rects = []
    n_rect         = 0
    n_rect_altered = 0
    mask_area      = 0

    for ridx, r in enumerate(regions):
        sa = r.get("shape_attributes") or {}
        ra = r.get("region_attributes") or {}

        if sa.get("name") != "rect":
            continue

        x = int(sa.get("x", 0))
        y = int(sa.get("y", 0))
        w = int(sa.get("width", 0))
        h = int(sa.get("height", 0))

        if w <= 0 or h <= 0:
            continue

        n_rect += 1

        prov = ra.get("region_provenance")
        prov = "original" if prov is None else str(prov).strip().casefold()

        if prov == "altered":
            n_rect_altered += 1
            altered_rects.append({
                "x": x, "y": y, "w": w, "h": h,
                "region_index": ridx,
                "field_name": _norm_field_name(ra.get("field_name")),
            })
            mask_area += w * h

    return {
        "subset":               subset,
        "stem":                 stem,
        "img_path":             img_path,
        "json_path":            json_path,
        "label":                int(label),
        "mask_rects_abs":       json.dumps(altered_rects),
        "mask_n_rects":         len(altered_rects),
        "mask_area_px":         mask_area,
        "n_rect_regions":       n_rect,
        "n_altered_rect_regions": n_rect_altered,
    }


def build_full_doc_df(df_base: pd.DataFrame, subset_name: str) -> pd.DataFrame:
    """Construye el DataFrame de anotaciones completo para un split."""
    rows = []
    for _, r in tqdm(df_base.iterrows(), total=len(df_base),
                     desc=f"Parsing JSON ({subset_name})"):
        rows.append(parse_doc_full_image(
            img_path  = r["img_path"],
            json_path = r["json_path"],
            stem      = r["stem"],
            subset    = subset_name,
            label     = r["label"],
        ))
    return pd.DataFrame(rows)


# ============================================================
# 4. PIPELINE tf.data
# ============================================================

def _to_py(x):
    """Convierte tensores o arrays a tipo Python nativo."""
    if hasattr(x, "numpy"):
        x = x.numpy()
    if isinstance(x, np.ndarray):
        x = x.item()
    return x


def _mask_from_rects_abs_json_py(rects_json, W, H) -> np.ndarray:
    """
    Reconstruye la máscara binaria (H×W, uint8) a partir del JSON
    de rectángulos alterados en coordenadas absolutas.
    """
    rects_json = _to_py(rects_json)
    W = int(_to_py(W))
    H = int(_to_py(H))

    if rects_json is None:
        s = "[]"
    elif isinstance(rects_json, (bytes, bytearray, np.bytes_)):
        s = rects_json.decode("utf-8", errors="ignore")
    else:
        s = str(rects_json)

    s = s.strip() or "[]"

    try:
        rects = json.loads(s)
    except Exception:
        try:
            rects = ast.literal_eval(s)
        except Exception:
            rects = []

    mask = np.zeros((H, W), dtype=np.uint8)

    for r in rects:
        rx, ry = int(r.get("x", 0)), int(r.get("y", 0))
        rw, rh = int(r.get("w", 0)), int(r.get("h", 0))
        if rw <= 0 or rh <= 0:
            continue
        x0, y0 = max(0, rx), max(0, ry)
        x1, y1 = min(W, rx + rw), min(H, ry + rh)
        if x1 > x0 and y1 > y0:
            mask[y0:y1, x0:x1] = 1

    return mask


def _load_full_image_and_mask(img_path, label, mask_rects_abs):
    """
    Función de mapeo para tf.data:
      - Carga y normaliza la imagen a float32 [0,1]
      - Reconstruye la máscara binaria desde los rectángulos JSON
      - Redimensiona imagen y máscara a PATCH_SIZE × PATCH_SIZE
      - Devuelve (imagen, {"cls": label, "mask": máscara})
    """
    img_bytes = tf.io.read_file(img_path)
    img = tf.io.decode_image(img_bytes, channels=3, expand_animations=False)
    img = tf.image.convert_image_dtype(img, tf.float32)
    img.set_shape([None, None, 3])

    H = tf.shape(img)[0]
    W = tf.shape(img)[1]

    mask = tf.py_function(
        func=_mask_from_rects_abs_json_py,
        inp=[mask_rects_abs, W, H],
        Tout=tf.uint8,
    )
    mask.set_shape([None, None])
    mask = tf.cast(mask, tf.float32)[..., None]

    img_r  = tf.clip_by_value(
        tf.image.resize(img,  (config.PATCH_SIZE, config.PATCH_SIZE), method="bilinear"),
        0.0, 1.0,
    )
    mask_r = tf.cast(
        tf.image.resize(mask, (config.PATCH_SIZE, config.PATCH_SIZE), method="nearest") > 0.5,
        tf.float32,
    )

    label = tf.expand_dims(tf.cast(label, tf.float32), axis=-1)

    return img_r, {"cls": label, "mask": mask_r}


def make_dataset(
    df: pd.DataFrame,
    training: bool,
    seed: int,
    reshuffle_each_iteration: bool = False,
) -> tf.data.Dataset:
    """
    Convierte un DataFrame de anotaciones en un tf.data.Dataset
    listo para entrenamiento o evaluación.
    """
    df = df.copy()
    df["mask_rects_abs"] = df["mask_rects_abs"].fillna("[]").astype(str)

    paths  = df["img_path"].astype(str).values
    labels = df["label"].astype(np.int32).values
    rects  = df["mask_rects_abs"].values

    ds = tf.data.Dataset.from_tensor_slices((paths, labels, rects))
    ds = ds.map(_load_full_image_and_mask, num_parallel_calls=tf.data.AUTOTUNE)

    if training:
        ds = ds.shuffle(
            min(len(df), 6000),
            seed=seed,
            reshuffle_each_iteration=reshuffle_each_iteration,
        )

    return ds.batch(config.BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
