"""
00_export_holdout.py — Exporta el holdout set de FantasyID al disco.

Reproduce EXACTAMENTE el mismo split de main.py (GroupShuffleSplit 85/15,
random_state=42, grupos por stem) para que TruFor se evalúe sobre las mismas
493 imágenes que DocVerify.

Genera:
  sota_comparison/holdout_gt.csv           — rutas + etiquetas + masks GT
  sota_comparison/holdout_images/bonafide/ — copias de las imágenes bonafide
  sota_comparison/holdout_images/attack/   — copias de las imágenes attack
  sota_comparison/holdout_masks/           — máscaras GT en formato .npy (HxW uint8)

Uso:
    cd E:/PycharmProjects/DocVerify
    python sota_comparison/00_export_holdout.py
    python sota_comparison/00_export_holdout.py --dataset_root ./FantasyID
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import ast
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import GroupShuffleSplit
from tqdm import tqdm

# Añadir el directorio raíz al path para importar módulos del proyecto
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

import config
from dataset import (
    add_json_paths,
    build_image_dataframe,
    build_full_doc_df,
)


# ============================================================
# SPLIT — réplica exacta de main.py:load_and_prepare_data()
# ============================================================

def reproduce_holdout_split(dataset_root: Path) -> pd.DataFrame:
    """
    Reproduce el split 85/15 de main.py y devuelve df_holdout.
    IMPORTANTE: usa los mismos parámetros que el experimento original.
    """
    root = dataset_root.resolve()
    assert root.exists(), f"Dataset no encontrado en: {root}"

    df_imgs = build_image_dataframe(root)
    df_imgs = add_json_paths(df_imgs)

    df_imgs["label"]  = (df_imgs["cls_dir"] == "attack").astype(int)
    groups            = df_imgs["stem"]

    # EXACTAMENTE los mismos parámetros que main.py
    gss = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=config.SEED_BASE)
    idx_rest, idx_test = next(gss.split(df_imgs, y=df_imgs["label"], groups=groups))

    df_test_base = df_imgs.iloc[idx_test].reset_index(drop=True)

    print(f"\n[OK] Holdout reproducido: {len(df_test_base)} imagenes")
    print(f"  bonafide : {(df_test_base['cls_dir']=='bonafide').sum()}")
    print(f"  attack   : {(df_test_base['cls_dir']=='attack').sum()}")

    # Parsear anotaciones JSON (genera mask_rects_abs)
    df_holdout = build_full_doc_df(df_test_base, "test")

    # Sanity check: no debe haber solapamiento con dev set
    df_rest = df_imgs.iloc[idx_rest]
    shared  = set(df_holdout["stem"]) & set(df_rest["stem"])
    assert not shared, f"[ERROR] Data leakage detectado: {len(shared)} stems compartidos"
    print("  [CHECK] Sin data leakage vs dev set OK")

    return df_holdout


# ============================================================
# GENERACIÓN DE MÁSCARAS GT
# ============================================================

def _mask_from_rects(rects_json: str, W: int, H: int) -> np.ndarray:
    s = (rects_json or "").strip() or "[]"
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


def build_gt_masks(df_holdout: pd.DataFrame, masks_dir: Path) -> None:
    """
    Genera una máscara GT en formato .npy por imagen.
    Nombre del archivo: <stem>.npy
    Dimensiones: (H, W) uint8 — 1=alterado, 0=original
    """
    masks_dir.mkdir(parents=True, exist_ok=True)
    n_with_mask = 0

    for _, row in tqdm(df_holdout.iterrows(), total=len(df_holdout), desc="Generando mascaras GT"):
        img = Image.open(row["img_path"])
        W, H = img.size
        img.close()

        mask = _mask_from_rects(str(row.get("mask_rects_abs", "[]")), W, H)
        stem = Path(row["img_path"]).stem
        np.save(masks_dir / f"{stem}.npy", mask)

        if mask.any():
            n_with_mask += 1

    print(f"  Mascaras generadas: {len(df_holdout)} total, {n_with_mask} con region alterada")


# ============================================================
# COPIA DE IMÁGENES
# ============================================================

def copy_images(df_holdout: pd.DataFrame, images_dir: Path,
                dataset_root: Path = None) -> pd.DataFrame:
    """
    Copia las imágenes al directorio de trabajo de TruFor con nombres únicos.

    Para evitar colisiones entre imágenes de distintos subdirectorios con el
    mismo filename, el nombre destino incorpora el subdirectorio de origen:
        <split>_<cls>_<attack_type_or_device>_<device>_<stem><ext>

    Si eso resulta demasiado largo, usa un hash MD5 de la ruta relativa.

    Estructura:
        holdout_images/
            bonafide/<unique_name>
            attack/<unique_name>

    Devuelve df_holdout con columnas 'exported_path' y 'unique_stem' añadidas.
    """
    (images_dir / "bonafide").mkdir(parents=True, exist_ok=True)
    (images_dir / "attack").mkdir(parents=True, exist_ok=True)

    if dataset_root is not None:
        dataset_root = dataset_root.resolve()

    exported_paths = []
    unique_stems   = []

    # Detectar si hay stems duplicados en el holdout
    stems = [Path(r["img_path"]).stem for _, r in df_holdout.iterrows()]
    has_duplicates = len(stems) != len(set(stems))

    for _, row in tqdm(df_holdout.iterrows(), total=len(df_holdout), desc="Copiando imagenes"):
        src    = Path(row["img_path"])
        subdir = "attack" if row["label"] == 1 else "bonafide"

        if has_duplicates and dataset_root is not None:
            # Nombre único: ruta relativa aplanada con guiones bajos
            try:
                rel = src.relative_to(dataset_root)
                parts = list(rel.parts)
                unique_name = "_".join(parts[:-1]) + "_" + src.name
                unique_name = unique_name.replace(" ", "_")
            except ValueError:
                unique_name = src.name
        else:
            unique_name = src.name

        unique_stem = Path(unique_name).stem
        dst = images_dir / subdir / unique_name

        if not dst.exists():
            shutil.copy2(src, dst)

        exported_paths.append(str(dst))
        unique_stems.append(unique_stem)

    if has_duplicates:
        print(f"  [WARN] Se detectaron stems duplicados en el holdout. "
              f"Usando nombres únicos (ruta aplanada).")

    df_out = df_holdout.copy()
    df_out["exported_path"] = exported_paths
    df_out["unique_stem"]   = unique_stems
    return df_out


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Exporta el holdout de FantasyID para comparacion SOTA"
    )
    parser.add_argument(
        "--dataset_root", type=Path,
        default=config.DATASET_ROOT,
        help="Ruta a FantasyID (default: config.DATASET_ROOT)"
    )
    parser.add_argument(
        "--out_dir", type=Path,
        default=Path(__file__).parent,
        help="Directorio de salida (default: sota_comparison/)"
    )
    args = parser.parse_args()

    out_dir     = args.out_dir.resolve()
    images_dir  = out_dir / "holdout_images"
    masks_dir   = out_dir / "holdout_masks"
    gt_csv_path = out_dir / "holdout_gt.csv"

    print("=" * 60)
    print(" DocVerify — Exportar Holdout para Comparacion SOTA")
    print("=" * 60)
    print(f"  Dataset root : {args.dataset_root.resolve()}")
    print(f"  Salida       : {out_dir}")

    # 1. Reproducir split
    df_holdout = reproduce_holdout_split(args.dataset_root)

    # 2. Generar máscaras GT
    print("\n[2/3] Generando mascaras ground-truth...")
    build_gt_masks(df_holdout, masks_dir)

    # 3. Copiar imágenes
    print("\n[3/3] Copiando imagenes al directorio de trabajo...")
    df_holdout = copy_images(df_holdout, images_dir, dataset_root=args.dataset_root)

    # 4. Guardar CSV de ground truth
    cols_to_save = ["stem", "unique_stem", "img_path", "exported_path", "label",
                    "mask_rects_abs", "mask_n_rects", "mask_area_px"]
    cols_to_save = [c for c in cols_to_save if c in df_holdout.columns]
    df_holdout[cols_to_save].to_csv(gt_csv_path, index=False)

    print(f"\n[OK] CSV de ground truth guardado: {gt_csv_path}")
    print(f"[OK] Imagenes en               : {images_dir}")
    print(f"[OK] Mascaras GT en            : {masks_dir}")
    print(f"\nResumen:")
    print(f"  Total     : {len(df_holdout)}")
    print(f"  Bonafide  : {(df_holdout['label']==0).sum()}")
    print(f"  Attack    : {(df_holdout['label']==1).sum()}")
    print(f"  Con mascara GT: {(df_holdout['mask_n_rects']>0).sum()}")
    print("\n[SIGUIENTE PASO] Ejecuta: python sota_comparison/01_run_trufor.py")


if __name__ == "__main__":
    main()
