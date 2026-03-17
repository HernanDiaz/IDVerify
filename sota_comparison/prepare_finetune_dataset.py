"""
prepare_finetune_dataset.py — Prepara el dataset FantasyID para fine-tuning de TruFor.

Pasos:
  1. Replica el split exacto de DocVerify (GroupShuffleSplit seed=42) para excluir el holdout.
  2. Genera máscaras binarias PNG desde las anotaciones JSON (bounding boxes de regiones alteradas).
  3. Escribe los list-files TruFor en dataset/data/:
       FID_train_list.txt
       FID_valid_list.txt

Ejecutar ANTES de lanzar run_trufor_finetune.py.
Puede ejecutarse desde cualquier directorio — usa rutas absolutas.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import GroupShuffleSplit
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Rutas
# ---------------------------------------------------------------------------
_SCRIPT_DIR = Path(__file__).resolve().parent           # sota_comparison/
_PROJ_ROOT  = _SCRIPT_DIR.parent                        # DocVerify/
_FID_ROOT   = _PROJ_ROOT / "FantasyID"                  # FantasyID/
_TRUFOR     = _SCRIPT_DIR / "trufor_repo" / "TruFor_train_test"
_DATA_DIR   = _TRUFOR / "dataset" / "data"
_MASKS_DIR  = _SCRIPT_DIR / "fantasyid_masks"

SEED     = 42
IMG_EXTS = {".jpg", ".png"}

_DATA_DIR.mkdir(parents=True, exist_ok=True)
_MASKS_DIR.mkdir(parents=True, exist_ok=True)


# ===========================================================================
# 1. Indexar imágenes del dataset FantasyID
# ===========================================================================

def list_images(directory: Path) -> list:
    return sorted(
        p for p in directory.rglob("*")
        if p.is_file() and p.suffix.lower() in IMG_EXTS
    )


def parse_path_info(img_path: Path, root_dir: Path) -> dict:
    parts = img_path.relative_to(root_dir).parts
    split     = parts[0]   # 'train' | 'test'
    cls_dir   = parts[1]   # 'attack' | 'bonafide'
    if cls_dir == "attack":
        attack_type = parts[2]
        device      = parts[3]
    else:
        attack_type = None
        device      = parts[2]
    return {
        "split": split, "cls_dir": cls_dir,
        "attack_type": attack_type, "device": device,
        "img_path": str(img_path),
        "stem": img_path.stem,
    }


def build_dataframe(root: Path) -> pd.DataFrame:
    train_imgs = list_images(root / "train") if (root / "train").exists() else []
    test_imgs  = list_images(root / "test")  if (root / "test").exists()  else []
    all_imgs   = train_imgs + test_imgs
    rows = [parse_path_info(p, root) for p in all_imgs]
    df = pd.DataFrame(rows)
    df["label"] = (df["cls_dir"] == "attack").astype(int)
    print(f"[OK] Total imágenes indexadas: {len(df)}")
    print(df.groupby(["split", "cls_dir"]).size().to_string())
    return df


# ===========================================================================
# 2. Replicar el split DocVerify para excluir el holdout
# ===========================================================================

def get_finetune_splits(df: pd.DataFrame):
    """
    Reproduce el mismo GroupShuffleSplit de main.py (seed=42, test_size=0.15).
    Devuelve (df_train, df_val, df_holdout) con el holdout excluido del fine-tuning.
    """
    groups = df["stem"]

    # Split 1: separar holdout (15%)
    gss1 = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=SEED)
    idx_rest, idx_test = next(gss1.split(df, y=df["label"], groups=groups))
    df_rest    = df.iloc[idx_rest].reset_index(drop=True)
    df_holdout = df.iloc[idx_test].reset_index(drop=True)

    # Split 2: train/val del resto (75/10 del dataset original)
    val_ratio = 0.10 / 0.85
    gss2 = GroupShuffleSplit(n_splits=1, test_size=val_ratio, random_state=SEED)
    idx_train, idx_val = next(gss2.split(df_rest, y=df_rest["label"], groups=df_rest["stem"]))
    df_train = df_rest.iloc[idx_train].reset_index(drop=True)
    df_val   = df_rest.iloc[idx_val].reset_index(drop=True)

    # Sanity check: ningún stem del holdout en train/val
    holdout_stems = set(df_holdout["stem"])
    leaked = holdout_stems & (set(df_train["stem"]) | set(df_val["stem"]))
    assert not leaked, f"[ERROR] Holdout leakage: {len(leaked)} stems compartidos!"

    def stats(d):
        return (f"attack={int((d['label']==1).sum())}, "
                f"bonafide={int((d['label']==0).sum())}")

    print(f"\n[OK] Fine-tune splits (holdout excluido del entrenamiento):")
    print(f"  FID train : {len(df_train):4d} imgs  ({stats(df_train)})")
    print(f"  FID valid : {len(df_val):4d} imgs  ({stats(df_val)})")
    print(f"  Holdout   : {len(df_holdout):4d} imgs  ({stats(df_holdout)})  <- solo para evaluacion")
    print(f"  [CHECK] Sin data leakage OK")

    return df_train, df_val, df_holdout


# ===========================================================================
# 3. Generar máscaras binarias PNG desde anotaciones JSON
# ===========================================================================

def generate_mask_from_json(img_path: str, json_path: str) -> np.ndarray:
    """
    Lee el JSON de anotaciones y genera máscara binaria (H×W, uint8):
    - 1 en las regiones marcadas como 'altered'
    - 0 en el resto
    """
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    with Image.open(img_path) as img:
        W, H = img.size

    mask = np.zeros((H, W), dtype=np.uint8)

    for r in data.get("regions", []) or []:
        sa   = r.get("shape_attributes") or {}
        ra   = r.get("region_attributes") or {}
        if sa.get("name") != "rect":
            continue
        x = int(sa.get("x", 0));     y = int(sa.get("y", 0))
        w = int(sa.get("width", 0)); h = int(sa.get("height", 0))
        if w <= 0 or h <= 0:
            continue
        # Incluir solo regiones alteradas (o todas si no hay campo de proveniencia)
        prov = ra.get("region_provenance")
        if prov is not None and str(prov).strip().casefold() != "altered":
            continue
        x0, y0 = max(0, x),     max(0, y)
        x1, y1 = min(W, x + w), min(H, y + h)
        if x1 > x0 and y1 > y0:
            mask[y0:y1, x0:x1] = 1

    return mask


def get_mask_save_path(img_path_str: str) -> Path:
    """Devuelve la ruta donde se guardará la máscara PNG en _MASKS_DIR."""
    img_path = Path(img_path_str)
    try:
        rel = img_path.relative_to(_FID_ROOT)
    except ValueError:
        rel = Path(img_path.name)
    return _MASKS_DIR / rel.with_suffix(".png")


def generate_and_save_masks(df: pd.DataFrame) -> dict:
    """
    Genera y guarda máscaras PNG para las imágenes de ataque.
    Para bonafide devuelve 'None' (AbstractDataset generará máscara de ceros).
    Retorna: {img_path_str -> mask_path_str | 'None'}
    """
    result = {}
    attack_df   = df[df["label"] == 1]
    bonafide_df = df[df["label"] == 0]

    print(f"\n[Masks] Generando {len(attack_df)} máscaras de ataque...")
    for _, row in tqdm(attack_df.iterrows(), total=len(attack_df)):
        img_p  = row["img_path"]
        json_p = Path(img_p).with_suffix(".json")

        mask_out = get_mask_save_path(img_p)
        mask_out.parent.mkdir(parents=True, exist_ok=True)

        if json_p.exists():
            mask = generate_mask_from_json(img_p, str(json_p))
        else:
            with Image.open(img_p) as im:
                W, H = im.size
            mask = np.zeros((H, W), dtype=np.uint8)
            print(f"  [WARN] Sin JSON para imagen de ataque: {img_p}")

        Image.fromarray(mask * 255).save(str(mask_out))
        result[img_p] = str(mask_out)

    print(f"[OK] {len(attack_df)} máscaras guardadas en: {_MASKS_DIR}")

    for _, row in bonafide_df.iterrows():
        result[row["img_path"]] = "None"

    return result


# ===========================================================================
# 4. Escribir list-files TruFor
# ===========================================================================

def write_list_file(df: pd.DataFrame, mask_map: dict, out_path: Path):
    """
    Escribe el fichero de lista TruFor.
    Formato por línea: abs_img_path,abs_mask_path_or_None
    """
    lines = []
    for _, row in df.iterrows():
        img_p  = row["img_path"]
        mask_p = mask_map.get(img_p, "None")
        lines.append(f"{img_p},{mask_p}")

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    n_attack   = sum(1 for l in lines if not l.endswith(",None"))
    n_bonafide = sum(1 for l in lines if l.endswith(",None"))
    print(f"[OK] {out_path.name}: {len(lines)} entradas  "
          f"(attack={n_attack}, bonafide={n_bonafide})")


# ===========================================================================
# MAIN
# ===========================================================================

if __name__ == "__main__":
    print("=" * 60)
    print(" Preparación dataset FantasyID para TruFor fine-tuning")
    print("=" * 60)

    if not _FID_ROOT.exists():
        print(f"[ERROR] FantasyID no encontrado en: {_FID_ROOT}")
        sys.exit(1)

    # 1. Indexar todas las imágenes
    df = build_dataframe(_FID_ROOT)

    # 2. Replicar el split DocVerify
    df_train, df_val, df_holdout = get_finetune_splits(df)

    # 3. Generar máscaras solo para train+val (no tocar holdout)
    df_finetune = pd.concat([df_train, df_val], ignore_index=True)
    mask_map = generate_and_save_masks(df_finetune)

    # 4. Escribir list-files en TruFor/dataset/data/
    print(f"\n[List-files] Escribiendo en: {_DATA_DIR}")
    write_list_file(df_train, mask_map, _DATA_DIR / "FID_train_list.txt")
    write_list_file(df_val,   mask_map, _DATA_DIR / "FID_valid_list.txt")

    print("\n" + "=" * 60)
    print("[OK] Dataset preparado correctamente.")
    print(f"     Train list : {_DATA_DIR / 'FID_train_list.txt'}")
    print(f"     Valid list : {_DATA_DIR / 'FID_valid_list.txt'}")
    print(f"     Máscaras   : {_MASKS_DIR}")
    print("\nSiguiente paso: lanzar sota_trufor_finetune")
    print("=" * 60)
