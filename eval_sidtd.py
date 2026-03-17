"""
eval_sidtd.py — Evaluación zero-shot de DocVerify sobre el dataset SIDTD.

Carga los N modelos entrenados en FantasyID (outer folds 1-10, 30 seeds cada uno
o simplemente los mejores checkpoints disponibles) y los evalúa sobre SIDTD
sin ningún ajuste adicional (zero-shot generalization).

Uso:
    python eval_sidtd.py --models_dir ./exports_hpo_pareto_nested/models
                         --sidtd_root ./SIDTD
                         --output     ./exports_hpo_pareto_nested/sidtd_zeroshot.csv

Opciones:
    --models_dir   Carpeta con los checkpoints .pt (por defecto config.MODELS_DIR)
    --sidtd_root   Raíz de SIDTD (por defecto config.SIDTD_ROOT)
    --output       CSV de salida con métricas (por defecto EXPORT_DIR/sidtd_zeroshot.csv)
    --kind         Subconjunto de SIDTD: templates | clips | clips_cropped
                   (por defecto: templates)
    --batch_size   Tamaño de lote (por defecto config.BATCH_SIZE)
    --device       cuda | cpu | mps (por defecto: auto)
    --download     Si se pasa, descarga SIDTD automáticamente antes de evaluar
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

import config
from dataset import VRAMCache
from dataset_sidtd import build_sidtd_dataframe, download_sidtd
from evaluate import eval_model
from model import build_model


# ============================================================
# CARGA DE CHECKPOINTS
# ============================================================

def _find_checkpoints(models_dir: Path) -> list[Path]:
    """
    Devuelve todos los .pt de models_dir, ordenados alfabéticamente.
    Espera nombres como: model_outer1_seed0.pt, model_final_outer3.pt, etc.
    """
    ckpts = sorted(models_dir.glob("*.pt"))
    if not ckpts:
        raise FileNotFoundError(
            f"No se encontraron checkpoints .pt en {models_dir}.\n"
            "Asegúrate de que la ruta es correcta y que el entrenamiento ha finalizado."
        )
    return ckpts


def _load_model(ckpt_path: Path, device: torch.device) -> torch.nn.Module:
    """
    Carga un checkpoint y reconstruye el modelo.

    Los checkpoints guardados por train.py tienen el formato:
        {"model_state_dict": ..., "params": {...}, ...}
    o directamente state_dict si se guardó con torch.save(model.state_dict(), path).
    """
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)

    # Detectar formato del checkpoint
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        params      = checkpoint.get("params", {})
        state_dict  = checkpoint["model_state_dict"]
    elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        params      = checkpoint.get("params", {})
        state_dict  = checkpoint["state_dict"]
    else:
        # Asumimos que ES el state_dict directamente
        params     = {}
        state_dict = checkpoint

    model = build_model(params, device)
    model.load_state_dict(state_dict)
    model.eval()
    return model


# ============================================================
# EVALUACIÓN PRINCIPAL
# ============================================================

def evaluate_zero_shot(
    models_dir: Path,
    sidtd_root: Path,
    output_csv: Path,
    kind: str = "templates",
    device: torch.device = None,
) -> pd.DataFrame:
    """
    Evalúa todos los checkpoints de models_dir sobre SIDTD.

    Devuelve un DataFrame con una fila por checkpoint y las métricas:
        checkpoint, pr_auc, roc_auc, bacc, f1_binary, f1_macro,
        dice_global, dice_pos_mean, miou, n_samples
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[EvalSIDTD] Dispositivo: {device}")

    # 1. Construir DataFrame de SIDTD
    print("[EvalSIDTD] Cargando SIDTD …")
    df_sidtd = build_sidtd_dataframe(root=sidtd_root, kind=kind)

    # Asegurar columna mask_rects_abs compatible
    df_sidtd["mask_rects_abs"] = df_sidtd["mask_rects_abs"].fillna("[]").astype(str)

    # 2. Pre-cargar imágenes en VRAM/RAM (una sola vez para todos los modelos)
    print("[EvalSIDTD] Precargando imágenes …")
    cache  = VRAMCache(df_sidtd, device, label="SIDTD zero-shot")
    indices = np.arange(len(df_sidtd))
    loader  = cache.make_loader(indices, training=False, seed=0)

    # 3. Evaluar cada checkpoint
    ckpts = _find_checkpoints(models_dir)
    print(f"[EvalSIDTD] Evaluando {len(ckpts)} checkpoints …")

    rows = []
    for ckpt_path in tqdm(ckpts, desc="checkpoints"):
        model = _load_model(ckpt_path, device)

        with torch.no_grad():
            metrics = eval_model(model, loader, thr_cls=0.5, device=device)

        # Añadir información del checkpoint y del dataset
        row = {
            "checkpoint":   ckpt_path.name,
            "dataset":      "sidtd",
            "kind":         kind,
            "n_samples":    len(df_sidtd),
            "n_bonafide":   int((df_sidtd["label"] == 0).sum()),
            "n_attack":     int((df_sidtd["label"] == 1).sum()),
        }
        row.update(metrics)
        rows.append(row)

        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    cache.free()

    df_results = pd.DataFrame(rows)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df_results.to_csv(output_csv, index=False)
    print(f"\n[EvalSIDTD] Resultados guardados en {output_csv}")

    # Resumen
    print("\n--- Metricas agregadas (media +/- std sobre checkpoints) ---")
    metric_cols = ["pr_auc", "roc_auc", "bacc", "f1_binary", "dice_global", "dice_pos_mean"]
    for col in metric_cols:
        if col in df_results.columns:
            print(f"  {col:20s}: {df_results[col].mean():.4f} +/- {df_results[col].std():.4f}")

    return df_results


def evaluate_zero_shot_by_class(
    models_dir: Path,
    sidtd_root: Path,
    output_csv: Path,
    kind: str = "templates",
    device: torch.device = None,
) -> pd.DataFrame:
    """
    Como evaluate_zero_shot(), pero desglosa los resultados por clase de documento
    (alb, aze, est, fin, grc, ltu, rus, srb, svk, esp) y por tipo de ataque.
    Útil para analizar generalización por nacionalidad.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    df_sidtd = build_sidtd_dataframe(root=sidtd_root, kind=kind)
    df_sidtd["mask_rects_abs"] = df_sidtd["mask_rects_abs"].fillna("[]").astype(str)

    ckpts = _find_checkpoints(models_dir)
    # Usar el primer checkpoint como representativo (o promediar si hay varios)
    model = _load_model(ckpts[0], device)

    rows = []
    for cls in df_sidtd["sidtd_class"].unique():
        df_cls = df_sidtd[df_sidtd["sidtd_class"] == cls].reset_index(drop=True)
        if len(df_cls) == 0:
            continue

        cache   = VRAMCache(df_cls, device, label=f"SIDTD {cls}")
        indices = np.arange(len(df_cls))
        loader  = cache.make_loader(indices, training=False, seed=0)

        with torch.no_grad():
            metrics = eval_model(model, loader, thr_cls=0.5, device=device)

        row = {
            "sidtd_class": cls,
            "n_samples":   len(df_cls),
            "n_bonafide":  int((df_cls["label"] == 0).sum()),
            "n_attack":    int((df_cls["label"] == 1).sum()),
        }
        row.update(metrics)
        rows.append(row)
        cache.free()

    del model
    df_results = pd.DataFrame(rows)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df_results.to_csv(output_csv, index=False)
    print(f"[EvalSIDTD] Resultados por clase guardados en {output_csv}")
    return df_results


# ============================================================
# CLI
# ============================================================

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluación zero-shot de DocVerify sobre SIDTD"
    )
    parser.add_argument(
        "--models_dir",
        type=Path,
        default=config.MODELS_DIR,
        help="Carpeta con checkpoints .pt (default: config.MODELS_DIR)",
    )
    parser.add_argument(
        "--sidtd_root",
        type=Path,
        default=config.SIDTD_ROOT,
        help="Raíz del dataset SIDTD (default: config.SIDTD_ROOT)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=config.EXPORT_DIR / "sidtd_zeroshot.csv",
        help="CSV de salida con métricas",
    )
    parser.add_argument(
        "--kind",
        type=str,
        default="templates",
        choices=["templates", "clips", "clips_cropped"],
        help="Subconjunto de SIDTD a evaluar",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Dispositivo: cuda | cpu | mps (default: auto)",
    )
    parser.add_argument(
        "--download",
        action="store_true",
        help="Descargar SIDTD automáticamente si no está disponible",
    )
    parser.add_argument(
        "--by_class",
        action="store_true",
        help="Además de la evaluación global, desglosar por clase de documento",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    device = (
        torch.device(args.device)
        if args.device
        else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )

    if args.download:
        download_sidtd(root=args.sidtd_root)

    evaluate_zero_shot(
        models_dir=args.models_dir,
        sidtd_root=args.sidtd_root,
        output_csv=args.output,
        kind=args.kind,
        device=device,
    )

    if args.by_class:
        evaluate_zero_shot_by_class(
            models_dir=args.models_dir,
            sidtd_root=args.sidtd_root,
            output_csv=args.output.with_name(
                args.output.stem + "_by_class" + args.output.suffix
            ),
            kind=args.kind,
            device=device,
        )
