"""
evaluate_challenge_metrics.py — Re-evaluación con las métricas del DeepID Challenge.

Carga los modelos .pt ya entrenados del blind test y calcula las dos métricas
que usa el challenge para el ranking, sin reentrenar nada:

  1. F1 de detección a umbral FIJO 0.5  (Track 1 del challenge)
  2. F1 de localización per-image promediado  (Track 2 del challenge)
     → F1 calculado imagen a imagen, luego media ponderada bonafide/manipuladas

Estas métricas permiten la comparación directa con los equipos del DeepID
Challenge (ICCV 2025), en particular con AG/EdgeDoc (3er puesto), que es
el más comparable por usar los mismos datos de entrenamiento y arquitectura propia.

Referencia:
  Korshunov et al., "DeepID Challenge of Detecting Synthetic Manipulations
  in ID Documents", ICCVW 2025.
  - Track 1: F1 binary (threshold=0.5)
  - Track 2: per-image F1 pixel-wise, media ponderada bonafide/manipuladas

Uso:
    python evaluate_challenge_metrics.py

Salida:
    exports_hpo_pareto_nested/challenge_metrics.csv
    exports_hpo_pareto_nested/challenge_metrics_summary.csv
    Resumen por consola comparando con los resultados publicados del challenge.

Notas:
    - Solo procesa la variante 'multitask' (la principal del paper).
    - Para añadir variantes: modificar VARIANTS_TO_EVAL.
    - No requiere GPU (corre en CPU si no hay CUDA disponible), pero es más lento.
"""

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import f1_score, precision_recall_fscore_support
from tqdm import tqdm

import config
from dataset import build_full_doc_df, build_image_dataframe, add_json_paths, make_dataloader
from model import build_model
from train import get_device

warnings.filterwarnings("ignore")

# ============================================================
# CONFIGURACIÓN
# ============================================================

# Variantes a evaluar. Por defecto solo multitask.
# Añadir "cls_only", "seg_only", "unweighted_losses" si se quieren comparar.
VARIANTS_TO_EVAL = ["multitask"]

# Seeds a evaluar. Por defecto todas las seeds del blind test.
SEEDS_TO_EVAL = config.FINAL_SEEDS  # [42, 43, ..., 71]

# Umbral fijo del challenge (Track 1)
CHALLENGE_THR = 0.5

# Umbral de binarización de máscara (mismo que en blind test)
THR_MASK = config.THR_MASK

# Directorio de salida
OUTPUT_DIR = config.EXPORT_DIR
CHALLENGE_CSV         = OUTPUT_DIR / "challenge_metrics.csv"
CHALLENGE_SUMMARY_CSV = OUTPUT_DIR / "challenge_metrics_summary.csv"

# ============================================================
# RESULTADOS PUBLICADOS DEL CHALLENGE (para comparación)
# ============================================================
# Fuente: Tabla 2 y Tabla 3 de Korshunov et al., ICCVW 2025
# tf/tp = inference time (no relevante para comparación)
# F1f = F1 en FantasyID test set

CHALLENGE_RESULTS = {
    # (equipo, track): F1 en FantasyID
    "Sunlight_det":       0.991,
    "Incode_det":         0.868,
    "AG_det":             0.958,
    "UAM-Biometrics_det": 0.712,
    "Hardik_det":         0.839,
    "Reagvis_det":        0.816,
    "TruFor_baseline_det": 0.807,
    "Sunlight_loc":       0.784,
    "UAM-Biometrics_loc": 0.620,
    "VISION_loc":         0.612,
    "AG_loc":             0.686,
    "TruFor_baseline_loc": 0.590,
}


# ============================================================
# FUNCIÓN: F1 DE DETECCIÓN A UMBRAL FIJO (Track 1)
# ============================================================

def f1_detection_fixed_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
) -> dict:
    """
    Calcula el F1 de detección binaria con umbral fijo.

    El challenge usa threshold=0.5 sobre la probabilidad de clase 'attack' (label=1).
    Las etiquetas son: 0=bonafide, 1=attack.

    Devuelve dict con f1, precision, recall, f1_macro, n_bonafide, n_attack.
    """
    y_pred = (y_prob >= threshold).astype(int)

    # F1 binario de la clase positiva (attack=1), igual que el challenge
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", pos_label=1, zero_division=0
    )
    f1_macro = float(f1_score(y_true, y_pred, average="macro", zero_division=0))

    return {
        "det_f1_thr05":       float(f1),
        "det_precision_thr05": float(prec),
        "det_recall_thr05":   float(rec),
        "det_f1_macro_thr05": f1_macro,
        "det_n_bonafide":     int((y_true == 0).sum()),
        "det_n_attack":       int((y_true == 1).sum()),
        "det_n_total":        int(len(y_true)),
        "det_threshold":      float(threshold),
    }


# ============================================================
# FUNCIÓN: F1 DE LOCALIZACIÓN PER-IMAGE (Track 2)
# ============================================================

def f1_localization_per_image(
    per_image_results: list[dict],
) -> dict:
    """
    Calcula el F1 de localización per-image promediado, igual que el challenge.

    El challenge (Track 2) calcula:
      - Para cada imagen: F1 pixel-wise entre máscara predicha y máscara GT.
      - Para imágenes bonafide (GT toda ceros): F1 = 1.0 si la predicción también
        es toda ceros, 0.0 en caso contrario.
      - Media ponderada igualando el número de bonafide y manipuladas:
          loc_f1 = 0.5 * mean_f1_bonafide + 0.5 * mean_f1_attack

    Recibe una lista de dicts con campos:
      - label: int (0=bonafide, 1=attack)
      - f1_pixel: float (F1 pixel-wise de esa imagen)

    Devuelve dict con loc_f1, mean_f1_bonafide, mean_f1_attack, n_bonafide, n_attack.
    """
    f1_bonafide = [r["f1_pixel"] for r in per_image_results if r["label"] == 0]
    f1_attack   = [r["f1_pixel"] for r in per_image_results if r["label"] == 1]

    mean_bf = float(np.mean(f1_bonafide)) if f1_bonafide else float("nan")
    mean_at = float(np.mean(f1_attack))   if f1_attack   else float("nan")

    # Media ponderada igualitaria (igual que el challenge para corregir desbalance)
    loc_f1 = 0.5 * mean_bf + 0.5 * mean_at

    return {
        "loc_f1_perimage":       float(loc_f1),
        "loc_f1_bonafide_mean":  mean_bf,
        "loc_f1_attack_mean":    mean_at,
        "loc_n_bonafide":        len(f1_bonafide),
        "loc_n_attack":          len(f1_attack),
        "loc_n_total":           len(per_image_results),
    }


def _pixel_f1_single_image(
    gt_mask_flat:   np.ndarray,  # booleano (H*W,)
    pred_mask_flat: np.ndarray,  # booleano (H*W,)
    label:          int,
) -> float:
    """
    F1 pixel-wise para una sola imagen.

    Casos especiales:
      - Bonafide (label=0): GT es toda ceros. F1 = 1.0 si predicción también
        es toda ceros, 0.0 si predice algún píxel como manipulado.
      - Attack sin ningún píxel GT positivo (raro, pero posible): F1 = 0.0
        (la anotación debería tener positivos, pero si no los tiene, la
        predicción no puede ser correcta en el sentido de recall).
    """
    if label == 0:
        # Bonafide: correcto si no se predice ningún píxel alterado
        return 1.0 if pred_mask_flat.sum() == 0 else 0.0

    # Attack: F1 estándar entre GT y predicción a nivel de píxel
    tp = int((gt_mask_flat & pred_mask_flat).sum())
    fp = int((~gt_mask_flat & pred_mask_flat).sum())
    fn = int((gt_mask_flat & ~pred_mask_flat).sum())

    denom = 2 * tp + fp + fn
    if denom == 0:
        # GT vacío en ataque — caso patológico, F1 indefinido → 0.0
        return 0.0

    return float(2 * tp / denom)


# ============================================================
# EVALUACIÓN COMPLETA DE UN MODELO
# ============================================================

def evaluate_model_challenge(
    model:   nn.Module,
    loader:  torch.utils.data.DataLoader,
    device:  torch.device,
) -> dict:
    """
    Evalúa un modelo con las métricas del challenge.

    Devuelve dict con:
      - Métricas de detección (Track 1): f1 a umbral 0.5
      - Métricas de localización (Track 2): f1 per-image promediado
    """
    model.eval()

    y_true_cls  = []
    y_prob_cls  = []
    per_image   = []  # lista de dicts para Track 2

    with torch.no_grad():
        for imgs, labels, masks in loader:
            imgs   = imgs.to(device,   non_blocking=True)
            masks  = masks.to(device,  non_blocking=True)

            with torch.autocast(device_type=device.type,
                                enabled=(config.USE_AMP and device.type == "cuda")):
                out = model(imgs)

            # Clasificación
            probs_cls = torch.sigmoid(out["cls"]).float().cpu().numpy().reshape(-1)
            labs      = labels.cpu().numpy().reshape(-1).astype(int)
            y_true_cls.append(labs)
            y_prob_cls.append(probs_cls)

            # Segmentación per-image
            gt_masks   = (masks > 0.5).cpu().numpy()          # (B, 1, H, W) bool
            pred_masks = (out["mask"].float() > THR_MASK).cpu().numpy()  # (B, 1, H, W) bool

            for i in range(len(labs)):
                gt_flat   = gt_masks[i, 0].reshape(-1)
                pred_flat = pred_masks[i, 0].reshape(-1)
                f1_pix    = _pixel_f1_single_image(gt_flat, pred_flat, labs[i])
                per_image.append({"label": int(labs[i]), "f1_pixel": f1_pix})

    y_true = np.concatenate(y_true_cls)
    y_prob = np.concatenate(y_prob_cls)

    det_metrics = f1_detection_fixed_threshold(y_true, y_prob, threshold=CHALLENGE_THR)
    loc_metrics = f1_localization_per_image(per_image)

    return {**det_metrics, **loc_metrics}


# ============================================================
# RECONSTRUCCIÓN DEL HOLDOUT
# ============================================================

def load_holdout(seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Reconstruye el split holdout con la misma semilla que usó _train_final_model.
    Usa GroupShuffleSplit con test_size=0.15 y random_state=SEED_BASE (global),
    igual que load_and_prepare_data() en main.py.
    """
    from sklearn.model_selection import GroupShuffleSplit

    root    = config.DATASET_ROOT.resolve()
    df_imgs = build_image_dataframe(root)
    df_imgs = add_json_paths(df_imgs)
    df_imgs["label"] = (df_imgs["cls_dir"] == "attack").astype(int)
    groups  = df_imgs["stem"]

    gss = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=config.SEED_BASE)
    idx_rest, idx_test = next(gss.split(df_imgs, y=df_imgs["label"], groups=groups))

    df_holdout_base = df_imgs.iloc[idx_test].reset_index(drop=True)
    df_holdout      = build_full_doc_df(df_holdout_base, "test")

    df_dev_base = df_imgs.iloc[idx_rest].reset_index(drop=True)
    df_dev      = build_full_doc_df(df_dev_base, "dev")

    return df_dev, df_holdout


# ============================================================
# BUCLE PRINCIPAL
# ============================================================

def main():
    print("\n" + "=" * 65)
    print(" Re-evaluación con métricas del DeepID Challenge (ICCVW 2025)")
    print("=" * 65)
    print(f"  Variantes : {VARIANTS_TO_EVAL}")
    print(f"  Seeds     : {len(SEEDS_TO_EVAL)} seeds ({SEEDS_TO_EVAL[0]}..{SEEDS_TO_EVAL[-1]})")
    print(f"  Umbral det: {CHALLENGE_THR} (fijo, Track 1 del challenge)")
    print(f"  Umbral seg: {THR_MASK} (igual que blind test)")

    device = get_device()
    print(f"  Device    : {device}\n")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Cargar holdout una sola vez (es el mismo para todas las seeds/variantes)
    print("[INFO] Reconstruyendo split holdout...")
    _, df_holdout = load_holdout(seed=config.SEED_BASE)
    n_bf  = int((df_holdout["label"] == 0).sum())
    n_atk = int((df_holdout["label"] == 1).sum())
    print(f"[INFO] Holdout: {len(df_holdout)} imágenes | bonafide={n_bf} | attack={n_atk}\n")

    all_rows = []

    for variant in VARIANTS_TO_EVAL:
        print(f"\n{'─'*65}")
        print(f" Variante: {variant}")
        print(f"{'─'*65}")

        seed_rows = []

        for seed in tqdm(SEEDS_TO_EVAL, desc=f"  {variant}", unit="seed"):
            model_path = config.MODELS_DIR / f"model_{variant}_seed{seed}{config.RUN_TAG}.pt"

            if not model_path.exists():
                print(f"  [WARN] Modelo no encontrado: {model_path} — saltando")
                continue

            # Cargar checkpoint
            ckpt   = torch.load(model_path, map_location=device, weights_only=False)
            params = ckpt["params"]
            model  = build_model(params, device)
            model.load_state_dict(ckpt["state_dict"])
            model.to(device)
            model.eval()

            # DataLoader del holdout
            loader = make_dataloader(df_holdout, training=False, seed=seed, device=device)

            # Evaluar con métricas del challenge
            metrics = evaluate_model_challenge(model, loader, device)

            row = {
                "variant": variant,
                "seed":    seed,
                **metrics,
            }
            seed_rows.append(row)
            all_rows.append(row)

            del model, loader
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # Resumen por variante
        if seed_rows:
            df_var = pd.DataFrame(seed_rows)
            det_f1 = df_var["det_f1_thr05"]
            loc_f1 = df_var["loc_f1_perimage"]
            print(f"\n  [RESUMEN {variant}]")
            print(f"    Track 1 — F1 det (thr=0.5) : {det_f1.mean():.4f} ± {det_f1.std():.4f}")
            print(f"    Track 2 — F1 loc (per-img) : {loc_f1.mean():.4f} ± {loc_f1.std():.4f}")
            print(f"      ↳ F1 bonafide            : {df_var['loc_f1_bonafide_mean'].mean():.4f}")
            print(f"      ↳ F1 attack              : {df_var['loc_f1_attack_mean'].mean():.4f}")

    if not all_rows:
        print("\n[ERROR] No se encontró ningún modelo. Comprueba MODELS_DIR y RUN_TAG.")
        return

    # ── Guardar CSV completo ─────────────────────────────────────────────
    df_all = pd.DataFrame(all_rows)
    df_all.to_csv(CHALLENGE_CSV, index=False)
    print(f"\n[OK] CSV completo guardado: {CHALLENGE_CSV}")

    # ── Guardar CSV resumen ──────────────────────────────────────────────
    summary_rows = []
    for variant in df_all["variant"].unique():
        d = df_all[df_all["variant"] == variant]
        for col in ["det_f1_thr05", "det_f1_macro_thr05",
                    "loc_f1_perimage", "loc_f1_bonafide_mean", "loc_f1_attack_mean"]:
            summary_rows.append({
                "variant": variant,
                "metric":  col,
                "mean":    float(d[col].mean()),
                "std":     float(d[col].std()),
                "n_seeds": len(d),
            })
    df_summary = pd.DataFrame(summary_rows)
    df_summary.to_csv(CHALLENGE_SUMMARY_CSV, index=False)
    print(f"[OK] CSV resumen guardado : {CHALLENGE_SUMMARY_CSV}")

    # ── Tabla comparativa con el challenge ───────────────────────────────
    _print_comparison_table(df_all)


def _print_comparison_table(df_all: pd.DataFrame):
    """Imprime la tabla comparativa con los resultados publicados del challenge."""

    print("\n" + "=" * 65)
    print(" COMPARATIVA CON DEEPID CHALLENGE (FantasyID test set)")
    print("=" * 65)
    print(f"\n  Nota: el challenge usa F1 a umbral=0.5 fijo y F1 per-image.")
    print(f"  Nuestros resultados son sobre el HOLDOUT (15% del dataset),")
    print(f"  no sobre el test set oficial del challenge.\n")

    # Track 1 — Detección
    print(f"  {'Sistema':<28} {'F1 det (thr=0.5)':>17}  {'Datos entrenamiento'}")
    print("  " + "─" * 62)

    challenge_det = [
        ("Sunlight (1º)",      0.991, "60K+ imgs externas"),
        ("AG / EdgeDoc (3º)",  0.958, "Solo FantasyID"),
        ("Incode (2º)",        0.868, "100K imgs propietarias"),
        ("Hardik (5º)",        0.839, "Solo FantasyID (sin reentrenar)"),
        ("TruFor baseline",    0.807, "876K imgs preentrenado"),
        ("UAM-Biometrics (4º)",0.712, "Solo FantasyID (fine-tune TruFor)"),
    ]
    for name, f1, data in challenge_det:
        print(f"  {name:<28} {f1:>17.3f}  {data}")

    print("  " + "─" * 62)
    for variant in df_all["variant"].unique():
        d   = df_all[df_all["variant"] == variant]
        m   = d["det_f1_thr05"].mean()
        s   = d["det_f1_thr05"].std()
        tag = "← NUESTRO SISTEMA"
        print(f"  {'Nuestro (' + variant + ')':<28} {m:>7.4f}±{s:.4f}  Solo FantasyID  {tag}")

    # Track 2 — Localización
    print(f"\n  {'Sistema':<28} {'F1 loc (per-img)':>17}  {'Datos entrenamiento'}")
    print("  " + "─" * 62)

    challenge_loc = [
        ("Sunlight (1º)",      0.784, "60K+ imgs externas"),
        ("UAM-Biometrics (2º)",0.620, "Solo FantasyID (fine-tune TruFor)"),
        ("VISION (3º)",        0.612, "Solo FantasyID (TruFor 1024px)"),
        ("AG / EdgeDoc (4º)",  0.686, "Solo FantasyID"),
        ("TruFor baseline",    0.590, "876K imgs preentrenado"),
    ]
    for name, f1, data in challenge_loc:
        print(f"  {name:<28} {f1:>17.3f}  {data}")

    print("  " + "─" * 62)
    for variant in df_all["variant"].unique():
        d   = df_all[df_all["variant"] == variant]
        m   = d["loc_f1_perimage"].mean()
        s   = d["loc_f1_perimage"].std()
        tag = "← NUESTRO SISTEMA"
        print(f"  {'Nuestro (' + variant + ')':<28} {m:>7.4f}±{s:.4f}  Solo FantasyID  {tag}")

    print()
    print("  Aviso: comparación aproximada — el challenge evalúa sobre su")
    print("  test set oficial (subconjunto de FantasyID con ataques no vistos),")
    print("  nosotros evaluamos sobre nuestro holdout (split 85/15 del dataset")
    print("  completo). Para comparación exacta se necesita el test set oficial.")
    print()


if __name__ == "__main__":
    main()
