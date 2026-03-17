"""
02_eval_comparison.py — Calcula métricas de TruFor sobre el holdout y genera
la tabla comparativa DocVerify vs TruFor.

Prerequisitos:
  - 00_export_holdout.py ejecutado (genera holdout_gt.csv)
  - 01_run_trufor.py ejecutado (genera trufor_scores.csv)
  - exports_hpo_pareto_nested/final_blind_test_multiseed.csv existente (resultados DocVerify)

Genera:
  sota_comparison/trufor_metrics.csv       — métricas TruFor sobre el holdout
  sota_comparison/comparison_table.csv     — tabla comparativa final
  sota_comparison/comparison_table.tex     — tabla LaTeX lista para incluir en el paper

Uso:
    cd E:/PycharmProjects/DocVerify
    python sota_comparison/02_eval_comparison.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import ast
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    f1_score,
    balanced_accuracy_score,
)

_THIS_DIR = Path(__file__).resolve().parent
_ROOT     = _THIS_DIR.parent
sys.path.insert(0, str(_ROOT))

# ============================================================
# RUTAS
# ============================================================

GT_CSV       = _THIS_DIR / "holdout_gt.csv"
SCORES_CSV   = _THIS_DIR / "trufor_scores.csv"
MASKS_DIR    = _THIS_DIR / "holdout_masks"
DOCVERIFY_CSV = _ROOT / "exports_hpo_pareto_nested" / "final_blind_test_multiseed.csv"

TRUFOR_METRICS_CSV = _THIS_DIR / "trufor_metrics.csv"
COMPARISON_CSV     = _THIS_DIR / "comparison_table.csv"
COMPARISON_TEX     = _THIS_DIR / "comparison_table.tex"


# ============================================================
# MÉTRICAS DE CLASIFICACIÓN
# ============================================================

def classification_metrics(
    y_true: np.ndarray,
    y_score: np.ndarray,
) -> dict:
    """
    Calcula PR-AUC, ROC-AUC, F1-macro óptimo (threshold sweep).
    Igual que evaluate.py para ser comparables con DocVerify.
    """
    y_true  = np.asarray(y_true, dtype=int)
    y_score = np.asarray(y_score, dtype=float)

    pr_auc  = float(average_precision_score(y_true, y_score))
    roc_auc = float(roc_auc_score(y_true, y_score))

    # Threshold sweep para F1-macro óptimo (igual que evaluate.py)
    thresholds = np.linspace(0.001, 1.0, 501)
    best_f1m, best_thr = 0.0, 0.5
    for thr in thresholds:
        yp  = (y_score >= thr).astype(int)
        f1m = float(f1_score(y_true, yp, average="macro", zero_division=0))
        if f1m > best_f1m:
            best_f1m, best_thr = f1m, float(thr)

    y_pred = (y_score >= best_thr).astype(int)
    bacc   = float(balanced_accuracy_score(y_true, y_pred))
    f1_bin = float(f1_score(y_true, y_pred, average="binary", zero_division=0))

    return {
        "pr_auc":  pr_auc,
        "roc_auc": roc_auc,
        "f1_macro_opt": best_f1m,
        "f1_binary_opt": f1_bin,
        "bacc_opt": bacc,
        "thr_opt": best_thr,
    }


# ============================================================
# MÉTRICAS DE SEGMENTACIÓN (Dice)
# ============================================================

def _load_gt_mask(stem: str) -> np.ndarray | None:
    """
    Carga la máscara GT desde holdout_masks/<stem>.npy.
    Las máscaras se guardan siempre con el stem ORIGINAL (Path(img_path).stem),
    no con unique_stem, porque se generan directamente desde img_path.
    """
    p = MASKS_DIR / f"{stem}.npy"
    if not p.exists():
        return None
    return np.load(p).astype(np.uint8)


def _load_pred_map(map_path: str) -> np.ndarray | None:
    """Carga el mapa de anomalía predicho por TruFor."""
    p = Path(map_path)
    if not p.exists() or map_path == "N/A":
        return None
    try:
        m = np.load(p).astype(np.float32)
        # Normalizar a [0,1] si no lo está
        if m.max() > 1.0:
            m = m / 255.0
        return m
    except Exception as e:
        print(f"  [WARN] No se pudo cargar {p}: {e}")
        return None


def segmentation_metrics(
    df_gt: pd.DataFrame,
    df_scores: pd.DataFrame,
    thr_mask: float = 0.5,
) -> dict:
    """
    Calcula Dice global y Dice medio en positivos.
    Solo en imágenes con máscara GT (label=1, mask_n_rects>0).

    Replicamos el cálculo de evaluate.py:
      - dice_global: 2*TP / (2*TP + FP + FN) sobre todos los píxeles del holdout
      - dice_pos_mean: media de Dice por imagen, solo en imágenes positivas con GT
    """
    # Merge GT + scores por unique_stem (nombre del archivo exportado a TruFor)
    # df_gt puede tener "unique_stem"; df_scores["stem"] = unique_stem del archivo exportado
    merge_key_gt = "unique_stem" if "unique_stem" in df_gt.columns else "stem"
    df = df_gt.merge(
        df_scores[["stem", "trufor_score", "map_npy_path"]].rename(
            columns={"stem": merge_key_gt}
        ),
        on=merge_key_gt, how="left"
    )

    TP = FP = TN = FN = 0
    dice_per_img = []
    n_missing_maps = 0

    for _, row in df.iterrows():
        gt_mask = _load_gt_mask(row["stem"])
        if gt_mask is None:
            continue

        pred_map = _load_pred_map(str(row.get("map_npy_path", "N/A")))

        if pred_map is None:
            n_missing_maps += 1
            # Sin mapa predicho → predecir todo cero
            pred_bin = np.zeros_like(gt_mask, dtype=np.uint8)
        else:
            # Redimensionar si es necesario
            if pred_map.shape != gt_mask.shape:
                from PIL import Image
                pred_pil = Image.fromarray((pred_map * 255).astype(np.uint8), mode="L")
                pred_pil = pred_pil.resize(
                    (gt_mask.shape[1], gt_mask.shape[0]), resample=Image.BILINEAR
                )
                pred_map = np.array(pred_pil).astype(np.float32) / 255.0
            pred_bin = (pred_map >= thr_mask).astype(np.uint8)

        gt_bin = (gt_mask > 0).astype(np.uint8)

        tp = int((gt_bin & pred_bin).sum())
        fp = int(((1 - gt_bin) & pred_bin).sum())
        tn = int(((1 - gt_bin) & (1 - pred_bin)).sum())
        fn = int((gt_bin & (1 - pred_bin)).sum())

        TP += tp; FP += fp; TN += tn; FN += fn

        # Dice por imagen — solo si hay píxeles positivos en GT
        if gt_bin.sum() > 0:
            denom = gt_bin.sum() + pred_bin.sum()
            dice  = (2 * tp + 1e-6) / (denom + 1e-6)
            dice_per_img.append(dice)

    dice_global  = float((2 * TP + 1e-6) / (2 * TP + FP + FN + 1e-6))
    dice_pos_mean = float(np.mean(dice_per_img)) if dice_per_img else float("nan")

    if n_missing_maps > 0:
        print(f"  [WARN] {n_missing_maps} imágenes sin mapa predicho (Dice=0 para ellas)")

    return {
        "dice_global":   dice_global,
        "dice_pos_mean": dice_pos_mean,
        "n_with_gt_mask": len(dice_per_img),
    }


# ============================================================
# DOCVERIFY: extraer métricas del CSV existente
# ============================================================

def load_docverify_metrics(variant: str = "multitask") -> dict:
    """
    Carga las métricas de DocVerify desde final_blind_test_multiseed.csv.
    Devuelve media ± std sobre las 30 seeds.
    """
    if not DOCVERIFY_CSV.exists():
        print(f"[ERROR] No se encontró {DOCVERIFY_CSV}")
        sys.exit(1)

    df = pd.read_csv(DOCVERIFY_CSV)
    df_v = df[df["variant"] == variant]

    if df_v.empty:
        print(f"[ERROR] Variant '{variant}' no encontrada. Disponibles: {df['variant'].unique()}")
        sys.exit(1)

    metrics = {}
    for col in ["test_pr_auc", "test_roc_auc", "test_dice_global", "test_dice_pos_mean",
                "test_f1_macro", "test_f1_1", "test_bacc"]:
        if col in df_v.columns:
            key = col.replace("test_", "")
            metrics[f"{key}_mean"] = float(df_v[col].mean())
            metrics[f"{key}_std"]  = float(df_v[col].std())

    metrics["n_seeds"] = len(df_v)
    metrics["variant"] = variant
    return metrics


# ============================================================
# TABLA COMPARATIVA
# ============================================================

def _fmt(mean: float, std: float, decimals: int = 4) -> str:
    """Formatea mean ± std."""
    if np.isnan(mean):
        return "N/A"
    return f"{mean:.{decimals}f} +/- {std:.{decimals}f}"


def _fmt_tex(mean: float, std: float, decimals: int = 3) -> str:
    """Formatea para LaTeX."""
    if np.isnan(mean):
        return r"\text{N/A}"
    m_str = f"{mean:.{decimals}f}"
    s_str = f"{std:.{decimals}f}"
    return f"{m_str} $\\pm$ {s_str}"


def build_comparison_table(
    docverify: dict,
    trufor_cls: dict,
    trufor_seg: dict,
) -> pd.DataFrame:
    """Construye el DataFrame de comparación final."""

    rows = [
        {
            "Method":       "TruFor (zero-shot)",
            "Training":     "NIST16 + others",
            "PR-AUC":       f"{trufor_cls['pr_auc']:.4f}",
            "ROC-AUC":      f"{trufor_cls['roc_auc']:.4f}",
            "F1-macro":     f"{trufor_cls['f1_macro_opt']:.4f}",
            "Dice (global)": f"{trufor_seg['dice_global']:.4f}",
            "Dice (pos)":   f"{trufor_seg['dice_pos_mean']:.4f}",
            "Test set":     "FantasyID holdout (N=493)",
            "Note":         "Pre-trained weights from authors",
        },
        {
            "Method":       "DocVerify (ours)",
            "Training":     "FantasyID",
            "PR-AUC":       _fmt(docverify.get("pr_auc_mean", float("nan")),
                                 docverify.get("pr_auc_std",  float("nan"))),
            "ROC-AUC":      _fmt(docverify.get("roc_auc_mean", float("nan")),
                                 docverify.get("roc_auc_std",  float("nan"))),
            "F1-macro":     _fmt(docverify.get("f1_macro_mean", float("nan")),
                                 docverify.get("f1_macro_std",  float("nan"))),
            "Dice (global)": _fmt(docverify.get("dice_global_mean", float("nan")),
                                  docverify.get("dice_global_std",  float("nan"))),
            "Dice (pos)":   _fmt(docverify.get("dice_pos_mean_mean", float("nan")),
                                 docverify.get("dice_pos_mean_std",  float("nan"))),
            "Test set":     "FantasyID holdout (N=493)",
            "Note":         f"Mean +/- std over {docverify.get('n_seeds', 30)} seeds",
        },
    ]
    return pd.DataFrame(rows)


def generate_latex_table(
    docverify: dict,
    trufor_cls: dict,
    trufor_seg: dict,
) -> str:
    """Genera una tabla LaTeX en formato IEEEtran."""

    def _b(v: float, v_ref: float, higher_better: bool = True) -> str:
        """Negrita si este valor es mejor que la referencia."""
        if np.isnan(v) or np.isnan(v_ref):
            return f"{v:.3f}"
        is_better = v > v_ref if higher_better else v < v_ref
        s = f"{v:.3f}"
        return f"\\textbf{{{s}}}" if is_better else s

    dv_prauc  = docverify.get("pr_auc_mean",  float("nan"))
    dv_rocauc = docverify.get("roc_auc_mean", float("nan"))
    dv_f1m    = docverify.get("f1_macro_mean", float("nan"))
    dv_dice_g = docverify.get("dice_global_mean", float("nan"))
    dv_dice_p = docverify.get("dice_pos_mean_mean", float("nan"))

    tf_prauc  = trufor_cls["pr_auc"]
    tf_rocauc = trufor_cls["roc_auc"]
    tf_f1m    = trufor_cls["f1_macro_opt"]
    tf_dice_g = trufor_seg["dice_global"]
    tf_dice_p = trufor_seg["dice_pos_mean"]

    dv_std_pr  = docverify.get("pr_auc_std",  0)
    dv_std_roc = docverify.get("roc_auc_std", 0)
    dv_std_f1  = docverify.get("f1_macro_std", 0)
    dv_std_dg  = docverify.get("dice_global_std", 0)
    dv_std_dp  = docverify.get("dice_pos_mean_std", 0)

    n_seeds = docverify.get("n_seeds", 30)

    tex = r"""\begin{table}[t]
\centering
\caption{Comparison with TruFor~\cite{trufor2023} on the FantasyID holdout set
($N=493$). TruFor is evaluated zero-shot using publicly released weights;
DocVerify is evaluated on the same images it was never trained on
(mean\,$\pm$\,std over """ + str(n_seeds) + r""" random seeds).
A direct comparison is possible only for the classification metrics;
anomaly-map localization is included for completeness but the two models
were optimized on different forgery types.}
\label{tab:sota_comparison}
\setlength{\tabcolsep}{5pt}
\begin{tabular}{lcccc}
\toprule
\textbf{Method} & \textbf{PR-AUC} & \textbf{ROC-AUC} & \textbf{F1-macro} & \textbf{Dice\textsubscript{global}} \\
\midrule
"""

    # TruFor row
    tex += (
        f"TruFor (zero-shot)~\\cite{{trufor2023}} & "
        f"{_b(tf_prauc, dv_prauc)} & "
        f"{_b(tf_rocauc, dv_rocauc)} & "
        f"{_b(tf_f1m, dv_f1m)} & "
        f"{_b(tf_dice_g, dv_dice_g)} \\\\\n"
    )

    # DocVerify row
    tex += (
        f"\\textbf{{DocVerify}} (ours) & "
        f"{_b(dv_prauc, tf_prauc)} $\\pm$ {dv_std_pr:.3f} & "
        f"{_b(dv_rocauc, tf_rocauc)} $\\pm$ {dv_std_roc:.3f} & "
        f"{_b(dv_f1m, tf_f1m)} $\\pm$ {dv_std_f1:.3f} & "
        f"{_b(dv_dice_g, tf_dice_g)} $\\pm$ {dv_std_dg:.3f} \\\\\n"
    )

    tex += r"""\bottomrule
\end{tabular}
\begin{tablenotes}
\small
\item TruFor~\cite{trufor2023} was not trained on FantasyID; its weights are publicly
available at \url{https://github.com/grip-unina/TruFor}. This comparison measures
the ability of a general-purpose forgery detector to generalize to identity
documents, relative to a domain-specific model trained on the same test distribution.
\end{tablenotes}
\end{table}
"""
    return tex


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 60)
    print(" DocVerify — Evaluacion TruFor + Tabla Comparativa")
    print("=" * 60)

    # Verificar archivos necesarios
    for path, desc in [
        (GT_CSV,       "holdout_gt.csv (ejecuta 00_export_holdout.py)"),
        (SCORES_CSV,   "trufor_scores.csv (ejecuta 01_run_trufor.py)"),
        (DOCVERIFY_CSV, "final_blind_test_multiseed.csv (resultados DocVerify)"),
    ]:
        if not path.exists():
            print(f"[ERROR] Falta: {path}")
            print(f"  --> {desc}")
            sys.exit(1)

    # 1. Cargar datos
    print("\n[1/4] Cargando datos...")
    df_gt     = pd.read_csv(GT_CSV)
    df_scores = pd.read_csv(SCORES_CSV)

    print(f"  Holdout GT    : {len(df_gt)} imagenes")
    print(f"  TruFor scores : {len(df_scores)} imagenes")

    # Merge por unique_stem (nombre del archivo exportado = clave de TruFor)
    # Si no existe unique_stem en el GT, usar stem directamente
    merge_key_gt = "unique_stem" if "unique_stem" in df_gt.columns else "stem"
    df_merged = df_gt.merge(
        df_scores[["stem", "trufor_score"]].rename(columns={"stem": merge_key_gt}),
        on=merge_key_gt, how="left"
    )
    n_missing = df_merged["trufor_score"].isna().sum()
    if n_missing > 0:
        print(f"  [WARN] {n_missing} imágenes sin score TruFor — se excluyen de las métricas")
        df_merged = df_merged.dropna(subset=["trufor_score"])

    y_true  = df_merged["label"].values.astype(int)
    y_score = df_merged["trufor_score"].values.astype(float)

    print(f"  Evaluando sobre {len(df_merged)} imagenes "
          f"({(y_true==1).sum()} attack, {(y_true==0).sum()} bonafide)")

    # Verificar distribución de scores
    print(f"  Score range: [{y_score.min():.4f}, {y_score.max():.4f}] "
          f"(media={y_score.mean():.4f})")

    # ADVERTENCIA: TruFor score 1 = forjado, 0 = auténtico — mismo convenio que DocVerify
    # Si la distribución está invertida (attacks con score bajo), invertir:
    if y_score[y_true == 1].mean() < y_score[y_true == 0].mean():
        print("\n  [NOTA] Los scores de TruFor parecen INVERTIDOS respecto a lo esperado.")
        print("         (attacks tienen score MENOR que bonafide)")
        print("         Invirtiendo scores: score_corregido = 1 - score_original")
        y_score = 1.0 - y_score

    # 2. Métricas de clasificación TruFor
    print("\n[2/4] Calculando metricas de clasificacion TruFor...")
    trufor_cls = classification_metrics(y_true, y_score)
    for k, v in trufor_cls.items():
        print(f"  {k:20s}: {v:.4f}")

    # 3. Métricas de segmentación TruFor
    print("\n[3/4] Calculando metricas de segmentacion TruFor...")
    has_maps = df_scores["map_npy_path"].ne("N/A").any()
    if has_maps:
        trufor_seg = segmentation_metrics(df_gt, df_scores)
    else:
        print("  [WARN] No hay mapas de anomalia disponibles. Dice = N/A.")
        trufor_seg = {"dice_global": float("nan"), "dice_pos_mean": float("nan"), "n_with_gt_mask": 0}

    for k, v in trufor_seg.items():
        if isinstance(v, float):
            print(f"  {k:20s}: {v:.4f}")
        else:
            print(f"  {k:20s}: {v}")

    # Guardar métricas TruFor detalladas
    trufor_all = {**trufor_cls, **trufor_seg, "n_total": len(df_merged),
                  "n_attack": int((y_true == 1).sum()), "n_bonafide": int((y_true == 0).sum())}
    pd.DataFrame([trufor_all]).to_csv(TRUFOR_METRICS_CSV, index=False)
    print(f"\n  Metricas TruFor guardadas en: {TRUFOR_METRICS_CSV}")

    # 4. Cargar métricas DocVerify
    print("\n[4/4] Cargando metricas DocVerify (multitask, 30 seeds)...")
    docverify = load_docverify_metrics(variant="multitask")
    print(f"  PR-AUC  : {docverify['pr_auc_mean']:.4f} +/- {docverify['pr_auc_std']:.4f}")
    print(f"  ROC-AUC : {docverify['roc_auc_mean']:.4f} +/- {docverify['roc_auc_std']:.4f}")
    print(f"  Dice(g) : {docverify['dice_global_mean']:.4f} +/- {docverify['dice_global_std']:.4f}")

    # 5. Generar tabla comparativa
    print("\n[5/5] Generando tabla comparativa...")
    df_table = build_comparison_table(docverify, trufor_cls, trufor_seg)
    df_table.to_csv(COMPARISON_CSV, index=False)
    print(f"\n  CSV guardado : {COMPARISON_CSV}")
    print("\n" + df_table.to_string(index=False))

    # 6. Generar tabla LaTeX
    tex = generate_latex_table(docverify, trufor_cls, trufor_seg)
    COMPARISON_TEX.write_text(tex, encoding="utf-8")
    print(f"\n  LaTeX guardado: {COMPARISON_TEX}")

    print("\n" + "=" * 60)
    print(" RESUMEN FINAL")
    print("=" * 60)
    print(f"  TruFor (zero-shot):")
    print(f"    PR-AUC  = {trufor_cls['pr_auc']:.4f}")
    print(f"    ROC-AUC = {trufor_cls['roc_auc']:.4f}")
    print(f"    F1-mac  = {trufor_cls['f1_macro_opt']:.4f} (thr={trufor_cls['thr_opt']:.3f})")
    print(f"    Dice(g) = {trufor_seg['dice_global']:.4f}")
    print(f"  DocVerify (multitask, mean over {docverify['n_seeds']} seeds):")
    print(f"    PR-AUC  = {docverify['pr_auc_mean']:.4f} +/- {docverify['pr_auc_std']:.4f}")
    print(f"    ROC-AUC = {docverify['roc_auc_mean']:.4f} +/- {docverify['roc_auc_std']:.4f}")
    print(f"    Dice(g) = {docverify['dice_global_mean']:.4f} +/- {docverify['dice_global_std']:.4f}")
    print()
    print("[OK] Tabla LaTeX lista para incluir en paper/sections/04_experiments.tex")
    print(f"     Ver: {COMPARISON_TEX}")


if __name__ == "__main__":
    main()
