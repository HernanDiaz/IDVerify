"""
sidtd_generalization.py — Generalizacion cross-domain a SIDTD/SSID (revision PRL).

Pregunta que responde:
    La data augmentation DEGRADA las metricas in-domain en FantasyID (ver
    augmentation_ablation.py: todas las metricas caen, Holm p<1e-7). La hipotesis
    aqui es si, a pesar de ello, los modelos entrenados CON augmentation
    GENERALIZAN MEJOR a un dominio distinto (SIDTD), es decir si la augmentation
    compra robustez cross-domain a costa de rendimiento in-domain.

Diseno (no invasivo — solo inferencia, no reentrena nada):
    - Condicion "no_aug" = baseline ya calculado: las 30 filas multitask
      (model_multitask_seedXX) de exports_hpo_pareto_nested/sidtd_zeroshot.csv,
      zero-shot sobre SIDTD/templates con umbral fijo 0.5. Se REUTILIZA.
    - Condicion "aug" = los 30 checkpoints augmentados
      (results/augment/checkpoints/model_aug_seedXX.pt) evaluados zero-shot sobre
      el MISMO SIDTD/templates con el MISMO protocolo (eval_sidtd.evaluate_zero_shot,
      thr=0.5). Se evaluan aqui.
    - Comparacion PAREADA por seed (Wilcoxon + Holm + Cohen's d).

Metricas:
    Titulares = PR-AUC y ROC-AUC (independientes de umbral → miden si el modelo
    RANKEA/generaliza mejor). BAcc/F1 a umbral fijo 0.5 son poco fiables
    cross-domain; se reportan con cautela. Dice mide localizacion cross-domain.

Salidas (en revision_experiments/results/augment/):
    sidtd_zeroshot_aug.csv — una fila por checkpoint aug (lo escribe eval_sidtd)
    sidtd_gen_summary.csv  — media+/-std por condicion (no_aug vs aug)
    sidtd_gen_stats.csv    — Wilcoxon + Cohen's d pareado (no_aug vs aug, Holm)

Uso:
    python revision_experiments/sidtd_generalization.py
    GEN_SMOKE=1 python revision_experiments/sidtd_generalization.py   # 3 seeds

Requisitos previos:
    - results/augment/checkpoints/model_aug_seedXX.pt (de augmentation_ablation.py)
    - exports_hpo_pareto_nested/sidtd_zeroshot.csv (baseline no_aug ya calculado)
    - SIDTD disponible en config.SIDTD_ROOT
"""

import os
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

# Permitir ejecutar el script desde cualquier directorio
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
from eval_sidtd import evaluate_zero_shot

# ============================================================
# KNOBS
# ============================================================
SMOKE = os.getenv("GEN_SMOKE", "0") == "1"

# ============================================================
# RUTAS
# ============================================================
AUG_DIR = Path(__file__).resolve().parent / "results" / "augment"
CKPT_DIR = AUG_DIR / "checkpoints"
AUG_ZS_CSV = AUG_DIR / "sidtd_zeroshot_aug.csv"
SUMMARY_CSV = AUG_DIR / "sidtd_gen_summary.csv"
STATS_CSV = AUG_DIR / "sidtd_gen_stats.csv"

BASELINE_CSV = config.EXPORT_DIR / "sidtd_zeroshot.csv"

# Metricas a comparar (titulares = pr_auc, roc_auc)
CMP_METRICS = ["pr_auc", "roc_auc", "bacc", "f1_1", "f1_macro", "dice_global"]

_SEED_RE = re.compile(r"seed(\d+)")


def _seed_from_name(name: str) -> int:
    m = _SEED_RE.search(str(name))
    if not m:
        raise ValueError(f"No se pudo extraer la seed de '{name}'")
    return int(m.group(1))


# ============================================================
# HELPERS ESTADISTICOS
# ============================================================

def _cohens_d(a, b):
    d = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    return float(np.mean(d) / (np.std(d, ddof=1) + 1e-12))


def _holm_bonferroni(pvals, alpha=0.05):
    pvals = np.asarray(pvals, dtype=float)
    m = len(pvals)
    order = np.argsort(pvals)
    adj = np.empty_like(pvals)
    reject = np.zeros(m, dtype=bool)
    for i, idx in enumerate(order):
        adj[idx] = min((m - i) * pvals[idx], 1.0)
    for i, idx in enumerate(order):
        if pvals[idx] <= alpha / (m - i):
            reject[idx] = True
        else:
            break
    return adj.tolist(), reject.tolist()


# ============================================================
# CARGA DE CONDICIONES
# ============================================================

def _load_baseline_rows() -> pd.DataFrame:
    """30 filas no_aug = multitask zero-shot ya calculado sobre SIDTD."""
    if not BASELINE_CSV.exists():
        raise FileNotFoundError(
            f"No se encuentra {BASELINE_CSV}.\n"
            f"Es el baseline no_aug (zero-shot multitask sobre SIDTD). "
            f"Ejecuta eval_sidtd.py primero si falta."
        )
    df = pd.read_csv(BASELINE_CSV)
    df = df[df["checkpoint"].str.contains("model_multitask_seed")].copy()
    df["seed"] = df["checkpoint"].map(_seed_from_name)
    df["condition"] = "no_aug"
    keep = ["seed", "condition", *CMP_METRICS]
    return df[keep].reset_index(drop=True)


def _load_aug_rows() -> pd.DataFrame:
    """30 filas aug = checkpoints augmentados zero-shot sobre SIDTD."""
    df = pd.read_csv(AUG_ZS_CSV)
    df = df[df["checkpoint"].str.contains("model_aug_seed")].copy()
    df["seed"] = df["checkpoint"].map(_seed_from_name)
    df["condition"] = "aug"
    keep = ["seed", "condition", *CMP_METRICS]
    return df[keep].reset_index(drop=True)


# ============================================================
# RESUMEN Y ESTADISTICA
# ============================================================

def _build_summary(df_all: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for cond in ["no_aug", "aug"]:
        d = df_all[df_all["condition"] == cond]
        if len(d) == 0:
            continue
        row = {"condition": cond, "n_seeds": len(d)}
        for m in CMP_METRICS:
            vals = d[m].dropna()
            row[f"{m}_mean"] = float(vals.mean())
            row[f"{m}_std"] = float(vals.std(ddof=1)) if len(vals) > 1 else 0.0
        rows.append(row)
    return pd.DataFrame(rows)


def _run_stats(df_all: pd.DataFrame) -> pd.DataFrame:
    try:
        from scipy.stats import wilcoxon
    except ImportError:
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "scipy", "-q"])
        from scipy.stats import wilcoxon

    df_base = df_all[df_all["condition"] == "no_aug"]
    df_aug = df_all[df_all["condition"] == "aug"]
    mrg = df_base.merge(df_aug, on="seed", suffixes=("_noaug", "_aug"))
    if len(mrg) < 3:
        print("[STATS] n insuficiente para comparar. Saltando.")
        return pd.DataFrame()

    pvals, tmp = [], []
    for m in CMP_METRICS:
        a = mrg[f"{m}_noaug"].values.astype(float)
        b = mrg[f"{m}_aug"].values.astype(float)
        try:
            p_w = float(wilcoxon(a, b, zero_method="wilcox").pvalue)
        except Exception:
            p_w = float("nan")
        pvals.append(p_w)
        tmp.append({
            "comparison": "no_aug vs aug",
            "metric": m,
            "n_paired": len(mrg),
            "mean_no_aug": float(np.mean(a)),
            "mean_aug": float(np.mean(b)),
            "delta_aug_minus_noaug": float(np.mean(b) - np.mean(a)),
            "wilcoxon_p": p_w,
            "cohens_d_paired": _cohens_d(b, a),
        })

    adj, rej = _holm_bonferroni(pvals)
    for i, r in enumerate(tmp):
        r["wilcoxon_p_holm"] = float(adj[i])
        r["wilcoxon_reject_holm"] = bool(rej[i])
    return pd.DataFrame(tmp)


def _print_summary(df_summary: pd.DataFrame, df_stats: pd.DataFrame):
    print("\n" + "=" * 70)
    print(" RESUMEN — Generalizacion cross-domain a SIDTD (media +/- std)")
    print("=" * 70)
    label = {"no_aug": "Sin augmentation", "aug": "Con augmentation"}
    for _, r in df_summary.iterrows():
        print(f"  [{label.get(r['condition'], r['condition']):>18}] (n={int(r['n_seeds'])}) "
              f"PR-AUC={r['pr_auc_mean']:.4f}+/-{r['pr_auc_std']:.4f} | "
              f"ROC-AUC={r['roc_auc_mean']:.4f}+/-{r['roc_auc_std']:.4f} | "
              f"BAcc={r['bacc_mean']:.4f} | Dice={r['dice_global_mean']:.4f}")

    if not df_stats.empty:
        print("\n" + "=" * 70)
        print(" ESTADISTICA — pareado no_aug vs aug (Wilcoxon + Holm)")
        print("=" * 70)
        for _, r in df_stats.iterrows():
            print(f"  {r['metric']:>12} | Delta={r['delta_aug_minus_noaug']:+.4f} | "
                  f"p_holm={r['wilcoxon_p_holm']:.4g} | d={r['cohens_d_paired']:+.3f} | "
                  f"reject={r['wilcoxon_reject_holm']}")


# ============================================================
# MAIN
# ============================================================

def main():
    print("\n" + "=" * 70)
    print(" DocVerify — Generalizacion cross-domain a SIDTD (aug vs no_aug)")
    print("=" * 70 + "\n")

    if not CKPT_DIR.exists() or not list(CKPT_DIR.glob("model_aug_seed*.pt")):
        raise FileNotFoundError(
            f"No hay checkpoints aug en {CKPT_DIR}.\n"
            f"Ejecuta primero augmentation_ablation.py (guarda los .pt)."
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Dispositivo: {device}")
    print(f"[INFO] Checkpoints aug: {CKPT_DIR}")
    print(f"[INFO] Baseline no_aug: {BASELINE_CSV}")
    print(f"[INFO] SIDTD root: {config.SIDTD_ROOT}\n")

    # ── 1. Evaluar los checkpoints aug zero-shot sobre SIDTD/templates ──────
    print("[STEP 1] Evaluando checkpoints aug zero-shot sobre SIDTD/templates...")
    evaluate_zero_shot(
        models_dir=CKPT_DIR,
        sidtd_root=config.SIDTD_ROOT,
        output_csv=AUG_ZS_CSV,
        kind="templates",
        device=device,
    )

    # ── 2. Cargar ambas condiciones y combinar ─────────────────────────────
    df_aug = _load_aug_rows()
    df_base = _load_baseline_rows()
    if SMOKE:
        keep_seeds = set(df_aug["seed"].tolist())
        df_base = df_base[df_base["seed"].isin(keep_seeds)].reset_index(drop=True)
    df_all = pd.concat([df_base, df_aug], ignore_index=True)
    print(f"\n[OK] no_aug n={len(df_base)} | aug n={len(df_aug)}")

    # ── 3. Resumen + estadistica ───────────────────────────────────────────
    df_summary = _build_summary(df_all)
    df_summary.to_csv(SUMMARY_CSV, index=False)
    print(f"[OK] Resumen guardado: {SUMMARY_CSV}")

    df_stats = _run_stats(df_all)
    if not df_stats.empty:
        df_stats.to_csv(STATS_CSV, index=False)
        print(f"[OK] Estadistica guardada: {STATS_CSV}")

    _print_summary(df_summary, df_stats)

    print(f"\n[OK] Experimento completado.")
    print(f"[OK] Resultados en: {AUG_DIR.resolve()}")


if __name__ == "__main__":
    main()
