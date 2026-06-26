"""
pos_weight_sensitivity.py — Análisis de sensibilidad a la corrección de
desbalance de clases (revisión PRL, R2.2.3 / R2.1.4).

Pregunta que responde:
    El modelo se entrena con BCEWithLogitsLoss sin pos_weight (todas las
    muestras pesan igual). El dataset está desbalanceado 71.6% attack /
    28.4% bonafide, y la clase POSITIVA (label=1) es attack = la MAYORITARIA.
    ¿Corregir el desbalance vía pos_weight cambia PR-AUC, Dice o el recall
    por clase en el test ciego?

Diseño (no invasivo — no toca ningún resultado/peso del paper):
    - Reutiliza los final_params del paper (config Pareto seleccionada),
      leídos de las columnas hp_* de final_blind_test_multiseed.csv.
    - Reutiliza el MISMO procedimiento del blind test (_train_with_early_stopping
      sobre el holdout random_state=42, umbral barrido en val de desarrollo).
    - Solo varía pos_weight:
          · pw_correct = N_neg/N_pos  → corrige hacia la minoritaria (bonafide)
          · pw = 1.0                  → baseline (paper); se REUTILIZA de
                                        final_blind_test_multiseed.csv (30 seeds)
          · pw_naive   = N_pos/N_neg  → fórmula ingenua → agrava (sube mayoritaria)
    - 30 seeds (config.FINAL_SEEDS) por valor → comparación pareada por seed.

Salidas (en revision_experiments/results/pos_weight/):
    pos_weight_sensitivity.csv — una fila por (pos_weight, seed)
    pos_weight_summary.csv     — media±std por pos_weight (incl. recall por clase)
    pos_weight_stats.csv       — Wilcoxon + Cohen's d pareado vs baseline (Holm)

Uso:
    python revision_experiments/pos_weight_sensitivity.py

Requisitos previos:
    - final_blind_test_multiseed.csv debe existir (generado por main.py)
    - El dataset debe estar disponible en config.DATASET_ROOT
"""

import gc
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import GroupShuffleSplit

# Permitir ejecutar el script desde cualquier directorio
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
import evaluate as ev
from dataset import make_dataloader
from model import build_model, build_optimizer
from scalar_experiment import _load_dataset
from train import (
    _append_row_csv,
    _maybe_compile,
    _set_seeds,
    _train_with_early_stopping,
    get_device,
)

# ============================================================
# RUTAS DE SALIDA (carpeta dedicada de revisión)
# ============================================================
RESULTS_DIR = Path(__file__).resolve().parent / "results" / "pos_weight"
SENS_CSV    = RESULTS_DIR / "pos_weight_sensitivity.csv"
SUMMARY_CSV = RESULTS_DIR / "pos_weight_summary.csv"
STATS_CSV   = RESULTS_DIR / "pos_weight_stats.csv"

# Métricas reportadas en summary/stats
SUMMARY_METRICS = ["test_pr_auc", "test_dice_global", "test_bacc",
                   "recall_attack", "recall_bonafide"]
STAT_METRICS    = ["test_pr_auc", "test_dice_global", "test_bacc",
                   "recall_bonafide"]


# ============================================================
# HELPERS ESTADÍSTICOS
# ============================================================

def _cohens_d(a, b):
    d = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    return float(np.mean(d) / (np.std(d, ddof=1) + 1e-12))


def _holm_bonferroni(pvals, alpha=0.05):
    pvals  = np.asarray(pvals, dtype=float)
    m      = len(pvals)
    order  = np.argsort(pvals)
    adj    = np.empty_like(pvals)
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
# CARGA DE LA CONFIG FINAL DEL PAPER
# ============================================================

def _load_final_params() -> dict:
    """Lee los final_params (config Pareto) de las columnas hp_* del blind test."""
    if not config.FINAL_TEST_CSV.exists():
        raise FileNotFoundError(
            f"No se encuentra {config.FINAL_TEST_CSV}.\n"
            f"Ejecuta primero main.py para generar el blind test."
        )
    df = pd.read_csv(config.FINAL_TEST_CSV)
    if "variant" in df.columns:
        df = df[df["variant"] == "multitask"].reset_index(drop=True)
    hp_cols = [c for c in df.columns if c.startswith("hp_")]
    row0 = df.iloc[0]
    params = {c.replace("hp_", ""): row0[c] for c in hp_cols}
    params["dec_ch"]       = int(params["dec_ch"])
    params["dropout_rate"] = float(params["dropout_rate"])
    params["lr"]           = float(params["lr"])
    params["weight_decay"] = float(params["weight_decay"])
    params["loss_w_mask"]  = float(params["loss_w_mask"])
    if "alpha" in params:
        params["alpha"] = float(params["alpha"])
    return params


def _load_baseline_rows() -> pd.DataFrame:
    """Reutiliza las 30 filas del blind test del paper como baseline pos_weight=1."""
    df = pd.read_csv(config.FINAL_TEST_CSV)
    if "variant" in df.columns:
        df = df[df["variant"] == "multitask"].reset_index(drop=True)
    out = pd.DataFrame()
    out["seed"]             = df["seed"].astype(int)
    out["pos_weight"]       = 1.0
    out["test_pr_auc"]      = df["test_pr_auc"].astype(float)
    out["test_dice_global"] = df["test_dice_global"].astype(float)
    out["test_bacc"]        = df["test_bacc"].astype(float)
    rec_a, rec_b = [], []
    for cm in df["test_cm_TN_FP_FN_TP"]:
        tn, fp, fn, tp = json.loads(cm)
        rec_a.append(tp / (tp + fn) if (tp + fn) else float("nan"))
        rec_b.append(tn / (tn + fp) if (tn + fp) else float("nan"))
    out["recall_attack"]   = rec_a
    out["recall_bonafide"] = rec_b
    return out


# ============================================================
# ENTRENAMIENTO DE UN MODELO CON pos_weight DADO
# ============================================================

def _train_one(
    final_params: dict,
    pos_weight:   float,
    seed:         int,
    df_dev:       pd.DataFrame,
    df_holdout:   pd.DataFrame,
    device:       torch.device,
) -> dict:
    """
    Replica el procedimiento del blind test (_train_final_model) inyectando
    pos_weight. NO guarda pesos y escribe solo en la carpeta de revisión.
    """
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    _set_seeds(seed)

    # Split train/sel idéntico al blind test del paper (misma semilla por seed)
    gss = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=seed)
    tr_idx, sel_idx = next(gss.split(df_dev, y=df_dev["label"], groups=df_dev["stem"]))

    loader_tr   = make_dataloader(df_dev.iloc[tr_idx].reset_index(drop=True),
                                  training=True,  seed=seed, device=device)
    loader_sel  = make_dataloader(df_dev.iloc[sel_idx].reset_index(drop=True),
                                  training=False, seed=seed, device=device)
    loader_test = make_dataloader(df_holdout, training=False, seed=seed, device=device)

    run_params = dict(final_params)
    run_params["pos_weight"] = float(pos_weight)

    model     = _maybe_compile(build_model(run_params, device))
    optimizer = build_optimizer(model, run_params)
    scaler    = torch.amp.GradScaler("cuda",
                                     enabled=config.USE_AMP and device.type == "cuda")

    t0 = time.perf_counter()
    model = _train_with_early_stopping(
        model, optimizer, scaler, loader_tr, loader_sel, device,
        params=run_params, max_epochs=config.MAX_EPOCHS_FINAL,
        patience=12, variant="multitask",
        desc=f"[pw={pos_weight:.3f} seed={seed}]",
    )
    train_time = time.perf_counter() - t0

    # Umbral óptimo desde la validación de desarrollo (NUNCA del holdout)
    y_true_sel, y_prob_sel = [], []
    model.eval()
    with torch.no_grad():
        for imgs, labels, _ in loader_sel:
            if imgs.device != device:
                imgs = imgs.to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, enabled=config.USE_AMP):
                out = model(imgs)
            y_true_sel.append(labels.cpu().numpy().reshape(-1).astype(int))
            y_prob_sel.append(out["cls"].float().cpu().numpy().reshape(-1))
    thr_bacc, best_bacc, thr_f1, best_f1m = ev.threshold_sweep(
        np.concatenate(y_true_sel),
        torch.sigmoid(torch.tensor(np.concatenate(y_prob_sel))).numpy(),
    )

    met = ev.eval_model(model, loader_test, thr_cls=thr_bacc, device=device)
    tn, fp, fn, tp = met["cm_TN_FP_FN_TP"]

    row = {
        "pos_weight":            float(pos_weight),
        "seed":                  int(seed),
        "train_time_sec":        float(train_time),
        "thr_cls_from_val_sel":  float(thr_bacc),
        "recall_attack":         tp / (tp + fn) if (tp + fn) else float("nan"),
        "recall_bonafide":       tn / (tn + fp) if (tn + fp) else float("nan"),
        **{f"test_{k}": v for k, v in met.items()},
        **{f"hp_{k}": v for k, v in final_params.items()},
    }
    row["test_cm_TN_FP_FN_TP"] = json.dumps(met["cm_TN_FP_FN_TP"])

    del model, optimizer, loader_tr, loader_sel, loader_test
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return row


# ============================================================
# RESUMEN Y ESTADÍSTICA
# ============================================================

def _build_summary(df_all: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for pw in sorted(df_all["pos_weight"].unique()):
        d = df_all[df_all["pos_weight"] == pw]
        row = {"pos_weight": pw, "n_seeds": len(d)}
        for m in SUMMARY_METRICS:
            vals = d[m].dropna()
            row[f"{m}_mean"] = float(vals.mean())
            row[f"{m}_std"]  = float(vals.std(ddof=1)) if len(vals) > 1 else 0.0
        rows.append(row)
    return pd.DataFrame(rows)


def _run_stats(df_all: pd.DataFrame) -> pd.DataFrame:
    try:
        from scipy.stats import wilcoxon
    except ImportError:
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "scipy", "-q"])
        from scipy.stats import wilcoxon

    df_base = df_all[df_all["pos_weight"] == 1.0]
    stats_rows = []

    for pw in sorted(df_all["pos_weight"].unique()):
        if pw == 1.0:
            continue
        df_comp = df_all[df_all["pos_weight"] == pw]
        mrg = df_base.merge(df_comp, on="seed", suffixes=("_base", "_pw"))
        if len(mrg) < 3:
            print(f"[STATS] n insuficiente para pw={pw} vs baseline. Saltando.")
            continue

        pvals, tmp = [], []
        for m in STAT_METRICS:
            a = mrg[f"{m}_base"].values.astype(float)
            b = mrg[f"{m}_pw"].values.astype(float)
            try:
                p_w = float(wilcoxon(a, b, zero_method="wilcox").pvalue)
            except Exception:
                p_w = float("nan")
            pvals.append(p_w)
            tmp.append({
                "comparison":      f"pos_weight={pw:.3f} vs baseline(1.0)",
                "metric":          m,
                "n_paired":        len(mrg),
                "mean_base":       float(np.mean(a)),
                "mean_pw":         float(np.mean(b)),
                "delta":           float(np.mean(b) - np.mean(a)),
                "wilcoxon_p":      p_w,
                "cohens_d_paired": _cohens_d(b, a),
            })

        adj, rej = _holm_bonferroni(pvals)
        for i, r in enumerate(tmp):
            r["wilcoxon_p_holm"]      = float(adj[i])
            r["wilcoxon_reject_holm"] = bool(rej[i])
            stats_rows.append(r)

    return pd.DataFrame(stats_rows)


def _print_summary(df_summary: pd.DataFrame, df_stats: pd.DataFrame):
    print(f"\n{'='*64}")
    print(" RESUMEN — SENSIBILIDAD A pos_weight (media ± std, n seeds)")
    print(f"{'='*64}")
    for _, r in df_summary.iterrows():
        print(f"  pw={r['pos_weight']:.3f} (n={int(r['n_seeds'])}) | "
              f"PR-AUC={r['test_pr_auc_mean']:.4f}±{r['test_pr_auc_std']:.4f} | "
              f"Dice={r['test_dice_global_mean']:.4f}±{r['test_dice_global_std']:.4f} | "
              f"bACC={r['test_bacc_mean']:.4f} | "
              f"rec_attack={r['recall_attack_mean']:.4f} | "
              f"rec_bonafide={r['recall_bonafide_mean']:.4f}")

    if not df_stats.empty:
        print(f"\n{'='*64}")
        print(" ESTADÍSTICA — pareado vs baseline (Wilcoxon + Holm)")
        print(f"{'='*64}")
        for _, r in df_stats.sort_values(["comparison", "wilcoxon_p_holm"]).iterrows():
            print(f"  {r['comparison']} | {r['metric']} | "
                  f"Δ={r['delta']:+.4f} | p_holm={r['wilcoxon_p_holm']:.4g} | "
                  f"d={r['cohens_d_paired']:+.3f} | reject={r['wilcoxon_reject_holm']}")


# ============================================================
# MAIN
# ============================================================

def main():
    print("\n" + "=" * 64)
    print(" DocVerify — Sensibilidad a pos_weight (revisión R2.2.3 / R2.1.4)")
    print("=" * 64 + "\n")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    final_params = _load_final_params()
    print(f"[OK] final_params (paper): {final_params}")

    device = get_device()
    print("\n[INFO] Cargando dataset...")
    df_dev, df_holdout = _load_dataset()

    # pos_weight derivados de los recuentos de clase del set de desarrollo
    n_pos = int((df_dev["label"] == 1).sum())   # attack (mayoritaria)
    n_neg = int((df_dev["label"] == 0).sum())   # bonafide (minoritaria)
    pw_correct = n_neg / n_pos    # baja la mayoritaria → corrige hacia bonafide
    pw_naive   = n_pos / n_neg    # fórmula ingenua → agrava el desbalance
    run_pos_weights = [pw_correct, pw_naive]

    print(f"[OK] df_dev: attack(pos)={n_pos}  bonafide(neg)={n_neg}")
    print(f"[OK] pos_weight a ejecutar: correct={pw_correct:.4f}  naive={pw_naive:.4f}")
    print(f"[OK] baseline pos_weight=1.0 → reutilizado de {config.FINAL_TEST_CSV.name}")
    print(f"[OK] seeds: {config.FINAL_SEEDS}  (n={len(config.FINAL_SEEDS)})")

    # ── Entrenar los valores nuevos (baseline se reutiliza) ──────────────
    new_rows = []
    for pw in run_pos_weights:
        for seed in config.FINAL_SEEDS:
            print(f"\n  [RUN] pos_weight={pw:.4f} seed={seed}")
            row = _train_one(final_params, pw, seed, df_dev, df_holdout, device)
            _append_row_csv(SENS_CSV, row)
            new_rows.append(row)
            print(f"    PR-AUC={row['test_pr_auc']:.4f} | "
                  f"Dice={row['test_dice_global']:.4f} | "
                  f"rec_bonafide={row['recall_bonafide']:.4f}")

    # ── Combinar con el baseline reutilizado ─────────────────────────────
    df_new      = pd.DataFrame(new_rows)
    df_baseline = _load_baseline_rows()
    keep = ["pos_weight", "seed", *SUMMARY_METRICS]
    df_all = pd.concat([df_baseline[keep], df_new[keep]], ignore_index=True)

    # ── Resumen + estadística ────────────────────────────────────────────
    df_summary = _build_summary(df_all)
    df_summary.to_csv(SUMMARY_CSV, index=False)
    print(f"\n[OK] Resumen guardado: {SUMMARY_CSV}")

    df_stats = _run_stats(df_all)
    if not df_stats.empty:
        df_stats.to_csv(STATS_CSV, index=False)
        print(f"[OK] Estadística guardada: {STATS_CSV}")

    _print_summary(df_summary, df_stats)

    print(f"\n[OK] Experimento completado.")
    print(f"[OK] Resultados en: {RESULTS_DIR.resolve()}")


if __name__ == "__main__":
    main()
