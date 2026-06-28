"""
pareto_vs_scalar_blindtest.py — Pareto vs escalarizacion a 30 seeds (revision PRL, R1.5 / R2.2.4).

Pregunta que responde:
    La Table 4 del paper compara la seleccion Pareto frente a reglas escalares
    (by_prauc, by_dice) usando los 10 outer folds del nested CV (n=10), de baja
    potencia estadistica. El revisor pide >=15 (preferiblemente 30) repeticiones
    por condicion, con intervalos de confianza y tamanos de efecto. Este script
    eleva la comparacion al MISMO protocolo de 30 seeds del blind test del paper.

Diseno (NO invasivo — no reentrena ni sobrescribe nada del paper):
    - Condicion "pareto" = el blind test del paper (final_blind_test_multiseed.csv,
      variant=multitask, 30 seeds) → se REUTILIZA tal cual.
    - Condiciones "by_prauc" y "by_dice" = se ENTRENAN de nuevo, con el MISMO
      procedimiento del blind test (_train_with_early_stopping, umbral barrido en
      val de desarrollo, evaluacion en el mismo holdout), pero con la config que
      cada regla escalar selecciona a nivel dev (sin tocar el holdout):
          · by_prauc = trial que MAXIMIZA val_cls_prauc (objetivo de validacion interna)
          · by_dice  = trial que MAXIMIZA val_mask_dice_global
      Las configs se eligen por argmax sobre optuna_trials_nested.csv (determinista).
    - 30 seeds (config.FINAL_SEEDS) → comparacion pareada por seed contra Pareto.

Reproducibilidad:
    - Configs seleccionadas por argmax determinista sobre el CSV de trials.
    - Seeds fijos (config.FINAL_SEEDS) + split GroupShuffleSplit(random_state=seed)
      identicos al blind test → regeneracion determinista. (No se guardan pesos:
      la reproducibilidad la dan los seeds, igual que en resnet18_motpe.py.)

Estadistica:
    Pareado por seed, pareto vs cada regla escalar:
    Wilcoxon + Cohen's d + IC 95% de la diferencia media; correccion Holm sobre
    todos los contrastes (2 comparaciones x 5 metricas).

Salidas (en revision_experiments/results/scalar30/):
    scalar30_blind_test.csv — una fila por (seed, condicion) de by_prauc/by_dice
    scalar30_summary.csv     — media+/-std e IC95% por condicion (pareto reutilizado)
    scalar30_stats.csv       — Wilcoxon + Cohen's d + IC95% diff (pareto vs escalar)

Uso:
    python revision_experiments/pareto_vs_scalar_blindtest.py
    SCALAR30_SMOKE=1 python revision_experiments/pareto_vs_scalar_blindtest.py  # 2 seeds

Requisitos previos:
    - final_blind_test_multiseed.csv (blind test del paper, baseline pareto)
    - optuna_trials_nested.csv (pool de trials con objetivos de validacion)
    - dataset disponible en config.DATASET_ROOT
"""

import gc
import json
import os
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
# KNOBS DE SMOKE TEST
# ============================================================
SMOKE = os.getenv("SCALAR30_SMOKE", "0") == "1"
N_SEEDS = 2 if SMOKE else config.N_FINAL_SEEDS
MAX_EPOCHS_FINAL = 5 if SMOKE else config.MAX_EPOCHS_FINAL
SEEDS = config.FINAL_SEEDS[:N_SEEDS]

# ============================================================
# RUTAS DE SALIDA (carpeta dedicada de revision)
# ============================================================
RESULTS_DIR = Path(__file__).resolve().parent / "results" / "scalar30"
BLIND_CSV = RESULTS_DIR / "scalar30_blind_test.csv"
SUMMARY_CSV = RESULTS_DIR / "scalar30_summary.csv"
STATS_CSV = RESULTS_DIR / "scalar30_stats.csv"

TRIALS_CSV = config.EXPORT_DIR / "optuna_trials_nested.csv"

# Las 5 hp tuneadas que varian entre condiciones (mismas que el blind test)
HP_KEYS = ["lr", "weight_decay", "dropout_rate", "dec_ch", "loss_w_mask"]

CMP_METRICS = ["test_pr_auc", "test_dice_global", "test_bacc",
               "test_f1_1", "test_f1_macro"]

# Condiciones escalares a entrenar (pareto se reutiliza del blind test)
SCALAR_RULES = {
    "by_prauc": "val_cls_prauc",
    "by_dice": "val_mask_dice_global",
}


# ============================================================
# HELPERS ESTADISTICOS
# ============================================================

def _cohens_d(a, b):
    d = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    return float(np.mean(d) / (np.std(d, ddof=1) + 1e-12))


def _ci95_mean(x):
    """IC 95% de la media (t de Student)."""
    x = np.asarray(x, dtype=float)
    n = len(x)
    if n < 2:
        return (float("nan"), float("nan"))
    from scipy.stats import t as tdist
    m = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n))
    h = se * float(tdist.ppf(0.975, n - 1))
    return (m - h, m + h)


def _ci95_diff(a, b):
    """IC 95% de la diferencia media pareada (a-b)."""
    return _ci95_mean(np.asarray(a, dtype=float) - np.asarray(b, dtype=float))


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
# SELECCION DE CONFIGS ESCALARES (nivel dev, determinista)
# ============================================================

def _select_scalar_configs() -> dict:
    """
    Elige by_prauc y by_dice por argmax del objetivo de validacion interna sobre
    el pool de trials del HPO. Devuelve {regla: {hp...}}. Determinista.
    """
    if not TRIALS_CSV.exists():
        raise FileNotFoundError(
            f"No se encuentra {TRIALS_CSV}.\n"
            f"Es el pool de trials del HPO (objetivos de validacion por trial)."
        )
    t = pd.read_csv(TRIALS_CSV)
    if "pruned" in t.columns:
        t = t[t["pruned"] == False]  # noqa: E712
    t = t.dropna(subset=list(SCALAR_RULES.values())).reset_index(drop=True)

    configs = {}
    for rule, obj_col in SCALAR_RULES.items():
        r = t.loc[t[obj_col].idxmax()]
        cfg = {
            "lr": float(r["lr"]),
            "weight_decay": float(r["weight_decay"]),
            "dropout_rate": float(r["dropout_rate"]),
            "dec_ch": int(r["dec_ch"]),
            "loss_w_mask": float(r["loss_w_mask"]),
        }
        cfg["_src"] = {
            "outer_fold": int(r["outer_fold"]),
            "trial_number": int(r["trial_number"]),
            "val_cls_prauc": float(r["val_cls_prauc"]),
            "val_mask_dice_global": float(r["val_mask_dice_global"]),
        }
        configs[rule] = cfg
    return configs


# ============================================================
# CARGA DEL BASELINE PARETO (blind test del paper)
# ============================================================

def _load_pareto_baseline_rows() -> pd.DataFrame:
    """Reutiliza las 30 filas multitask del blind test del paper como condicion pareto."""
    if not config.FINAL_TEST_CSV.exists():
        raise FileNotFoundError(
            f"No se encuentra {config.FINAL_TEST_CSV}.\n"
            f"Ejecuta primero main.py para generar el blind test."
        )
    df = pd.read_csv(config.FINAL_TEST_CSV)
    if "variant" in df.columns:
        df = df[df["variant"] == "multitask"].reset_index(drop=True)
    out = pd.DataFrame()
    out["seed"] = df["seed"].astype(int)
    out["condition"] = "pareto"
    for m in CMP_METRICS:
        out[m] = df[m].astype(float)
    return out


# ============================================================
# ENTRENAMIENTO DE UN MODELO CON UNA CONFIG ESCALAR
# ============================================================

def _train_one(cfg: dict, condition: str, seed: int, df_dev: pd.DataFrame,
               df_holdout: pd.DataFrame, device: torch.device) -> dict:
    """
    Replica EXACTAMENTE el procedimiento del blind test del paper (_train_final_model)
    con la config escalar dada. Sin augmentation, sin guardar pesos.
    """
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    _set_seeds(seed)

    # Split train/sel identico al blind test del paper (misma semilla por seed)
    gss = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=seed)
    tr_idx, sel_idx = next(gss.split(df_dev, y=df_dev["label"], groups=df_dev["stem"]))

    loader_tr = make_dataloader(df_dev.iloc[tr_idx].reset_index(drop=True),
                                training=True, seed=seed, device=device)
    loader_sel = make_dataloader(df_dev.iloc[sel_idx].reset_index(drop=True),
                                 training=False, seed=seed, device=device)
    loader_test = make_dataloader(df_holdout, training=False, seed=seed, device=device)

    run_params = {k: cfg[k] for k in HP_KEYS}
    model = _maybe_compile(build_model(run_params, device))
    optimizer = build_optimizer(model, run_params)
    scaler = torch.amp.GradScaler("cuda",
                                  enabled=config.USE_AMP and device.type == "cuda")

    t0 = time.perf_counter()
    model = _train_with_early_stopping(
        model, optimizer, scaler, loader_tr, loader_sel, device,
        params=run_params, max_epochs=MAX_EPOCHS_FINAL,
        patience=12, variant="multitask",
        desc=f"[{condition} seed={seed}]",
    )
    train_time = time.perf_counter() - t0

    # Umbral optimo desde la validacion de desarrollo (NUNCA del holdout)
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
    thr_bacc, _bb, _tf1, _bf1 = ev.threshold_sweep(
        np.concatenate(y_true_sel),
        torch.sigmoid(torch.tensor(np.concatenate(y_prob_sel))).numpy(),
    )

    met = ev.eval_model(model, loader_test, thr_cls=thr_bacc, device=device)

    row = {
        "seed": int(seed),
        "condition": condition,
        "train_time_sec": float(train_time),
        "thr_cls_from_val_sel": float(thr_bacc),
        **{f"test_{k}": v for k, v in met.items()},
        **{f"hp_{k}": cfg[k] for k in HP_KEYS},
    }
    row["test_cm_TN_FP_FN_TP"] = json.dumps(met["cm_TN_FP_FN_TP"])

    del model, optimizer, loader_tr, loader_sel, loader_test
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return row


# ============================================================
# RESUMEN Y ESTADISTICA
# ============================================================

def _build_summary(df_all: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for cond in ["pareto", "by_prauc", "by_dice"]:
        d = df_all[df_all["condition"] == cond]
        if len(d) == 0:
            continue
        row = {"condition": cond, "n_seeds": len(d)}
        for m in CMP_METRICS:
            vals = d[m].dropna()
            lo, hi = _ci95_mean(vals)
            row[f"{m}_mean"] = float(vals.mean())
            row[f"{m}_std"] = float(vals.std(ddof=1)) if len(vals) > 1 else 0.0
            row[f"{m}_ci95_lo"] = lo
            row[f"{m}_ci95_hi"] = hi
        rows.append(row)
    return pd.DataFrame(rows)


def _run_stats(df_all: pd.DataFrame) -> pd.DataFrame:
    try:
        from scipy.stats import wilcoxon
    except ImportError:
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "scipy", "-q"])
        from scipy.stats import wilcoxon

    df_par = df_all[df_all["condition"] == "pareto"]
    pvals, tmp = [], []
    for rule in ["by_prauc", "by_dice"]:
        df_sc = df_all[df_all["condition"] == rule]
        mrg = df_par.merge(df_sc, on="seed", suffixes=("_par", "_sc"))
        if len(mrg) < 3:
            print(f"[STATS] n insuficiente para pareto vs {rule}. Saltando.")
            continue
        for m in CMP_METRICS:
            a = mrg[f"{m}_par"].values.astype(float)   # pareto
            b = mrg[f"{m}_sc"].values.astype(float)    # escalar
            try:
                p_w = float(wilcoxon(a, b, zero_method="wilcox").pvalue)
            except Exception:
                p_w = float("nan")
            lo, hi = _ci95_diff(a, b)  # pareto - escalar
            pvals.append(p_w)
            tmp.append({
                "comparison": f"pareto vs {rule}",
                "metric": m,
                "n_paired": len(mrg),
                "mean_pareto": float(np.mean(a)),
                "mean_scalar": float(np.mean(b)),
                "delta_pareto_minus_scalar": float(np.mean(a) - np.mean(b)),
                "diff_ci95_lo": lo,
                "diff_ci95_hi": hi,
                "wilcoxon_p": p_w,
                "cohens_d_paired": _cohens_d(a, b),
            })

    if not tmp:
        return pd.DataFrame()
    adj, rej = _holm_bonferroni(pvals)
    for i, r in enumerate(tmp):
        r["wilcoxon_p_holm"] = float(adj[i])
        r["wilcoxon_reject_holm"] = bool(rej[i])
    return pd.DataFrame(tmp)


def _print_summary(df_summary: pd.DataFrame, df_stats: pd.DataFrame):
    print("\n" + "=" * 70)
    print(" RESUMEN — Pareto vs escalar a 30 seeds (media +/- std [IC95%])")
    print("=" * 70)
    label = {"pareto": "Pareto (ours)", "by_prauc": "by_prauc", "by_dice": "by_dice"}
    for _, r in df_summary.iterrows():
        print(f"  [{label.get(r['condition'], r['condition']):>14}] (n={int(r['n_seeds'])}) "
              f"PR-AUC={r['test_pr_auc_mean']:.4f}+/-{r['test_pr_auc_std']:.4f} "
              f"[{r['test_pr_auc_ci95_lo']:.4f},{r['test_pr_auc_ci95_hi']:.4f}] | "
              f"Dice={r['test_dice_global_mean']:.4f}+/-{r['test_dice_global_std']:.4f} "
              f"[{r['test_dice_global_ci95_lo']:.4f},{r['test_dice_global_ci95_hi']:.4f}]")

    if not df_stats.empty:
        print("\n" + "=" * 70)
        print(" ESTADISTICA — pareado (Wilcoxon + Holm + Cohen's d + IC95% diff)")
        print("=" * 70)
        for _, r in df_stats.iterrows():
            print(f"  {r['comparison']:>16} | {r['metric']:>16} | "
                  f"d_par-sc={r['delta_pareto_minus_scalar']:+.4f} "
                  f"[{r['diff_ci95_lo']:+.4f},{r['diff_ci95_hi']:+.4f}] | "
                  f"p_holm={r['wilcoxon_p_holm']:.4g} | d={r['cohens_d_paired']:+.3f} | "
                  f"reject={r['wilcoxon_reject_holm']}")


# ============================================================
# MAIN
# ============================================================

def main():
    print("\n" + "=" * 70)
    print(" DocVerify — Pareto vs escalar a 30 seeds (revision R1.5 / R2.2.4)")
    print("=" * 70 + "\n")

    if SMOKE:
        print(f"[WARN] SCALAR30_SMOKE=1 -> N_SEEDS={N_SEEDS} MAX_EPOCHS_FINAL={MAX_EPOCHS_FINAL}\n")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    configs = _select_scalar_configs()
    for rule, cfg in configs.items():
        src = cfg["_src"]
        print(f"[OK] {rule:>9}: fold{src['outer_fold']}/trial{src['trial_number']} "
              f"(val_prauc={src['val_cls_prauc']:.4f}, val_dice={src['val_mask_dice_global']:.4f}) "
              f"-> { {k: cfg[k] for k in HP_KEYS} }")

    device = get_device()
    print("\n[INFO] Cargando dataset...")
    df_dev, df_holdout = _load_dataset()
    print(f"[OK] df_dev={len(df_dev)}  df_holdout={len(df_holdout)}")
    print(f"[OK] baseline (pareto) reutilizado de {config.FINAL_TEST_CSV.name}")
    print(f"[OK] seeds: {SEEDS}  (n={len(SEEDS)})")

    # ── Entrenar by_prauc y by_dice (pareto se reutiliza) ────────────────
    new_rows = []
    for rule in ["by_prauc", "by_dice"]:
        cfg = configs[rule]
        for seed in SEEDS:
            print(f"\n  [RUN] {rule} seed={seed}")
            row = _train_one(cfg, rule, seed, df_dev, df_holdout, device)
            _append_row_csv(BLIND_CSV, row)
            new_rows.append(row)
            print(f"    PR-AUC={row['test_pr_auc']:.4f} | "
                  f"Dice={row['test_dice_global']:.4f} | "
                  f"F1macro={row['test_f1_macro']:.4f}")

    # ── Combinar con el baseline pareto reutilizado ──────────────────────
    df_new = pd.DataFrame(new_rows)
    df_pareto = _load_pareto_baseline_rows()
    keep = ["seed", "condition", *CMP_METRICS]
    df_all = pd.concat([df_pareto[keep], df_new[keep]], ignore_index=True)

    # ── Resumen + estadistica ────────────────────────────────────────────
    df_summary = _build_summary(df_all)
    df_summary.to_csv(SUMMARY_CSV, index=False)
    print(f"\n[OK] Resumen guardado: {SUMMARY_CSV}")

    df_stats = _run_stats(df_all)
    if not df_stats.empty:
        df_stats.to_csv(STATS_CSV, index=False)
        print(f"[OK] Estadistica guardada: {STATS_CSV}")

    _print_summary(df_summary, df_stats)

    print(f"\n[OK] Experimento completado.")
    print(f"[OK] Resultados en: {RESULTS_DIR.resolve()}")


if __name__ == "__main__":
    main()
