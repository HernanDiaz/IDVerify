"""
scalar_experiment.py — Comparativa de escalarización clásica vs optimización multi-objetivo.

Pregunta que responde:
    ¿Es necesario el frente de Pareto + criterio de distancia al punto ideal
    para seleccionar loss_w_mask, o basta con barrer un grid fijo y elegir
    por una única métrica (PR-AUC o Dice)?

Diseño:
    Para cada fold externo del nested CV:
      - Se reutilizan los hiperparámetros base (lr, weight_decay, dropout_rate,
        dec_ch) seleccionados por el HPO multi-objetivo (leídos de nested_outer_results.csv).
      - Se entrena un modelo por cada valor de SCALAR_GRID (loss_w_mask fijo).
      - Se evalúan todos en el mismo outer_test que usó el nested CV.
      - Se selecciona el "ganador" de la escalarización según dos criterios:
          · by_prauc : mayor PR-AUC en outer_test
          · by_dice  : mayor Dice global en outer_test
    Los resultados del método multi-objetivo se incluyen en scalar_grid_selected.csv
    para que la comparativa sea autocontenida.

Salidas (en config.SCALAR_EXPORT_DIR):
    scalar_grid_full.csv     — métricas completas para cada (fold, loss_w_mask)
    scalar_grid_selected.csv — fila ganadora por (fold, criterio) + fila del método original
    scalar_stats.csv         — Wilcoxon + t-test + Holm-Bonferroni

Uso:
    python scalar_experiment.py

Requisitos previos:
    - nested_outer_results.csv debe existir (generado por main.py)
    - El dataset debe estar disponible en config.DATASET_ROOT
"""

import gc
import json
import math
import os
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import GroupShuffleSplit

import config
import evaluate as ev
from dataset import make_dataloader
from model import build_model, build_optimizer
from train import (
    _append_row_csv,
    _maybe_compile,
    _set_seeds,
    _train_with_early_stopping,
    get_device,
)

# Splitter idéntico al del nested CV para reproducir los mismos splits
try:
    from sklearn.model_selection import StratifiedGroupKFold

    def _make_sgkf(n_splits: int, seed: int):
        return StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)

except ImportError:
    from sklearn.model_selection import GroupKFold

    def _make_sgkf(n_splits: int, seed: int):
        return GroupKFold(n_splits=n_splits)


# ============================================================
# CARGA DE DATOS
# ============================================================

def _load_dataset() -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Carga y prepara df_dev y df_holdout con exactamente el mismo
    split que usa main.py, para garantizar que los outer_test son idénticos.
    """
    from dataset import add_json_paths, build_full_doc_df, build_image_dataframe

    root = config.DATASET_ROOT.resolve()
    assert root.exists(), (
        f"\n[ERROR] Dataset no encontrado: {root}\n"
        f"  Comprueba DATASET_ROOT en config.py"
    )

    df_imgs = build_image_dataframe(root)
    df_imgs = add_json_paths(df_imgs)
    df_imgs["label"] = (df_imgs["cls_dir"] == "attack").astype(int)

    gss1 = GroupShuffleSplit(n_splits=1, test_size=0.15,
                             random_state=config.SEED_BASE)
    idx_rest, idx_test = next(gss1.split(df_imgs,
                                          y=df_imgs["label"],
                                          groups=df_imgs["stem"]))
    df_rest      = df_imgs.iloc[idx_rest].reset_index(drop=True)
    df_test_base = df_imgs.iloc[idx_test].reset_index(drop=True)

    val_ratio = 0.10 / 0.85
    gss2 = GroupShuffleSplit(n_splits=1, test_size=val_ratio,
                             random_state=config.SEED_BASE)
    idx_train, _ = next(gss2.split(df_rest,
                                    y=df_rest["label"],
                                    groups=df_rest["stem"]))
    df_train_base = df_rest.iloc[idx_train].reset_index(drop=True)

    df_dev = pd.concat(
        [build_full_doc_df(df_train_base, "train"),
         build_full_doc_df(df_rest.drop(index=idx_train).reset_index(drop=True), "val")],
        ignore_index=True,
    )
    df_holdout = build_full_doc_df(df_test_base, "test")

    return df_dev, df_holdout


# ============================================================
# ENTRENAMIENTO DE UNA CONFIGURACIÓN ESCALAR
# ============================================================

def _train_scalar(
    params: dict,
    loss_w_mask: float,
    outer_fold: int,
    df_outer_train: pd.DataFrame,
    df_outer_test: pd.DataFrame,
    device: torch.device,
) -> dict:
    """
    Entrena un modelo con loss_w_mask fijo y evalúa en df_outer_test.
    Devuelve un dict con todas las métricas + metadatos.
    """
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    seed = config.SEED_BASE + 9000 + outer_fold * 100 + int(loss_w_mask * 10)
    _set_seeds(seed)

    # Split train/sel idéntico al nested CV (misma semilla por fold)
    gss = GroupShuffleSplit(n_splits=1, test_size=0.15,
                            random_state=config.SEED_BASE + outer_fold)
    tr_idx, sel_idx = next(gss.split(
        df_outer_train,
        y=df_outer_train["label"],
        groups=df_outer_train["stem"],
    ))

    loader_tr   = make_dataloader(
        df_outer_train.iloc[tr_idx].reset_index(drop=True),
        training=True, seed=seed, device=device,
    )
    loader_sel  = make_dataloader(
        df_outer_train.iloc[sel_idx].reset_index(drop=True),
        training=False, seed=seed, device=device,
    )
    loader_test = make_dataloader(
        df_outer_test, training=False, seed=seed, device=device,
    )

    # loss_w_mask en params para _train_with_early_stopping
    run_params = dict(params)
    run_params["loss_w_mask"] = loss_w_mask

    model     = _maybe_compile(build_model(run_params, device))
    optimizer = build_optimizer(model, run_params)
    scaler    = torch.amp.GradScaler("cuda",
                                     enabled=config.USE_AMP and device.type == "cuda")

    t0 = time.perf_counter()
    model = _train_with_early_stopping(
        model, optimizer, scaler,
        loader_tr, loader_sel, device,
        params=run_params,
        max_epochs=config.MAX_EPOCHS_SCALAR,
        patience=12,
        variant="multitask",   # ambas pérdidas activas, lw_mask = loss_w_mask fijo
        desc=f"[scalar fold={outer_fold} w={loss_w_mask}]",
    )
    train_time = time.perf_counter() - t0

    # Umbral óptimo desde loader_sel
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

    row = {
        "outer_fold":           outer_fold,
        "loss_w_mask":          loss_w_mask,
        "train_time_sec":       float(train_time),
        "thr_cls_from_val_sel": float(thr_bacc),
        "val_sel_best_bacc":    float(best_bacc),
        "val_sel_best_f1m":     float(best_f1m),
        **{f"test_{k}": v for k, v in met.items()},
        **{f"hp_{k}": v for k, v in params.items()},
    }
    row["test_cm_TN_FP_FN_TP"] = json.dumps(met["cm_TN_FP_FN_TP"])

    del model, optimizer, loader_tr, loader_sel, loader_test
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return row


# ============================================================
# SELECCIÓN DEL GANADOR POR CRITERIO MONO-OBJETIVO
# ============================================================

def _select_winners(df_full: pd.DataFrame) -> pd.DataFrame:
    """
    Para cada fold, selecciona el loss_w_mask ganador según dos criterios:
      · by_prauc : maximiza test_pr_auc
      · by_dice  : maximiza test_dice_global
    Devuelve un DataFrame con una fila por (fold, criterio).
    """
    rows = []
    for fold in sorted(df_full["outer_fold"].unique()):
        df_fold = df_full[df_full["outer_fold"] == fold]
        for criterion, metric in [("by_prauc", "test_pr_auc"),
                                   ("by_dice",  "test_dice_global")]:
            best_idx = df_fold[metric].idxmax()
            best_row = df_fold.loc[best_idx].to_dict()
            best_row["selection_criterion"] = criterion
            rows.append(best_row)
    return pd.DataFrame(rows)


# ============================================================
# TESTS ESTADÍSTICOS
# ============================================================

def _run_scalar_stats(df_selected: pd.DataFrame):
    """
    Compara el método multi-objetivo (method=multiobjective) contra
    cada criterio de escalarización (by_prauc, by_dice) usando
    Wilcoxon + t-test pareado + corrección Holm-Bonferroni.
    Guarda los resultados en config.SCALAR_STATS_CSV.
    """
    try:
        from scipy.stats import ttest_rel, wilcoxon
    except ImportError:
        import subprocess, sys
        subprocess.check_call([sys.executable, "-m", "pip", "install", "scipy", "-q"])
        from scipy.stats import ttest_rel, wilcoxon

    METRICS = [
        "test_pr_auc", "test_dice_global", "test_dice_pos_mean",
        "test_miou", "test_pix_specificity", "test_pix_f1",
        "test_bacc", "test_f1_macro",
    ]

    def cohens_d(a, b):
        d = np.asarray(a) - np.asarray(b)
        return float(np.mean(d) / (np.std(d, ddof=1) + 1e-12))

    def holm_bonferroni(pvals, alpha=0.05):
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

    df_ref = df_selected[df_selected["method"] == "multiobjective"]
    stats_rows = []

    for criterion in ["by_prauc", "by_dice"]:
        df_comp = df_selected[df_selected["selection_criterion"] == criterion]
        mrg = df_ref.merge(df_comp, on="outer_fold", suffixes=("_A", "_B"))

        if len(mrg) < 3:
            print(f"[STATS] n insuficiente para multiobjective vs {criterion}. Saltando.")
            continue

        pvals_w, pvals_t, tmp = [], [], []
        for met in METRICS:
            col_a = f"{met}_A" if f"{met}_A" in mrg.columns else met
            col_b = f"{met}_B" if f"{met}_B" in mrg.columns else met

            if col_a not in mrg.columns or col_b not in mrg.columns:
                continue

            a = mrg[col_a].values.astype(float)
            b = mrg[col_b].values.astype(float)

            try:
                p_w = float(wilcoxon(a, b, zero_method="wilcox").pvalue)
            except Exception:
                p_w = float("nan")
            try:
                p_t = float(ttest_rel(a, b).pvalue)
            except Exception:
                p_t = float("nan")

            pvals_w.append(p_w)
            pvals_t.append(p_t)
            tmp.append({
                "comparison":      f"multiobjective vs {criterion}",
                "metric":          met,
                "n_paired":        len(mrg),
                "mean_A":          float(np.mean(a)),
                "std_A":           float(np.std(a, ddof=1)),
                "mean_B":          float(np.mean(b)),
                "std_B":           float(np.std(b, ddof=1)),
                "wilcoxon_p":      p_w,
                "ttest_p":         p_t,
                "cohens_d_paired": cohens_d(a, b),
            })

        adj_w, rej_w = holm_bonferroni(pvals_w)
        adj_t, rej_t = holm_bonferroni(pvals_t)

        for i, row in enumerate(tmp):
            row.update({
                "wilcoxon_p_holm":      float(adj_w[i]),
                "wilcoxon_reject_holm": bool(rej_w[i]),
                "ttest_p_holm":         float(adj_t[i]),
                "ttest_reject_holm":    bool(rej_t[i]),
            })
            stats_rows.append(row)

    if stats_rows:
        df_stats = pd.DataFrame(stats_rows)
        df_stats.to_csv(config.SCALAR_STATS_CSV, index=False)
        print(f"\n[OK] Estadísticas guardadas: {config.SCALAR_STATS_CSV}")

        print(f"\n{'='*60}")
        print(" RESUMEN ESTADÍSTICO (Wilcoxon + Holm)")
        print(f"{'='*60}")
        for _, r in df_stats.sort_values(["comparison", "wilcoxon_p_holm"]).iterrows():
            print(f"  {r['comparison']} | {r['metric']} | "
                  f"p_holm={r['wilcoxon_p_holm']:.4g} | "
                  f"d={r['cohens_d_paired']:.3f} | "
                  f"reject={r['wilcoxon_reject_holm']}")


# ============================================================
# RESUMEN EN CONSOLA
# ============================================================

def _print_summary(df_full: pd.DataFrame, df_selected: pd.DataFrame):
    print(f"\n{'='*60}")
    print(" RESUMEN — GRID COMPLETO (media ± std por loss_w_mask)")
    print(f"{'='*60}")
    for w in sorted(df_full["loss_w_mask"].unique()):
        d = df_full[df_full["loss_w_mask"] == w]
        pr  = f"{d['test_pr_auc'].mean():.4f}±{d['test_pr_auc'].std(ddof=1):.4f}"
        dic = f"{d['test_dice_global'].mean():.4f}±{d['test_dice_global'].std(ddof=1):.4f}"
        print(f"  w={w:.1f} | PR-AUC={pr} | Dice={dic}")

    print(f"\n{'='*60}")
    print(" RESUMEN — GANADORES POR CRITERIO vs MÉTODO MULTI-OBJETIVO")
    print(f"{'='*60}")
    for method in df_selected["method"].unique():
        d = df_selected[df_selected["method"] == method]

        def ms(col):
            vals = d[col].dropna()
            return f"{vals.mean():.4f}±{vals.std(ddof=1):.4f}"

        crit = d["selection_criterion"].iloc[0] if "selection_criterion" in d.columns else "—"
        print(f"  [{method} / {crit}] "
              f"PR-AUC={ms('test_pr_auc')} | "
              f"Dice={ms('test_dice_global')} | "
              f"mIoU={ms('test_miou')}")


# ============================================================
# MAIN
# ============================================================

def main():
    print("\n" + "=" * 60)
    print(" DocVerify — Experimento de Escalarización Clásica")
    print("=" * 60 + "\n")

    # Verificar que el nested CV ya se ejecutó
    if not config.OUTER_CSV.exists():
        raise FileNotFoundError(
            f"No se encuentra {config.OUTER_CSV}.\n"
            f"Ejecuta primero main.py para generar el nested CV."
        )

    config.SCALAR_EXPORT_DIR.mkdir(parents=True, exist_ok=True)

    # Cargar hiperparámetros base del nested CV
    df_outer = pd.read_csv(config.OUTER_CSV)
    hp_cols  = [c for c in df_outer.columns if c.startswith("hp_")]
    print(f"[OK] nested_outer_results.csv cargado — {len(df_outer)} folds")
    print(f"[OK] Hiperparámetros base: {[c.replace('hp_', '') for c in hp_cols]}")
    print(f"[OK] Grid loss_w_mask: {config.SCALAR_GRID}")
    print(f"[OK] MAX_EPOCHS_SCALAR: {config.MAX_EPOCHS_SCALAR}")

    # Cargar dataset
    print("\n[INFO] Cargando dataset...")
    os.environ["PYTHONHASHSEED"] = str(config.SEED_BASE)
    random.seed(config.SEED_BASE)
    np.random.seed(config.SEED_BASE)
    torch.manual_seed(config.SEED_BASE)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.SEED_BASE)

    df_dev, _ = _load_dataset()
    device    = get_device()

    # Reproducir los mismos outer splits del nested CV
    outer_splitter = _make_sgkf(config.N_OUTER, seed=config.SEED_BASE + 7)
    y = df_dev["label"].values
    g = df_dev["stem"].values

    all_grid_rows = []

    for outer_fold, (train_idx, test_idx) in enumerate(
        outer_splitter.split(df_dev, y=y, groups=g), start=1
    ):
        print(f"\n{'='*60}")
        print(f" OUTER FOLD {outer_fold}/{config.N_OUTER}")
        print(f"{'='*60}")

        df_outer_train = df_dev.iloc[train_idx].reset_index(drop=True)
        df_outer_test  = df_dev.iloc[test_idx].reset_index(drop=True)

        # Hiperparámetros base de este fold
        fold_row = df_outer[df_outer["outer_fold"] == outer_fold].iloc[0]
        base_params = {
            c.replace("hp_", ""): fold_row[c]
            for c in hp_cols
            if c != "hp_loss_w_mask"   # este lo variamos nosotros
        }
        # Asegurar tipos correctos
        base_params["dec_ch"]       = int(base_params["dec_ch"])
        base_params["dropout_rate"] = float(base_params["dropout_rate"])
        base_params["lr"]           = float(base_params["lr"])
        base_params["weight_decay"] = float(base_params["weight_decay"])
        if "alpha" in base_params:
            base_params["alpha"]    = float(base_params["alpha"])

        print(f"  Parámetros base: {base_params}")
        print(f"  Outer test size: {len(df_outer_test)}")

        for w in config.SCALAR_GRID:
            print(f"\n  [Grid] loss_w_mask={w}")
            row = _train_scalar(
                params         = base_params,
                loss_w_mask    = w,
                outer_fold     = outer_fold,
                df_outer_train = df_outer_train,
                df_outer_test  = df_outer_test,
                device         = device,
            )
            _append_row_csv(config.SCALAR_GRID_CSV, row)
            all_grid_rows.append(row)
            print(f"    PR-AUC={row['test_pr_auc']:.4f} | "
                  f"Dice={row['test_dice_global']:.4f} | "
                  f"mIoU={row['test_miou']:.4f}")

    # ── Selección de ganadores por criterio mono-objetivo ──────────────
    df_full     = pd.DataFrame(all_grid_rows)
    df_winners  = _select_winners(df_full)
    df_winners["method"] = "scalar"

    # ── Añadir resultados del método multi-objetivo ────────────────────
    # Tomamos las métricas de nested_outer_results.csv para comparar
    metric_map = {
        "test_pr_auc":          "outer_test_pr_auc",
        "test_roc_auc":         "outer_test_roc_auc",
        "test_bacc":            "outer_test_bacc",
        "test_f1_1":            "outer_test_f1_1",
        "test_f1_macro":        "outer_test_f1_macro",
        "test_prec1":           "outer_test_prec1",
        "test_rec1":            "outer_test_rec1",
        "test_dice_global":     "outer_test_dice_global",
        "test_dice_pos_mean":   "outer_test_dice_pos_mean",
        "test_miou":            "outer_test_miou",
        "test_pix_specificity": "outer_test_pix_specificity",
        "test_pix_f1":          "outer_test_pix_f1",
        "test_pix_prec":        "outer_test_pix_prec",
        "test_pix_rec":         "outer_test_pix_rec",
    }

    mo_rows = []
    for _, r in df_outer.iterrows():
        mo_row = {"outer_fold": int(r["outer_fold"]),
                  "method": "multiobjective",
                  "selection_criterion": "pareto_distance",
                  "loss_w_mask": float(r.get("hp_loss_w_mask", float("nan")))}
        for test_col, outer_col in metric_map.items():
            mo_row[test_col] = float(r[outer_col]) if outer_col in r.index else float("nan")
        mo_rows.append(mo_row)

    df_multiobjective = pd.DataFrame(mo_rows)

    # Combinar en un único CSV de seleccionados
    df_selected = pd.concat([df_winners, df_multiobjective], ignore_index=True)
    df_selected.to_csv(config.SCALAR_SELECTED_CSV, index=False)
    print(f"\n[OK] scalar_grid_selected.csv guardado: {config.SCALAR_SELECTED_CSV}")

    # ── Tests estadísticos ─────────────────────────────────────────────
    _run_scalar_stats(df_selected)

    # ── Resumen en consola ─────────────────────────────────────────────
    _print_summary(df_full, df_selected)

    print(f"\n[OK] Experimento completado.")
    print(f"[OK] Resultados en: {config.SCALAR_EXPORT_DIR.resolve()}")


if __name__ == "__main__":
    main()
