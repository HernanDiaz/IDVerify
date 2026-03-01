"""
train.py — Lógica de entrenamiento: Nested CV, HPO con Optuna,
           test ciego multi-seed, ablación y estadística inferencial.
"""

import gc
import json
import math
import os
import random
import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import GroupShuffleSplit

import config
import evaluate as ev
from dataset import make_dataset
from model import (
    AddValDistance,
    EpochCounter,
    build_and_compile,
    build_and_compile_variant,
)

# ============================================================
# UTILIDADES
# ============================================================

def _set_seeds(seed: int):
    """Fija todas las semillas de aleatoriedad para reproducibilidad."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)


def _append_row_csv(path: Path, row: dict):
    """Añade una fila a un CSV de forma incremental (sin perder datos previos)."""
    df_row = pd.DataFrame([row])
    df_row.to_csv(path, mode="a", header=not path.exists(), index=False)


# ============================================================
# SPLITTERS
# ============================================================

try:
    from sklearn.model_selection import StratifiedGroupKFold

    def _make_sgkf(n_splits: int, seed: int):
        return StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    print("[INFO] Usando StratifiedGroupKFold")
except ImportError:
    from sklearn.model_selection import GroupKFold

    def _make_sgkf(n_splits: int, seed: int):
        return GroupKFold(n_splits=n_splits)

    print("[INFO] StratifiedGroupKFold no disponible, usando GroupKFold")


# ============================================================
# OPTUNA
# ============================================================

try:
    import optuna
except ImportError:
    import subprocess, sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", "optuna", "-q"])
    import optuna

optuna.logging.set_verbosity(optuna.logging.INFO)


def _make_sampler(seed: int):
    """Crea un sampler multi-objetivo (MOTPESampler o NSGAIISampler como fallback)."""
    try:
        return optuna.samplers.MOTPESampler(seed=seed, multivariate=True)
    except Exception:
        return optuna.samplers.NSGAIISampler(seed=seed)


def _get_pareto_trials(study: optuna.Study) -> list:
    """Devuelve el frente de Pareto compatible con versiones antiguas y nuevas de Optuna."""
    if hasattr(study, "get_pareto_front_trials"):
        return study.get_pareto_front_trials()
    if hasattr(study, "best_trials"):
        bt = list(study.best_trials)
        if bt:
            return bt

    # Fallback manual: filtro de no-dominados
    dirs = getattr(study, "directions", None)
    cand = [t for t in study.trials
            if t.state == optuna.trial.TrialState.COMPLETE and t.values is not None]

    if dirs is None:
        return cand

    sign = [-1.0 if str(d).lower().endswith("minimize") else 1.0 for d in dirs]

    def vals_max(t):
        return [sign[i] * float(t.values[i]) for i in range(len(sign))]

    def dominates(a, b):
        va, vb = vals_max(a), vals_max(b)
        return all(x >= y for x, y in zip(va, vb)) and any(x > y for x, y in zip(va, vb))

    return [t for t in cand if not any(dominates(u, t) for u in cand if u.number != t.number)]


def _select_best_trial(study: optuna.Study) -> tuple:
    """Selecciona el trial del frente de Pareto con menor distancia al punto ideal (1, 1)."""
    pareto = _get_pareto_trials(study)
    if not pareto:
        pareto = [t for t in study.trials
                  if t.state == optuna.trial.TrialState.COMPLETE and t.values is not None]

    best_t, best_d = None, float("inf")
    for t in pareto:
        pr, dc = float(t.values[0]), float(t.values[1])
        dist = math.sqrt((1.0 - pr) ** 2 + (1.0 - dc) ** 2)
        if dist < best_d:
            best_d, best_t = dist, t
    return best_t, best_d


# ============================================================
# OBJETIVO OPTUNA (inner CV)
# ============================================================

def _make_inner_objective(df_outer_train: pd.DataFrame, outer_fold_id: int):
    """
    Devuelve la función objetivo de Optuna para el inner CV de un fold externo.
    Maximiza simultáneamente PR-AUC y Dice global (multi-objetivo).
    """
    splitter = _make_sgkf(config.N_INNER, seed=config.SEED_BASE + 100 + outer_fold_id)
    y = df_outer_train["label"].values
    g = df_outer_train["stem"].values

    def objective(trial: optuna.Trial) -> tuple[float, float]:
        t0 = time.perf_counter()
        tf.keras.backend.clear_session()
        gc.collect()

        params = {
            "lr":            trial.suggest_float("lr", 1e-5, 1e-3, log=True),
            "weight_decay":  trial.suggest_float("weight_decay", 1e-7, 1e-4, log=True),
            "dropout_rate":  trial.suggest_float("dropout_rate", 0.0, 0.6),
            "alpha":         trial.suggest_float("alpha", 0.05, 0.30),
            "dec_ch":        trial.suggest_categorical("dec_ch", [96, 128, 192]),
            "loss_w_mask":   trial.suggest_float("loss_w_mask", 0.5, 3.0),
        }

        fold_metrics = []
        epochs_completed = 0

        try:
            for inner_id, (tr_idx, va_idx) in enumerate(
                splitter.split(df_outer_train, y=y, groups=g)
            ):
                tf.keras.backend.clear_session()
                gc.collect()

                seed_fold = config.SEED_BASE + 1000 * outer_fold_id + 10 * inner_id
                _set_seeds(seed_fold)

                df_tr = df_outer_train.iloc[tr_idx].reset_index(drop=True)
                df_va = df_outer_train.iloc[va_idx].reset_index(drop=True)

                ds_tr = make_dataset(df_tr, training=True,  seed=seed_fold)
                ds_va = make_dataset(df_va, training=False, seed=seed_fold)

                model    = build_and_compile(params)
                epc      = EpochCounter()
                callbacks = [epc]

                fit_kwargs = dict(
                    x               = ds_tr,
                    validation_data = ds_va,
                    epochs          = config.MAX_EPOCHS_TRIAL,
                    callbacks       = callbacks,
                    validation_freq = config.TRIAL_VALIDATION_FREQ,
                    verbose         = 0,
                )

                # EarlyStopping solo si se valida en cada epoch
                if config.TRIAL_VALIDATION_FREQ == 1 and config.MAX_EPOCHS_TRIAL >= 2:
                    add_dist = AddValDistance()
                    es = tf.keras.callbacks.EarlyStopping(
                        monitor="val_distance_to_ideal", mode="min",
                        patience=config.TRIAL_PATIENCE,
                        restore_best_weights=True, verbose=0,
                    )
                    fit_kwargs["callbacks"] = [add_dist, epc, es]

                if config.TRIAL_STEPS_PER_EPOCH > 0:
                    fit_kwargs["steps_per_epoch"] = config.TRIAL_STEPS_PER_EPOCH
                if config.TRIAL_VAL_STEPS > 0:
                    fit_kwargs["validation_steps"] = config.TRIAL_VAL_STEPS

                model.fit(**fit_kwargs)
                epochs_completed += epc.epochs

                result = model.evaluate(ds_va, verbose=0, return_dict=True)
                fold_metrics.append({
                    "prauc": float(result.get("cls_prauc", np.nan)),
                    "dice":  float(result.get("mask_dice_global", np.nan)),
                })

        except (tf.errors.ResourceExhaustedError, MemoryError):
            _append_row_csv(config.TRIALS_CSV, {
                "trial_number":    trial.number,
                "outer_fold":      outer_fold_id,
                "pruned":          True,
                "epochs_completed": epochs_completed,
                "time_sec":        time.perf_counter() - t0,
                **params,
                "notes":           "OOM/ResourceExhausted",
            })
            raise optuna.exceptions.TrialPruned("OOM/ResourceExhausted")

        pr_mean   = float(np.nanmean([m["prauc"] for m in fold_metrics]))
        dc_mean   = float(np.nanmean([m["dice"]  for m in fold_metrics]))
        dist_mean = math.sqrt((1.0 - pr_mean) ** 2 + (1.0 - dc_mean) ** 2)

        _append_row_csv(config.TRIALS_CSV, {
            "trial_number":         trial.number,
            "outer_fold":           outer_fold_id,
            "pruned":               False,
            "epochs_completed":     epochs_completed,
            "time_sec":             time.perf_counter() - t0,
            **params,
            "val_cls_prauc":        pr_mean,
            "val_mask_dice_global": dc_mean,
            "distance_to_ideal":    dist_mean,
        })

        return pr_mean, dc_mean

    return objective


# ============================================================
# ENTRENAMIENTO FINAL (un solo modelo)
# ============================================================

def _train_final_model(
    params: dict,
    seed: int,
    variant: str,
    df_dev: pd.DataFrame,
    df_holdout: pd.DataFrame,
) -> dict:
    """
    Entrena el modelo final sobre df_dev, selecciona umbral en
    un split interno y evalúa sobre df_holdout.
    Devuelve un diccionario de métricas.
    """
    tf.keras.backend.clear_session()
    gc.collect()
    _set_seeds(seed)

    gss = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=seed)
    tr_idx, sel_idx = next(gss.split(df_dev, y=df_dev["label"], groups=df_dev["stem"]))

    df_train = df_dev.iloc[tr_idx].reset_index(drop=True)
    df_sel   = df_dev.iloc[sel_idx].reset_index(drop=True)

    ds_train = make_dataset(df_train, training=True,  seed=seed, reshuffle_each_iteration=True)
    ds_sel   = make_dataset(df_sel,   training=False, seed=seed)
    ds_test  = make_dataset(df_holdout, training=False, seed=seed)

    model     = build_and_compile_variant(params, variant=variant)
    callbacks = []

    epochs_budget = (
        config.MAX_EPOCHS_FINAL if variant == "multitask"
        else min(config.MAX_EPOCHS_ABLATION, config.MAX_EPOCHS_FINAL)
    )

    if variant in ("multitask", "unweighted_losses"):
        add_dist = AddValDistance()
        callbacks += [
            add_dist,
            tf.keras.callbacks.EarlyStopping(
                monitor="val_distance_to_ideal", mode="min",
                patience=8, restore_best_weights=True, verbose=0,
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor="val_distance_to_ideal", mode="min",
                factor=0.5, patience=3, min_lr=1e-6, verbose=0,
            ),
        ]
    elif variant == "cls_only":
        callbacks += [
            tf.keras.callbacks.EarlyStopping(
                monitor="val_cls_prauc", mode="max",
                patience=6, restore_best_weights=True, verbose=0,
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor="val_cls_prauc", mode="max",
                factor=0.5, patience=3, min_lr=1e-6, verbose=0,
            ),
        ]
    elif variant == "seg_only":
        callbacks += [
            tf.keras.callbacks.EarlyStopping(
                monitor="val_mask_dice_global", mode="max",
                patience=6, restore_best_weights=True, verbose=0,
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor="val_mask_dice_global", mode="max",
                factor=0.5, patience=3, min_lr=1e-6, verbose=0,
            ),
        ]

    t0 = time.perf_counter()
    model.fit(ds_train, validation_data=ds_sel,
              epochs=epochs_budget, callbacks=callbacks, verbose=0)
    train_time = time.perf_counter() - t0

    # Selección de umbral desde ds_sel
    if variant != "seg_only":
        y_true_sel, y_prob_sel = [], []
        for x_b, y_b in ds_sel:
            y_true_sel.append(y_b["cls"].numpy().reshape(-1).astype(int))
            y_prob_sel.append(model(x_b, training=False)["cls"].numpy().reshape(-1).astype(float))
        thr_bacc, best_bacc, thr_f1, best_f1m = ev.threshold_sweep(
            np.concatenate(y_true_sel), np.concatenate(y_prob_sel)
        )
    else:
        thr_bacc, best_bacc, thr_f1, best_f1m = 0.5, float("nan"), 0.5, float("nan")

    met = ev.eval_model(model, ds_test, thr_cls=thr_bacc)

    row = {
        "seed":               seed,
        "variant":            variant,
        "train_time_sec":     float(train_time),
        "epochs_budget":      epochs_budget,
        "thr_cls_from_val_sel": float(thr_bacc),
        "val_sel_best_bacc":  float(best_bacc),
        "val_sel_best_f1m":   float(best_f1m),
        **{f"test_{k}": v for k, v in met.items()},
        **{f"hp_{k}": v for k, v in params.items()},
    }
    row["test_cm_TN_FP_FN_TP"] = json.dumps(met["cm_TN_FP_FN_TP"])

    _append_row_csv(config.FINAL_TEST_CSV, row)
    return row


# ============================================================
# NESTED CV COMPLETO
# ============================================================

def run_nested_cv(df_dev: pd.DataFrame) -> list[dict]:
    """
    Ejecuta el loop completo de Nested Cross-Validation con HPO Optuna.
    Devuelve la lista de resultados por fold externo.
    """
    outer_splitter = _make_sgkf(config.N_OUTER, seed=config.SEED_BASE + 7)
    y = df_dev["label"].values
    g = df_dev["stem"].values
    outer_rows = []

    for outer_fold, (train_idx, test_idx) in enumerate(
        outer_splitter.split(df_dev, y=y, groups=g), start=1
    ):
        print(f"\n{'='*60}")
        print(f" OUTER FOLD {outer_fold}/{config.N_OUTER}")
        print(f"{'='*60}")

        df_outer_train = df_dev.iloc[train_idx].reset_index(drop=True)
        df_outer_test  = df_dev.iloc[test_idx].reset_index(drop=True)

        study_name = f"DOCVERIFY{config.RUN_TAG}_OUTER{outer_fold}"
        storage    = f"sqlite:///{config.SQLITE_PATH}"

        study = optuna.create_study(
            study_name    = study_name,
            directions    = ["maximize", "maximize"],
            sampler       = _make_sampler(config.SEED_BASE + outer_fold),
            pruner        = optuna.pruners.NopPruner(),
            storage       = storage,
            load_if_exists= True,
        )

        objective = _make_inner_objective(df_outer_train, outer_fold_id=outer_fold)

        t_hpo0 = time.perf_counter()
        study.optimize(objective, n_trials=config.N_TRIALS, gc_after_trial=True)
        t_hpo1 = time.perf_counter()

        best_trial, best_dist = _select_best_trial(study)
        if best_trial is None:
            raise RuntimeError(
                f"No hay trials completados en outer fold {outer_fold}. "
                "Revisa si todos fallaron por OOM."
            )

        pareto = _get_pareto_trials(study)
        print(f"[HPO] Tiempo: {t_hpo1 - t_hpo0:.1f}s | Pareto trials: {len(pareto)}")
        print(f"[HPO] Trial seleccionado: #{best_trial.number} | dist={best_dist:.4f} | "
              f"valores (PR-AUC, Dice): {best_trial.values}")

        # Exportar frente de Pareto
        for t in pareto:
            pr, dc = (t.values or [np.nan, np.nan])
            _append_row_csv(config.PARETO_CSV, {
                "outer_fold":           outer_fold,
                "trial_number":         t.number,
                "val_cls_prauc":        float(pr),
                "val_mask_dice_global": float(dc),
                "distance_to_ideal":    math.sqrt((1 - float(pr)) ** 2 + (1 - float(dc)) ** 2),
                **t.params,
            })

        # Entrenamiento final del fold externo
        gss = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=config.SEED_BASE + outer_fold)
        tr2_idx, sel_idx = next(gss.split(
            df_outer_train, y=df_outer_train["label"], groups=df_outer_train["stem"]
        ))

        df_train_final = df_outer_train.iloc[tr2_idx].reset_index(drop=True)
        df_val_sel     = df_outer_train.iloc[sel_idx].reset_index(drop=True)

        seed_final = config.SEED_BASE + 5000 + outer_fold
        _set_seeds(seed_final)

        ds_train = make_dataset(df_train_final, training=True,  seed=seed_final,
                                reshuffle_each_iteration=True)
        ds_sel   = make_dataset(df_val_sel,     training=False, seed=seed_final)
        ds_test  = make_dataset(df_outer_test,  training=False, seed=seed_final)

        params = dict(best_trial.params)
        model  = build_and_compile(params)

        add_dist = AddValDistance()
        t0 = time.perf_counter()
        model.fit(
            ds_train,
            validation_data = ds_sel,
            epochs          = config.MAX_EPOCHS_FINAL,
            callbacks       = [
                add_dist,
                tf.keras.callbacks.EarlyStopping(
                    monitor="val_distance_to_ideal", mode="min",
                    patience=8, restore_best_weights=True, verbose=0,
                ),
                tf.keras.callbacks.ReduceLROnPlateau(
                    monitor="val_distance_to_ideal", mode="min",
                    factor=0.5, patience=3, min_lr=1e-6, verbose=0,
                ),
            ],
            verbose=0,
        )
        train_time = time.perf_counter() - t0

        # Umbral desde ds_sel
        y_true_sel, y_prob_sel = [], []
        for x_b, y_b in ds_sel:
            y_true_sel.append(y_b["cls"].numpy().reshape(-1).astype(int))
            y_prob_sel.append(model(x_b, training=False)["cls"].numpy().reshape(-1).astype(float))

        thr_bacc, best_bacc, thr_f1, best_f1m = ev.threshold_sweep(
            np.concatenate(y_true_sel), np.concatenate(y_prob_sel)
        )

        met = ev.eval_model(model, ds_test, thr_cls=thr_bacc)

        row = {
            "outer_fold":                outer_fold,
            "selected_trial_number":     best_trial.number,
            "distance_to_ideal_innercv": float(best_dist),
            "hpo_time_sec":              float(t_hpo1 - t_hpo0),
            "train_time_sec":            float(train_time),
            "thr_cls_from_val_sel":      float(thr_bacc),
            "val_sel_best_bacc":         float(best_bacc),
            "val_sel_best_f1m":          float(best_f1m),
            **{f"outer_test_{k}": v for k, v in met.items()},
            **{f"hp_{k}": v for k, v in params.items()},
        }
        _append_row_csv(config.OUTER_CSV, row)
        outer_rows.append(row)

    return outer_rows


# ============================================================
# RESUMEN NESTED CV
# ============================================================

def print_nested_cv_summary(outer_rows: list[dict]):
    df = pd.DataFrame(outer_rows)
    print(f"\n{'='*60}")
    print(" RESUMEN NESTED CV (outer-test)")
    print(f"{'='*60}")
    for k in ["outer_test_pr_auc", "outer_test_dice_global", "outer_test_miou",
              "outer_test_pix_specificity", "outer_test_bacc", "outer_test_f1_macro"]:
        if k in df.columns:
            m = float(df[k].mean())
            s = float(df[k].std(ddof=1)) if len(df) > 1 else 0.0
            print(f"  {k}: {m:.4f} ± {s:.4f}")


# ============================================================
# TEST CIEGO MULTI-SEED + ABLACIÓN
# ============================================================

def run_blind_test(df_dev: pd.DataFrame, df_holdout: pd.DataFrame, final_params: dict):
    """Ejecuta el test ciego con múltiples seeds y variantes de ablación."""
    variants = config.ABLATION_VARIANTS if config.RUN_ABLATIONS else ["multitask"]

    print(f"\n{'='*60}")
    print(" TEST CIEGO (HOLDOUT)")
    print(f"{'='*60}")
    print(f"  Holdout size : {len(df_holdout)}")
    print(f"  Seeds        : {config.FINAL_SEEDS}")
    print(f"  Variantes    : {variants}")

    for variant in variants:
        for seed in config.FINAL_SEEDS:
            print(f"  [FINAL] variant={variant} seed={seed}")
            _train_final_model(final_params, seed=seed, variant=variant,
                               df_dev=df_dev, df_holdout=df_holdout)

    df_final = pd.read_csv(config.FINAL_TEST_CSV)
    _print_final_summary(df_final)


def _print_final_summary(df_final: pd.DataFrame):
    print(f"\n{'='*60}")
    print(" RESUMEN TEST CIEGO (HOLDOUT)")
    print(f"{'='*60}")
    for variant in df_final["variant"].unique():
        d = df_final[df_final["variant"] == variant]
        def ms(col):
            vals = d[col].dropna()
            m = float(vals.mean())
            s = float(vals.std(ddof=1)) if len(vals) > 1 else 0.0
            return f"{m:.4f}±{s:.4f}"
        print(f"  [{variant}] PR-AUC={ms('test_pr_auc')} | "
              f"Dice(global)={ms('test_dice_global')} | "
              f"Dice(+)={ms('test_dice_pos_mean')} | "
              f"mIoU={ms('test_miou')}")


# ============================================================
# ESTADÍSTICA INFERENCIAL
# ============================================================

def run_stats_tests():
    """Pruebas estadísticas pareadas (Wilcoxon + t-test + Holm-Bonferroni + Cohen's d)."""
    if not config.FINAL_TEST_CSV.exists():
        print("[STATS] No se encuentra el CSV de resultados finales. Saltando.")
        return

    try:
        from scipy.stats import ttest_rel, wilcoxon
    except ImportError:
        import subprocess, sys
        subprocess.check_call([sys.executable, "-m", "pip", "install", "scipy", "-q"])
        from scipy.stats import ttest_rel, wilcoxon

    df_final = pd.read_csv(config.FINAL_TEST_CSV)
    BASE = "multitask"
    METRICS = [
        "test_pr_auc", "test_dice_global", "test_dice_pos_mean",
        "test_miou", "test_pix_specificity", "test_pix_f1",
        "test_bacc", "test_f1_macro",
    ]

    def cohens_d(a, b):
        d = np.asarray(a) - np.asarray(b)
        return float(np.mean(d) / (np.std(d, ddof=1) + 1e-12))

    def holm_bonferroni(pvals, alpha=0.05):
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

    stats_rows = []
    for comp in [v for v in df_final["variant"].unique() if v != BASE]:
        dfA = df_final[df_final["variant"] == BASE]
        dfB = df_final[df_final["variant"] == comp]
        mrg = dfA.merge(dfB, on="seed", suffixes=("_A", "_B"))

        if len(mrg) < 3:
            print(f"[STATS] n insuficiente para {BASE} vs {comp} (n={len(mrg)}). Saltando.")
            continue

        pvals_w, pvals_t, tmp = [], [], []

        for met in METRICS:
            a = mrg[f"{met}_A"].values
            b = mrg[f"{met}_B"].values
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
                "comparison": f"{BASE} vs {comp}",
                "metric":     met,
                "n_paired":   len(mrg),
                "mean_A":     float(np.mean(a)), "std_A": float(np.std(a, ddof=1)),
                "mean_B":     float(np.mean(b)), "std_B": float(np.std(b, ddof=1)),
                "wilcoxon_p": p_w,
                "ttest_p":    p_t,
                "cohens_d_paired": cohens_d(a, b),
            })

        adj_w, rej_w = holm_bonferroni(pvals_w)
        adj_t, rej_t = holm_bonferroni(pvals_t)

        for i, row in enumerate(tmp):
            row.update({
                "wilcoxon_p_holm":    float(adj_w[i]),
                "wilcoxon_reject_holm": bool(rej_w[i]),
                "ttest_p_holm":       float(adj_t[i]),
                "ttest_reject_holm":  bool(rej_t[i]),
            })
            stats_rows.append(row)

    if stats_rows:
        df_stats = pd.DataFrame(stats_rows)
        df_stats.to_csv(config.STATS_CSV, index=False)
        print(f"\n[OK] CSV estadística: {config.STATS_CSV}")
        print(f"\n{'='*60}")
        print(" RESUMEN ESTADÍSTICO (Wilcoxon + Holm)")
        print(f"{'='*60}")
        for _, r in df_stats.sort_values(["comparison", "wilcoxon_p_holm"]).iterrows():
            print(f"  {r['comparison']} | {r['metric']} | "
                  f"p_holm={r['wilcoxon_p_holm']:.4g} | "
                  f"d={r['cohens_d_paired']:.3f} | "
                  f"reject={r['wilcoxon_reject_holm']}")


# ============================================================
# EXPORTACIÓN ZIP
# ============================================================

def export_zip():
    """Comprime todos los resultados del EXPORT_DIR en un ZIP."""
    ts       = time.strftime("%Y%m%d_%H%M%S")
    zip_path = config.EXPORT_DIR.parent / f"{config.EXPORT_DIR.name}_{ts}.zip"

    n = 0
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for p in config.EXPORT_DIR.rglob("*"):
            if p.is_file():
                zf.write(p, arcname=p.relative_to(config.EXPORT_DIR.parent))
                n += 1

    print(f"[OK] ZIP exportado: {zip_path} ({n} archivos)")
    return zip_path
