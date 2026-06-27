"""
augmentation_ablation.py — Ablacion de data augmentation (revision PRL, R1.4 / R2.1.5).

Pregunta que responde:
    El modelo del paper se entrena SIN data augmentation (la cache en VRAM
    precarga tensores fijos). El revisor pide entrenar CON augmentation
    estandar y reportar resultados con y sin ella; si la augmentation degrada,
    "eso tambien es informativo".

Diseno (no invasivo — no toca ningun resultado/peso del paper):
    - Condicion "sin aug" = el blind test del paper (final_blind_test_multiseed.csv,
      variant=multitask, 30 seeds) → se REUTILIZA tal cual como baseline.
    - Condicion "con aug" = se ENTRENA de nuevo con la MISMA config Pareto del
      paper (final_params) y el MISMO procedimiento del blind test
      (_train_with_early_stopping, umbral barrido en val de desarrollo,
      evaluacion en el mismo holdout), inyectando augmentation SOLO en el
      loader de entrenamiento.
    - 30 seeds (config.FINAL_SEEDS) → comparacion pareada por seed.

Augmentation (5, aplicada solo en train, p=0.5 por batch, randomness por-seed):
    1. Rotacion +/-10 deg  (imagen bilineal, MASCARA nearest → sigue binaria)
    2. Brillo  x[0.8,1.2]  (solo imagen)
    3. Contraste x[0.8,1.2] (solo imagen)
    4. Gaussian blur (k=3, sigma in [0.1,2.0]) (solo imagen)
    5. Compresion JPEG (q in [50,95]) (solo imagen, round-trip uint8/CPU)
    Eval/val/holdout: SIN augmentation.

Checkpoints:
    Guarda el modelo augmentado de cada seed en results/augment/checkpoints/
    (model_aug_seedXX.pt, formato compatible con eval_sidtd._load_model) para
    reutilizarlo en una futura generalizacion sin reentrenar. Los no-augmentados
    ya existen en experimento1/models/model_multitask_seedXX.pt.

Salidas (en revision_experiments/results/augment/):
    aug_blind_test.csv — una fila por seed de la condicion con-aug
    aug_summary.csv    — media+/-std por condicion (no_aug reutilizado vs aug)
    aug_stats.csv      — Wilcoxon + Cohen's d pareado (no_aug vs aug, Holm)

Uso:
    python revision_experiments/augmentation_ablation.py
    AUG_SMOKE=1 python revision_experiments/augmentation_ablation.py   # prueba rapida

Requisitos previos:
    - final_blind_test_multiseed.csv debe existir (generado por main.py)
    - El dataset debe estar disponible en config.DATASET_ROOT
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
from torchvision.io import decode_jpeg, encode_jpeg
from torchvision.transforms import functional as TF

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
SMOKE = os.getenv("AUG_SMOKE", "0") == "1"
N_SEEDS = 2 if SMOKE else config.N_FINAL_SEEDS
MAX_EPOCHS_FINAL = 5 if SMOKE else config.MAX_EPOCHS_FINAL
SEEDS = config.FINAL_SEEDS[:N_SEEDS]

# ============================================================
# RUTAS DE SALIDA (carpeta dedicada de revision)
# ============================================================
RESULTS_DIR = Path(__file__).resolve().parent / "results" / "augment"
CKPT_DIR = RESULTS_DIR / "checkpoints"
AUG_CSV = RESULTS_DIR / "aug_blind_test.csv"
SUMMARY_CSV = RESULTS_DIR / "aug_summary.csv"
STATS_CSV = RESULTS_DIR / "aug_stats.csv"

CMP_METRICS = ["test_pr_auc", "test_dice_global", "test_bacc",
               "test_f1_1", "test_f1_macro"]


# ============================================================
# AUGMENTATION (solo en entrenamiento)
# ============================================================

def _jpeg_compress(imgs: torch.Tensor, quality: int) -> torch.Tensor:
    """Round-trip JPEG (uint8/CPU) por muestra; devuelve float[0,1] en el device."""
    x = (imgs.clamp(0, 1) * 255).round().to(torch.uint8).cpu()
    out = []
    for i in range(x.shape[0]):
        out.append(decode_jpeg(encode_jpeg(x[i], quality=int(quality))))
    return torch.stack(out).to(imgs.device).float() / 255.0


def _augment_batch(imgs: torch.Tensor, masks: torch.Tensor,
                   gen: torch.Generator) -> tuple:
    """
    Aplica las 5 augmentations a un batch (p=0.5 cada una, un parametro por batch).
    imgs:  (B,3,H,W) float[0,1] en device.  masks: (B,1,H,W) float{0,1} en device.
    Geometrica (rotacion) → imagen Y mascara.  Fotometricas → solo imagen.
    """
    def _u(lo, hi):
        return lo + (hi - lo) * float(torch.rand(1, generator=gen).item())

    def _flip():
        return float(torch.rand(1, generator=gen).item()) < 0.5

    # 1. Rotacion (geometrica → imagen bilineal + mascara nearest)
    if _flip():
        angle = _u(-10.0, 10.0)
        imgs = TF.rotate(imgs, angle, interpolation=TF.InterpolationMode.BILINEAR)
        masks = TF.rotate(masks, angle, interpolation=TF.InterpolationMode.NEAREST)
    # 2. Brillo
    if _flip():
        imgs = TF.adjust_brightness(imgs, _u(0.8, 1.2))
    # 3. Contraste
    if _flip():
        imgs = TF.adjust_contrast(imgs, _u(0.8, 1.2))
    # 4. Gaussian blur
    if _flip():
        sigma = _u(0.1, 2.0)
        imgs = TF.gaussian_blur(imgs, kernel_size=3, sigma=[sigma, sigma])
    # 5. Compresion JPEG
    if _flip():
        imgs = _jpeg_compress(imgs, int(round(_u(50, 95))))

    return imgs.clamp(0, 1), masks


class AugmentingLoader:
    """
    Envuelve un DataLoader de entrenamiento aplicando augmentation a cada batch.
    Re-iterable: cada epoca produce augmentation nueva. El generador por-seed
    garantiza reproducibilidad. No modifica el loader base (shuffle/cache intactos).
    """

    def __init__(self, base_loader, seed: int):
        self.base = base_loader
        self.gen = torch.Generator().manual_seed(int(seed) + 9973)

    def __iter__(self):
        for imgs, labels, masks in self.base:
            imgs, masks = _augment_batch(imgs, masks, self.gen)
            yield imgs, labels, masks

    def __len__(self):
        return len(self.base)

    def __getattr__(self, name):
        # Delegar cualquier otro atributo (p.ej. .dataset) al loader base
        return getattr(self.base, name)


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
# CARGA DE LA CONFIG FINAL DEL PAPER Y DEL BASELINE
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
    params["dec_ch"] = int(params["dec_ch"])
    params["dropout_rate"] = float(params["dropout_rate"])
    params["lr"] = float(params["lr"])
    params["weight_decay"] = float(params["weight_decay"])
    params["loss_w_mask"] = float(params["loss_w_mask"])
    if "alpha" in params:
        params["alpha"] = float(params["alpha"])
    return params


def _load_baseline_rows() -> pd.DataFrame:
    """Reutiliza las 30 filas multitask del blind test del paper como condicion no_aug."""
    df = pd.read_csv(config.FINAL_TEST_CSV)
    if "variant" in df.columns:
        df = df[df["variant"] == "multitask"].reset_index(drop=True)
    out = pd.DataFrame()
    out["seed"] = df["seed"].astype(int)
    out["condition"] = "no_aug"
    for m in CMP_METRICS:
        out[m] = df[m].astype(float)
    return out


# ============================================================
# ENTRENAMIENTO DE UN MODELO CON AUGMENTATION
# ============================================================

def _train_one_aug(final_params: dict, seed: int, df_dev: pd.DataFrame,
                   df_holdout: pd.DataFrame, device: torch.device) -> dict:
    """
    Replica el procedimiento del blind test (_train_final_model) inyectando
    augmentation SOLO en el loader de entrenamiento. Guarda el checkpoint.
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

    # Inyectar augmentation SOLO en entrenamiento
    aug_loader_tr = AugmentingLoader(loader_tr, seed=seed)

    run_params = dict(final_params)
    model = _maybe_compile(build_model(run_params, device))
    optimizer = build_optimizer(model, run_params)
    scaler = torch.amp.GradScaler("cuda",
                                  enabled=config.USE_AMP and device.type == "cuda")

    t0 = time.perf_counter()
    model = _train_with_early_stopping(
        model, optimizer, scaler, aug_loader_tr, loader_sel, device,
        params=run_params, max_epochs=MAX_EPOCHS_FINAL,
        patience=12, variant="multitask",
        desc=f"[aug seed={seed}]",
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

    # Guardar checkpoint (state_dict limpio, sin prefijo _orig_mod si esta compilado)
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    clean = getattr(model, "_orig_mod", model)
    torch.save(
        {"model_state_dict": clean.state_dict(), "params": run_params,
         "seed": int(seed), "variant": "multitask_aug"},
        CKPT_DIR / f"model_aug_seed{seed}.pt",
    )

    row = {
        "seed": int(seed),
        "condition": "aug",
        "train_time_sec": float(train_time),
        "thr_cls_from_val_sel": float(thr_bacc),
        **{f"test_{k}": v for k, v in met.items()},
        **{f"hp_{k}": v for k, v in final_params.items()},
    }
    row["test_cm_TN_FP_FN_TP"] = json.dumps(met["cm_TN_FP_FN_TP"])

    del model, optimizer, loader_tr, loader_sel, loader_test, aug_loader_tr
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return row


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
    print("\n" + "=" * 64)
    print(" RESUMEN — Augmentation ablation (media +/- std)")
    print("=" * 64)
    label = {"no_aug": "Sin augmentation", "aug": "Con augmentation"}
    for _, r in df_summary.iterrows():
        print(f"  [{label.get(r['condition'], r['condition']):>18}] (n={int(r['n_seeds'])}) "
              f"PR-AUC={r['test_pr_auc_mean']:.4f}+/-{r['test_pr_auc_std']:.4f} | "
              f"Dice={r['test_dice_global_mean']:.4f}+/-{r['test_dice_global_std']:.4f} | "
              f"F1macro={r['test_f1_macro_mean']:.4f}+/-{r['test_f1_macro_std']:.4f}")

    if not df_stats.empty:
        print("\n" + "=" * 64)
        print(" ESTADISTICA — pareado no_aug vs aug (Wilcoxon + Holm)")
        print("=" * 64)
        for _, r in df_stats.iterrows():
            print(f"  {r['metric']:>16} | Delta={r['delta_aug_minus_noaug']:+.4f} | "
                  f"p_holm={r['wilcoxon_p_holm']:.4g} | d={r['cohens_d_paired']:+.3f} | "
                  f"reject={r['wilcoxon_reject_holm']}")


# ============================================================
# MAIN
# ============================================================

def main():
    print("\n" + "=" * 64)
    print(" DocVerify — Augmentation ablation (revision R1.4 / R2.1.5)")
    print("=" * 64 + "\n")

    if SMOKE:
        print(f"[WARN] AUG_SMOKE=1 -> N_SEEDS={N_SEEDS} MAX_EPOCHS_FINAL={MAX_EPOCHS_FINAL}\n")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    final_params = _load_final_params()
    print(f"[OK] final_params (paper): {final_params}")

    device = get_device()
    print("\n[INFO] Cargando dataset...")
    df_dev, df_holdout = _load_dataset()
    print(f"[OK] df_dev={len(df_dev)}  df_holdout={len(df_holdout)}")
    print(f"[OK] Augmentation: rotacion+/-10, brillo, contraste, blur, JPEG (solo train)")
    print(f"[OK] baseline (no_aug) reutilizado de {config.FINAL_TEST_CSV.name}")
    print(f"[OK] seeds: {SEEDS}  (n={len(SEEDS)})")
    print(f"[OK] checkpoints -> {CKPT_DIR}")

    # ── Entrenar la condicion con-aug (no_aug se reutiliza) ──────────────
    new_rows = []
    for seed in SEEDS:
        print(f"\n  [RUN] aug seed={seed}")
        row = _train_one_aug(final_params, seed, df_dev, df_holdout, device)
        _append_row_csv(AUG_CSV, row)
        new_rows.append(row)
        print(f"    PR-AUC={row['test_pr_auc']:.4f} | "
              f"Dice={row['test_dice_global']:.4f} | "
              f"F1macro={row['test_f1_macro']:.4f}")

    # ── Combinar con el baseline reutilizado ─────────────────────────────
    df_new = pd.DataFrame(new_rows)
    df_baseline = _load_baseline_rows()
    keep = ["seed", "condition", *CMP_METRICS]
    df_all = pd.concat([df_baseline[keep], df_new[keep]], ignore_index=True)

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
