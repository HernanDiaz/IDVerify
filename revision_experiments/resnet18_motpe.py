"""
resnet18_motpe.py — ¿Es el método MOTPE/Pareto agnóstico a la arquitectura?
                    (revisión PRL, R1.7).

Pregunta que responde:
    La contribución del paper es el MÉTODO de optimización multi-objetivo
    (MOTPE + selección por distancia al punto ideal), no la arquitectura.
    El revisor (R1.7) pide demostrar que el método funciona con un backbone
    moderno preentrenado. Aquí ejecutamos EXACTAMENTE el mismo pipeline MOTPE
    sobre un encoder ResNet-18 (ImageNet) y evaluamos la config Pareto
    resultante en el holdout ciego, comparándola con DocVerify.

Diseño (no invasivo — no toca ningún resultado/peso/CSV del paper):
    1) Se reconstruye el MISMO split 75/10/15 del paper (random_state=42) →
       df_dev (desarrollo) y df_holdout (test ciego).
    2) Se ejecuta un estudio MOTPE con la MISMA metodología del paper:
         - mismo espacio de búsqueda (lr, weight_decay, dropout, dec_ch,
           loss_w_mask), misma N_INNER=5 inner folds, misma N_TRIALS=50,
           mismo MAX_EPOCHS_TRIAL=15, mismo sampler MOTPE, misma selección
           Pareto por distancia mínima a (1,1).
       La ÚNICA diferencia es el encoder: ResNet-18 en lugar del CNN de Patel.
    3) Con la config Pareto seleccionada, se entrena en el holdout ciego con
       30 seeds (config.FINAL_SEEDS), igual que el blind test del paper, y se
       compara con DocVerify (reutilizando final_blind_test_multiseed.csv).

Garantías de no-invasividad:
    - Solo LEE final_blind_test_multiseed.csv (baseline DocVerify).
    - NUNCA llama a torch.save (no escribe pesos).
    - El estudio Optuna es in-memory (no toca config.SQLITE_PATH).
    - Escribe SOLO en revision_experiments/results/resnet18/.
    - No modifica config.py, model.py, train.py ni ningún CSV del paper.
    - Totalmente reproducible: seeds y splits deterministas.

Salidas (en revision_experiments/results/resnet18/):
    resnet18_trials.csv      — un registro por trial MOTPE (PR-AUC, Dice, dist)
    resnet18_pareto.csv      — frente de Pareto del estudio
    resnet18_selected.json   — config Pareto seleccionada (distancia al ideal)
    resnet18_blind_test.csv  — una fila por seed (métricas en holdout)
    resnet18_summary.csv     — media±std ResNet-18 vs DocVerify
    resnet18_stats.csv       — Wilcoxon + Cohen's d pareado (ResNet-18 vs DocVerify)

Uso:
    python revision_experiments/resnet18_motpe.py
    # Prueba rápida de humo (config reducida, NO para el paper):
    RESNET18_SMOKE=1 python revision_experiments/resnet18_motpe.py

Requisitos previos:
    - final_blind_test_multiseed.csv debe existir (generado por main.py).
    - El dataset debe estar disponible en config.DATASET_ROOT.
    - Acceso para descargar los pesos ImageNet de ResNet-18 (la primera vez).
"""

import gc
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import GroupShuffleSplit
from torchvision.models import ResNet18_Weights, resnet18

# Permitir ejecutar el script desde cualquier directorio
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
import evaluate as ev
from model import DecBlock
from scalar_experiment import _load_dataset
from train import (
    _append_row_csv,
    _eval_prauc_dice,
    _get_pareto_trials,
    _make_sampler,
    _make_sgkf,
    _maybe_compile,
    _select_best_trial,
    _set_seeds,
    _train_one_epoch,
    _train_with_early_stopping,
    get_device,
)

try:
    import optuna
except ImportError:
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "optuna", "-q"])
    import optuna

# ============================================================
# CONFIG LOCAL (knobs de prueba de humo; por defecto = paper)
# ============================================================
SMOKE = os.getenv("RESNET18_SMOKE", "0") == "1"
N_TRIALS         = 2  if SMOKE else config.N_TRIALS
N_SEEDS          = 2  if SMOKE else config.N_FINAL_SEEDS
MAX_EPOCHS_TRIAL = 2  if SMOKE else config.MAX_EPOCHS_TRIAL
MAX_EPOCHS_FINAL = 5  if SMOKE else config.MAX_EPOCHS_FINAL
SEEDS            = config.FINAL_SEEDS[:N_SEEDS]

# ============================================================
# RUTAS DE SALIDA (carpeta dedicada de revisión)
# ============================================================
RESULTS_DIR  = Path(__file__).resolve().parent / "results" / "resnet18"
TRIALS_CSV   = RESULTS_DIR / "resnet18_trials.csv"
PARETO_CSV   = RESULTS_DIR / "resnet18_pareto.csv"
SELECTED_JSON = RESULTS_DIR / "resnet18_selected.json"
BLIND_CSV    = RESULTS_DIR / "resnet18_blind_test.csv"
SUMMARY_CSV  = RESULTS_DIR / "resnet18_summary.csv"
STATS_CSV    = RESULTS_DIR / "resnet18_stats.csv"

# Métricas comparadas contra DocVerify (presentes en final_blind_test_multiseed.csv)
CMP_METRICS = ["test_pr_auc", "test_dice_global", "test_bacc",
               "test_f1_1", "test_f1_macro"]


# ============================================================
# MODELO: ResNet-18 (ImageNet) encoder + U-Net decoder + cabeza cls
# ============================================================

class ResNet18UNet(nn.Module):
    """
    Encoder ResNet-18 preentrenado (ImageNet) + decoder U-Net + cabeza de
    clasificación, con la MISMA interfaz que DocVerifyModel (devuelve
    {"cls", "mask"} en logits). Mantiene el decoder y la cabeza del paper;
    solo cambia el encoder. La normalización ImageNet se aplica dentro del
    forward para usar correctamente los pesos preentrenados (la entrada llega
    en [0,1] como en el pipeline del paper).

    Skips de ResNet-18 (entrada 224):
        stem  (conv1+bn+relu) → 112×112,  64ch
        layer1                → 56×56,    64ch
        layer2                → 28×28,   128ch
        layer3                → 14×14,   256ch
        layer4 (bottleneck)   → 7×7,     512ch
    """

    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD  = [0.229, 0.224, 0.225]

    def __init__(self, dropout_rate: float = 0.3, dec_ch: int = 128,
                 alpha: float = 0.2, pretrained: bool = True):
        super().__init__()
        weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        net = resnet18(weights=weights)

        # Encoder (skips a 112/56/28/14/7)
        self.stem    = nn.Sequential(net.conv1, net.bn1, net.relu)  # → 112, 64
        self.maxpool = net.maxpool                                  # → 56
        self.layer1  = net.layer1   # 56,  64
        self.layer2  = net.layer2   # 28, 128
        self.layer3  = net.layer3   # 14, 256
        self.layer4  = net.layer4   # 7,  512

        self.register_buffer("mean", torch.tensor(self.IMAGENET_MEAN).view(1, 3, 1, 1))
        self.register_buffer("std",  torch.tensor(self.IMAGENET_STD).view(1, 3, 1, 1))

        # Cabeza de clasificación (idéntica a DocVerify salvo la entrada 512)
        self.cls_gap = nn.AdaptiveAvgPool2d(1)
        self.cls_head = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(512, 32), nn.LeakyReLU(alpha, inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(32, 16),  nn.LeakyReLU(alpha, inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(16, 16),  nn.LeakyReLU(alpha, inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(16, 1),
        )

        # Decoder U-Net (mismos DecBlock del paper, canales adaptados a ResNet-18)
        self.mask_proj = nn.Sequential(
            nn.Conv2d(512, dec_ch, 1, bias=False),
            nn.LeakyReLU(alpha, inplace=True),
        )
        self.dec14  = DecBlock(dec_ch,      256, dec_ch,      alpha)  # 7  → 14  (skip layer3)
        self.dec28  = DecBlock(dec_ch,      128, dec_ch // 2, alpha)  # 14 → 28  (skip layer2)
        self.dec56  = DecBlock(dec_ch // 2,  64, dec_ch // 4, alpha)  # 28 → 56  (skip layer1)
        self.dec112 = DecBlock(dec_ch // 4,  64, dec_ch // 8, alpha)  # 56 → 112 (skip stem)
        self.mask_out = nn.Conv2d(dec_ch // 8, 1, 1)

    def forward(self, x: torch.Tensor) -> dict:
        H, W = x.shape[2], x.shape[3]
        x = (x - self.mean) / self.std

        f112 = self.stem(x)         # 112, 64
        p    = self.maxpool(f112)   # 56
        f56  = self.layer1(p)       # 56,  64
        f28  = self.layer2(f56)     # 28, 128
        f14  = self.layer3(f28)     # 14, 256
        f7   = self.layer4(f14)     # 7,  512

        c = self.cls_gap(f7).flatten(1)
        cls_out = self.cls_head(c)  # logits (B, 1)

        m = self.mask_proj(f7)
        m = self.dec14(m,  f14)
        m = self.dec28(m,  f28)
        m = self.dec56(m,  f56)
        m = self.dec112(m, f112)
        m = F.interpolate(m, size=(H, W), mode="bilinear", align_corners=False)
        mask_out = self.mask_out(m)  # logits (B, 1, H, W)

        return {"cls": cls_out, "mask": mask_out}


def build_resnet_model(params: dict, device: torch.device) -> ResNet18UNet:
    model = ResNet18UNet(
        dropout_rate = float(params["dropout_rate"]),
        dec_ch       = int(params["dec_ch"]),
        alpha        = float(params.get("alpha", config.LEAKY_RELU_ALPHA)),
        pretrained   = True,
    )
    return model.to(device)


def build_resnet_optimizer(model: nn.Module, params: dict) -> torch.optim.Optimizer:
    return torch.optim.AdamW(
        model.parameters(),
        lr           = float(params["lr"]),
        weight_decay = float(params["weight_decay"]),
    )


# ============================================================
# OBJETIVO MOTPE (mismo esquema que train._make_inner_objective)
# ============================================================

def _make_inner_objective(cache, df_dev: pd.DataFrame, device: torch.device):
    # outer_fold_id=0 → semillas análogas al paper (SEED_BASE+100, +10*inner_id)
    splitter = _make_sgkf(config.N_INNER, seed=config.SEED_BASE + 100)
    y = df_dev["label"].values
    g = df_dev["stem"].values

    def objective(trial: optuna.Trial) -> tuple[float, float]:
        t0 = time.perf_counter()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        params = {
            "lr":           trial.suggest_float("lr", 5e-5, 9e-4, log=True),
            "weight_decay": trial.suggest_float("weight_decay", 1e-7, 1e-4, log=True),
            "dropout_rate": trial.suggest_float("dropout_rate", 0.1, 0.4),
            "alpha":        0.2,
            "dec_ch":       trial.suggest_categorical("dec_ch", [96, 128, 192, 256]),
            "loss_w_mask":  trial.suggest_float("loss_w_mask", 0.5, 3.0),
        }

        fold_metrics = []
        try:
            for inner_id, (tr_idx, va_idx) in enumerate(
                splitter.split(df_dev, y=y, groups=g)
            ):
                seed_fold = config.SEED_BASE + 10 * inner_id
                _set_seeds(seed_fold)

                loader_tr = cache.make_loader(tr_idx, training=True,  seed=seed_fold)
                loader_va = cache.make_loader(va_idx, training=False, seed=seed_fold)

                model     = _maybe_compile(build_resnet_model(params, device))
                optimizer = build_resnet_optimizer(model, params)
                scaler    = torch.amp.GradScaler(
                    "cuda", enabled=config.USE_AMP and device.type == "cuda")

                for epoch in range(1, MAX_EPOCHS_TRIAL + 1):
                    _train_one_epoch(
                        model, loader_tr, optimizer, scaler, device,
                        lw_cls=1.0, lw_mask=float(params["loss_w_mask"]),
                        epoch=epoch, max_epochs=MAX_EPOCHS_TRIAL,
                        desc=f"[resnet trial={trial.number} inner={inner_id}]",
                    )

                pr_auc, dice = _eval_prauc_dice(model, loader_va, device)
                fold_metrics.append({"prauc": pr_auc, "dice": dice})

                del model, optimizer
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        except torch.cuda.OutOfMemoryError:
            _append_row_csv(TRIALS_CSV, {
                "trial_number": trial.number, "pruned": True,
                "time_sec": time.perf_counter() - t0, **params, "notes": "OOM",
            })
            raise optuna.exceptions.TrialPruned("CUDA OOM")

        pr_mean   = float(np.nanmean([m["prauc"] for m in fold_metrics]))
        dc_mean   = float(np.nanmean([m["dice"]  for m in fold_metrics]))
        dist_mean = math.sqrt((1.0 - pr_mean) ** 2 + (1.0 - dc_mean) ** 2)

        _append_row_csv(TRIALS_CSV, {
            "trial_number":         trial.number,
            "pruned":               False,
            "time_sec":             time.perf_counter() - t0,
            **params,
            "val_cls_prauc":        pr_mean,
            "val_mask_dice_global": dc_mean,
            "distance_to_ideal":    dist_mean,
        })

        return pr_mean, dc_mean

    return objective


# ============================================================
# ENTRENAMIENTO FINAL (un seed) EN EL HOLDOUT CIEGO
# ============================================================

def _train_final(params: dict, seed: int, df_dev: pd.DataFrame,
                 df_holdout: pd.DataFrame, device: torch.device) -> dict:
    """Réplica de train._train_final_model (variant=multitask) con ResNet-18.
    NO guarda pesos; escribe solo en la carpeta de revisión."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    _set_seeds(seed)

    gss = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=seed)
    tr_idx, sel_idx = next(gss.split(df_dev, y=df_dev["label"], groups=df_dev["stem"]))

    from dataset import make_dataloader
    loader_tr   = make_dataloader(df_dev.iloc[tr_idx].reset_index(drop=True),
                                  training=True,  seed=seed, device=device)
    loader_sel  = make_dataloader(df_dev.iloc[sel_idx].reset_index(drop=True),
                                  training=False, seed=seed, device=device)
    loader_test = make_dataloader(df_holdout, training=False, seed=seed, device=device)

    model     = _maybe_compile(build_resnet_model(params, device))
    optimizer = build_resnet_optimizer(model, params)
    scaler    = torch.amp.GradScaler("cuda", enabled=config.USE_AMP and device.type == "cuda")

    t0 = time.perf_counter()
    model = _train_with_early_stopping(
        model, optimizer, scaler, loader_tr, loader_sel, device,
        params, MAX_EPOCHS_FINAL, patience=12, variant="multitask",
        desc=f"[resnet18 seed={seed}]",
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

    row = {
        "seed": int(seed), "variant": "resnet18_multitask",
        "train_time_sec": float(train_time),
        "thr_cls_from_val_sel": float(thr_bacc),
        "val_sel_best_bacc": float(best_bacc),
        "val_sel_best_f1m": float(best_f1m),
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
# ESTADÍSTICA: ResNet-18 vs DocVerify (pareado por seed)
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


def _build_summary(df_resnet: pd.DataFrame, df_docverify: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for name, d in [("DocVerify", df_docverify), ("ResNet18-MOTPE", df_resnet)]:
        row = {"model": name, "n_seeds": len(d)}
        for m in CMP_METRICS:
            vals = d[m].dropna()
            row[f"{m}_mean"] = float(vals.mean())
            row[f"{m}_std"]  = float(vals.std(ddof=1)) if len(vals) > 1 else 0.0
        rows.append(row)
    return pd.DataFrame(rows)


def _run_stats(df_resnet: pd.DataFrame, df_docverify: pd.DataFrame) -> pd.DataFrame:
    try:
        from scipy.stats import wilcoxon
    except ImportError:
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "scipy", "-q"])
        from scipy.stats import wilcoxon

    mrg = df_docverify.merge(df_resnet, on="seed", suffixes=("_dv", "_rn"))
    if len(mrg) < 3:
        print("[STATS] n insuficiente para comparar. Saltando.")
        return pd.DataFrame()

    pvals, tmp = [], []
    for m in CMP_METRICS:
        a = mrg[f"{m}_dv"].values.astype(float)   # DocVerify
        b = mrg[f"{m}_rn"].values.astype(float)   # ResNet-18
        try:
            p_w = float(wilcoxon(a, b, zero_method="wilcox").pvalue)
        except Exception:
            p_w = float("nan")
        pvals.append(p_w)
        tmp.append({
            "comparison":      "DocVerify vs ResNet18-MOTPE",
            "metric":          m,
            "n_paired":        len(mrg),
            "mean_docverify":  float(np.mean(a)),
            "mean_resnet18":   float(np.mean(b)),
            "delta_dv_minus_rn": float(np.mean(a) - np.mean(b)),
            "wilcoxon_p":      p_w,
            "cohens_d_paired": _cohens_d(a, b),
        })

    adj, rej = _holm_bonferroni(pvals)
    for i, r in enumerate(tmp):
        r["wilcoxon_p_holm"]      = float(adj[i])
        r["wilcoxon_reject_holm"] = bool(rej[i])
    return pd.DataFrame(tmp)


def _load_docverify_baseline() -> pd.DataFrame:
    if not config.FINAL_TEST_CSV.exists():
        raise FileNotFoundError(
            f"No se encuentra {config.FINAL_TEST_CSV}.\n"
            f"Ejecuta primero main.py para generar el blind test del paper."
        )
    df = pd.read_csv(config.FINAL_TEST_CSV)
    if "variant" in df.columns:
        df = df[df["variant"] == "multitask"].reset_index(drop=True)
    keep = ["seed", *CMP_METRICS]
    return df[keep].copy()


# ============================================================
# MAIN
# ============================================================

def main():
    print("\n" + "=" * 64)
    print(" DocVerify — MOTPE sobre ResNet-18 (revisión R1.7)")
    print("=" * 64 + "\n")
    if SMOKE:
        print("[WARN] RESNET18_SMOKE=1 -> config reducida (prueba de humo, NO para el paper)")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    device = get_device()
    print("[INFO] Cargando dataset...")
    df_dev, df_holdout = _load_dataset()
    print(f"[OK] df_dev={len(df_dev)}  df_holdout={len(df_holdout)}")
    print(f"[OK] MOTPE: N_TRIALS={N_TRIALS} N_INNER={config.N_INNER} "
          f"MAX_EPOCHS_TRIAL={MAX_EPOCHS_TRIAL}")
    print(f"[OK] Blind test: {N_SEEDS} seeds {SEEDS[0]}..{SEEDS[-1]} "
          f"MAX_EPOCHS_FINAL={MAX_EPOCHS_FINAL}")

    # ── 1) Estudio MOTPE sobre ResNet-18 (cache en VRAM, in-memory study) ──
    from dataset import VRAMCache
    cache = VRAMCache(df_dev, device, label=f"df_dev ({len(df_dev)} imgs)")

    study = optuna.create_study(
        study_name = "DOCVERIFY_RESNET18",
        directions = ["maximize", "maximize"],
        sampler    = _make_sampler(config.SEED_BASE),
        pruner     = optuna.pruners.NopPruner(),
    )  # in-memory: no storage → no toca config.SQLITE_PATH

    objective = _make_inner_objective(cache, df_dev, device)
    t0 = time.perf_counter()
    study.optimize(objective, n_trials=N_TRIALS, gc_after_trial=True)
    t1 = time.perf_counter()

    cache.free()
    del cache

    best_trial, best_dist = _select_best_trial(study)
    if best_trial is None:
        raise RuntimeError("No hay trials completados.")
    pareto = _get_pareto_trials(study)
    print(f"\n[HPO] Tiempo: {t1 - t0:.1f}s | Pareto: {len(pareto)} trials")
    print(f"[HPO] Trial #{best_trial.number} | dist={best_dist:.4f} | "
          f"(PR-AUC, Dice)={best_trial.values}")

    for t in pareto:
        pr, dc = (t.values or [np.nan, np.nan])
        _append_row_csv(PARETO_CSV, {
            "trial_number": t.number,
            "val_cls_prauc": float(pr), "val_mask_dice_global": float(dc),
            "distance_to_ideal": math.sqrt((1 - float(pr)) ** 2 + (1 - float(dc)) ** 2),
            **t.params,
        })

    final_params = dict(best_trial.params)
    final_params["alpha"] = 0.2
    with open(SELECTED_JSON, "w", encoding="utf-8") as f:
        json.dump({
            "selected_trial": best_trial.number,
            "distance_to_ideal": best_dist,
            "val_pr_auc": float(best_trial.values[0]),
            "val_dice": float(best_trial.values[1]),
            "params": final_params,
        }, f, indent=2)
    print(f"[OK] Config Pareto guardada: {SELECTED_JSON}")
    print(f"[OK] final_params: {final_params}")

    # ── 2) Blind test con 30 seeds (config Pareto del backbone) ───────────
    blind_rows = []
    for seed in SEEDS:
        print(f"\n  [BLIND] resnet18 seed={seed}")
        row = _train_final(final_params, seed, df_dev, df_holdout, device)
        _append_row_csv(BLIND_CSV, row)
        blind_rows.append(row)
        print(f"    PR-AUC={row['test_pr_auc']:.4f} | "
              f"Dice={row['test_dice_global']:.4f} | "
              f"F1macro={row['test_f1_macro']:.4f}")

    df_resnet = pd.DataFrame(blind_rows)[["seed", *CMP_METRICS]]

    # ── 3) Comparación con DocVerify ──────────────────────────────────────
    df_docverify = _load_docverify_baseline()

    df_summary = _build_summary(df_resnet, df_docverify)
    df_summary.to_csv(SUMMARY_CSV, index=False)
    print(f"\n[OK] Resumen guardado: {SUMMARY_CSV}")

    df_stats = _run_stats(df_resnet, df_docverify)
    if not df_stats.empty:
        df_stats.to_csv(STATS_CSV, index=False)
        print(f"[OK] Estadística guardada: {STATS_CSV}")

    # ── Resumen en consola ────────────────────────────────────────────────
    print(f"\n{'='*64}")
    print(" RESUMEN — ResNet18-MOTPE vs DocVerify (media ± std)")
    print(f"{'='*64}")
    for _, r in df_summary.iterrows():
        print(f"  [{r['model']:>16}] (n={int(r['n_seeds'])}) "
              f"PR-AUC={r['test_pr_auc_mean']:.4f}±{r['test_pr_auc_std']:.4f} | "
              f"Dice={r['test_dice_global_mean']:.4f}±{r['test_dice_global_std']:.4f} | "
              f"F1macro={r['test_f1_macro_mean']:.4f}±{r['test_f1_macro_std']:.4f}")
    if not df_stats.empty:
        print(f"\n{'='*64}")
        print(" ESTADÍSTICA — pareado por seed (Wilcoxon + Holm)")
        print(f"{'='*64}")
        for _, r in df_stats.sort_values("wilcoxon_p_holm").iterrows():
            print(f"  {r['metric']} | Δ(dv-rn)={r['delta_dv_minus_rn']:+.4f} | "
                  f"p_holm={r['wilcoxon_p_holm']:.4g} | d={r['cohens_d_paired']:+.3f} | "
                  f"reject={r['wilcoxon_reject_holm']}")

    print(f"\n[OK] Experimento completado.")
    print(f"[OK] Resultados en: {RESULTS_DIR.resolve()}")


if __name__ == "__main__":
    main()
