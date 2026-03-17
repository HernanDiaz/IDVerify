"""
finetune_sidtd.py — Few-shot fine-tuning de DocVerify sobre SIDTD.

Protocolo:
  1. Split SIDTD en test fijo (20 % estratificado por clase+label) y pool (80 %).
  2. Para cada N en {10, 25, 50, 100, 200, 400, ALL}: samplear N imágenes del pool.
  3. Fine-tunear el modelo pre-entrenado en FantasyID durante max_epochs con
     early stopping sobre un 20 % de los datos de fine-tuning (solo si N>=25).
  4. Evaluar sobre el test fijo y guardar métricas en CSV.

Uso:
    python finetune_sidtd.py

Salida:
    exports_hpo_pareto_nested/sidtd_finetune.csv
"""

from __future__ import annotations

import copy
import gc
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

import config
from dataset import VRAMCache
from dataset_sidtd import build_sidtd_dataframe
from evaluate import eval_model
from model import build_model

# ─────────────────────────────────────────────
# Configuración del experimento
# ─────────────────────────────────────────────
BASE_MODEL_SEEDS  = [42, 43, 44, 45, 46]        # modelos base pre-entrenados
SAMPLE_SEEDS      = [0, 1, 2, 3, 4]             # seeds de muestreo del pool
N_SHOTS_LIST      = [10, 25, 50, 100, 200, 400] # tamaños de fine-tuning
TEST_FRAC         = 0.20                         # fracción reservada como test fijo
FT_LR             = 5e-5                         # lr de fine-tuning
FT_WD             = 1e-5                         # weight decay
FT_MAX_EPOCHS     = 40                           # épocas máximas
FT_PATIENCE       = 8                            # early stopping (si val disponible)
FT_MIN_VAL        = 20                           # mínimo de muestras para usar val split
OUTPUT_CSV        = (config.EXPORT_DIR /
                     "sidtd_finetune.csv")
SIDTD_ROOT        = Path("SIDTD")
MODELS_DIR        = config.MODELS_DIR


def _set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _load_pretrained(seed: int, device: torch.device) -> tuple[nn.Module, dict]:
    """Carga model_multitask_seed<seed>.pt y devuelve (model, params)."""
    ckpt_path = MODELS_DIR / f"model_multitask_seed{seed}.pt"
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    params = ckpt.get("params", {})
    model  = build_model(params, device)
    model.load_state_dict(ckpt["state_dict"])
    return model, params


def _split_sidtd(df: pd.DataFrame, test_frac: float,
                 seed: int = 99) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Divide df en (pool, test) estratificando por sidtd_class + label.
    El test es fijo e igual para todos los experimentos.
    """
    strat_key = df["sidtd_class"] + "_" + df["label"].astype(str)
    sss = StratifiedShuffleSplit(n_splits=1, test_size=test_frac,
                                 random_state=seed)
    pool_idx, test_idx = next(sss.split(df, strat_key))
    return (df.iloc[pool_idx].reset_index(drop=True),
            df.iloc[test_idx].reset_index(drop=True))


def _sample_pool(pool: pd.DataFrame, n_shots: int,
                 sample_seed: int) -> pd.DataFrame:
    """
    Samplea n_shots imágenes del pool equilibrando bonafide/attack.
    Si n_shots >= len(pool), devuelve todo el pool.
    """
    if n_shots >= len(pool):
        return pool.copy()
    n_each  = n_shots // 2
    rng     = np.random.default_rng(sample_seed)
    real_idx = pool[pool["label"] == 0].index.tolist()
    fake_idx = pool[pool["label"] == 1].index.tolist()
    sel_real = rng.choice(real_idx, size=min(n_each, len(real_idx)),
                          replace=False).tolist()
    sel_fake = rng.choice(fake_idx, size=min(n_shots - len(sel_real),
                          len(fake_idx)), replace=False).tolist()
    selected = sorted(sel_real + sel_fake)
    return pool.loc[selected].reset_index(drop=True)


def _make_tensor_loader(cache: VRAMCache, indices: np.ndarray,
                        batch_size: int = 32,
                        shuffle: bool = True) -> DataLoader:
    imgs   = cache.imgs[indices]
    labels = cache.labels[indices]
    masks  = cache.masks[indices]
    ds     = TensorDataset(imgs, labels, masks)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)


def finetune_and_eval(
    base_model: nn.Module,
    params:     dict,
    df_ft:      pd.DataFrame,
    df_test:    pd.DataFrame,
    cache_pool: VRAMCache,
    cache_test: VRAMCache,
    pool_indices:  np.ndarray,
    test_indices:  np.ndarray,
    ft_indices:    np.ndarray,
    device:        torch.device,
    ft_seed:       int,
) -> dict:
    """
    Fine-tunea una COPIA del base_model sobre df_ft y evalúa sobre df_test.
    Devuelve dict de métricas.
    """
    _set_seeds(ft_seed)
    model = copy.deepcopy(base_model).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=FT_LR,
                                  weight_decay=FT_WD)
    scaler = torch.amp.GradScaler("cuda",
                                  enabled=config.USE_AMP and device.type == "cuda")

    # Split fine-tuning en train/val si hay suficientes datos
    # Usamos solo label (0/1) como estrato para evitar clases con 1 muestra
    labels_ft = cache_pool.df.iloc[ft_indices]["label"].values
    use_val   = (len(ft_indices) >= FT_MIN_VAL and
                 (labels_ft == 0).sum() >= 2 and
                 (labels_ft == 1).sum() >= 2)
    if use_val:
        sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2,
                                     random_state=ft_seed)
        tr_rel, va_rel = next(sss.split(ft_indices, labels_ft))
        tr_idx = ft_indices[tr_rel]
        va_idx = ft_indices[va_rel]
    else:
        tr_idx = ft_indices
        va_idx = None

    loader_tr = _make_tensor_loader(cache_pool, tr_idx,
                                    batch_size=min(16, len(tr_idx)),
                                    shuffle=True)

    bce = nn.BCEWithLogitsLoss()
    bce_dice_w = float(params.get("loss_w_mask", 2.0))

    best_val  = -np.inf
    best_state = copy.deepcopy(model.state_dict())
    patience_cnt = 0

    for epoch in range(1, FT_MAX_EPOCHS + 1):
        model.train()
        for imgs_b, labels_b, masks_b in loader_tr:
            imgs_b   = imgs_b.to(device,   non_blocking=True)
            labels_b = labels_b.to(device, non_blocking=True)
            masks_b  = masks_b.to(device,  non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type,
                                 enabled=config.USE_AMP and device.type == "cuda"):
                out      = model(imgs_b)
                loss_cls = bce(out["cls"], labels_b)
                # Dice + BCE para la máscara
                pred_m   = torch.sigmoid(out["mask"])
                dice_num = 2 * (pred_m * masks_b).sum() + 1e-6
                dice_den = pred_m.sum() + masks_b.sum() + 1e-6
                loss_seg = bce(out["mask"], masks_b) + (1 - dice_num / dice_den)
                loss     = loss_cls + bce_dice_w * loss_seg
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        # Early stopping sobre val si disponible
        if use_val and va_idx is not None:
            model.eval()
            loader_va = _make_tensor_loader(cache_pool, va_idx,
                                             batch_size=64, shuffle=False)
            with torch.no_grad():
                metrics_va = eval_model(model, loader_va,
                                        thr_cls=0.5, device=device)
            val_score = metrics_va.get("pr_auc", 0.0)
            if val_score > best_val:
                best_val   = val_score
                best_state = copy.deepcopy(model.state_dict())
                patience_cnt = 0
            else:
                patience_cnt += 1
                if patience_cnt >= FT_PATIENCE:
                    break
        else:
            best_state = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_state)
    model.eval()

    # Evaluar sobre test fijo
    loader_test = _make_tensor_loader(cache_test, test_indices,
                                       batch_size=64, shuffle=False)
    with torch.no_grad():
        metrics = eval_model(model, loader_test, thr_cls=0.5, device=device)

    del model
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return metrics


def run_fewshot_experiment() -> pd.DataFrame:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[FT-SIDTD] Dispositivo: {device}")

    # 1. Cargar SIDTD y crear split test fijo
    print("[FT-SIDTD] Cargando SIDTD ...")
    df_all  = build_sidtd_dataframe(root=SIDTD_ROOT, kind="templates")
    df_pool, df_test = _split_sidtd(df_all, TEST_FRAC, seed=99)
    print(f"[FT-SIDTD] Pool: {len(df_pool)} | Test: {len(df_test)}")
    print(f"           Pool real/fake: {(df_pool.label==0).sum()}/{(df_pool.label==1).sum()}")
    print(f"           Test real/fake: {(df_test.label==0).sum()}/{(df_test.label==1).sum()}")

    # 2. Pre-cargar en VRAM
    print("[FT-SIDTD] Precargando pool en VRAM ...")
    cache_pool = VRAMCache(df_pool, device, label="SIDTD-pool")
    cache_test = VRAMCache(df_test, device,  label="SIDTD-test")
    pool_indices = np.arange(len(df_pool))
    test_indices = np.arange(len(df_test))

    rows = []

    # 3. Baseline zero-shot (N=0) — ya calculado, lo cargamos del CSV existente
    zs_csv = config.EXPORT_DIR / "sidtd_zeroshot.csv"
    if zs_csv.exists():
        df_zs = pd.read_csv(zs_csv)
        mt_zs = df_zs[df_zs["checkpoint"].str.contains("multitask")]
        if not mt_zs.empty:
            for col in ["pr_auc", "roc_auc", "bacc", "dice_global",
                        "dice_pos_mean", "miou"]:
                if col in mt_zs.columns:
                    rows.append({
                        "n_shots":     0,
                        "base_seed":   -1,
                        "sample_seed": -1,
                        col:           float(mt_zs[col].mean()),
                    })
            # Guardar una sola fila resumen para N=0
            rows = []   # limpiar, añadiremos por separado al final
            zs_summary = {
                "n_shots": 0, "base_seed": -1, "sample_seed": -1,
                **{col: float(mt_zs[col].mean())
                   for col in ["pr_auc", "roc_auc", "bacc",
                               "dice_global", "dice_pos_mean", "miou"]
                   if col in mt_zs.columns}
            }
        else:
            zs_summary = None
    else:
        zs_summary = None

    # 4. Fine-tuning para cada N y cada combinación de seeds
    n_shots_all = N_SHOTS_LIST + [len(df_pool)]  # último: ALL

    total_runs = len(BASE_MODEL_SEEDS) * len(SAMPLE_SEEDS) * len(n_shots_all)
    print(f"\n[FT-SIDTD] {total_runs} experimentos de fine-tuning ...")

    pbar = tqdm(total=total_runs, desc="fine-tuning runs")

    for base_seed in BASE_MODEL_SEEDS:
        # Cargar modelo base una sola vez por seed
        print(f"\n[FT-SIDTD] Cargando base model seed={base_seed} ...")
        base_model, params = _load_pretrained(base_seed, device)
        base_model.eval()

        for sample_seed in SAMPLE_SEEDS:
            for n_shots in n_shots_all:
                label = len(df_pool) if n_shots >= len(df_pool) else n_shots

                # Samplear datos de fine-tuning
                df_ft    = _sample_pool(df_pool, n_shots, sample_seed)
                ft_idx   = np.array(
                    [pool_indices[i] for i in
                     df_pool.index.get_indexer(df_ft.index)
                     if pool_indices[i] < len(df_pool)]
                )
                # Rebuild: get actual pool indices for sampled rows
                ft_idx = np.array(df_ft.index.tolist())

                metrics = finetune_and_eval(
                    base_model=base_model,
                    params=params,
                    df_ft=df_ft,
                    df_test=df_test,
                    cache_pool=cache_pool,
                    cache_test=cache_test,
                    pool_indices=pool_indices,
                    test_indices=test_indices,
                    ft_indices=ft_idx,
                    device=device,
                    ft_seed=base_seed * 100 + sample_seed,
                )

                row = {
                    "n_shots":     label,
                    "base_seed":   base_seed,
                    "sample_seed": sample_seed,
                    **metrics,
                }
                rows.append(row)

                pbar.update(1)
                pbar.set_postfix(
                    n=label, bs=base_seed, ss=sample_seed,
                    prauc=f"{metrics.get('pr_auc', 0):.3f}"
                )

        del base_model
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    pbar.close()

    # 5. Añadir baseline zero-shot
    df_results = pd.DataFrame(rows)
    if zs_summary:
        df_zs_row = pd.DataFrame([zs_summary])
        df_results = pd.concat([df_zs_row, df_results], ignore_index=True)

    # 6. Guardar
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df_results.to_csv(OUTPUT_CSV, index=False)
    print(f"\n[FT-SIDTD] Resultados guardados en {OUTPUT_CSV}")

    # 7. Resumen
    print("\n=== RESUMEN (media +/- std por N) ===")
    for n in sorted(df_results["n_shots"].unique()):
        sub = df_results[df_results["n_shots"] == n]
        prauc = sub["pr_auc"].dropna()
        dice  = sub["dice_pos_mean"].dropna()
        label = "ALL" if n == len(df_pool) else str(n)
        print(f"  N={label:>5}: PR-AUC={prauc.mean():.4f}+/-{prauc.std():.4f} "
              f"| Dice+={dice.mean():.4f}+/-{dice.std():.4f}")

    return df_results


if __name__ == "__main__":
    run_fewshot_experiment()
