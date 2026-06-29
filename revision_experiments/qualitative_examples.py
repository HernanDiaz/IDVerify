"""
qualitative_examples.py — Figura cualitativa de R1.8 (revision PRL).

El revisor 1 pide ejemplos cualitativos que ilustren el comportamiento del
modelo, incluyendo fallos. Genera una figura 3x3 (paper/prltemplate/figures):
    Fila A — Acierto: ataque detectado con buena localizacion (Dice alto).
    Fila B — Fallo de localizacion: ataque detectado correctamente (clasif.),
             pero la mascara predicha no cubre la region alterada (Dice bajo).
    Fila C — Falso positivo bonafide: documento autentico (sin region alterada)
             clasificado como ataque, con mascara espuria.
Columnas: Entrada | Verdad (GT) | Prediccion del modelo.

NO reentrena: reutiliza el checkpoint del blind test del paper
(model_multitask_seed42.pt) y su umbral guardado (thr_cls_from_val_sel).

Uso:
    venv/Scripts/python.exe revision_experiments/qualitative_examples.py
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

# Igualar tipografia al resto de figuras del paper (IEEEStyle: serif Times 9pt)
matplotlib.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 9,
    "axes.titlesize": 9,
    "axes.labelsize": 9,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
from dataset import make_dataloader
from model import build_model
from scalar_experiment import _load_dataset
from train import get_device

SEED = 42
CKPT = config.MODELS_DIR / f"model_multitask_seed{SEED}.pt"
OUT_DIR = Path(__file__).resolve().parent.parent / "paper" / "prltemplate" / "figures"
OUT_STEM = OUT_DIR / "fig_qualitative"


def _dice(pred: np.ndarray, gt: np.ndarray) -> float:
    p = pred.astype(bool)
    g = gt.astype(bool)
    inter = np.logical_and(p, g).sum()
    denom = p.sum() + g.sum()
    if denom == 0:
        return 1.0  # ambos vacios: acuerdo perfecto (caso bonafide bien resuelto)
    return float(2.0 * inter / denom)


def _load_threshold(seed: int) -> float:
    df = pd.read_csv(config.FINAL_TEST_CSV)
    df = df[(df["seed"] == seed) & (df["variant"] == "multitask")]
    return float(df["thr_cls_from_val_sel"].iloc[0])


def main():
    device = get_device()
    thr = _load_threshold(SEED)
    print(f"[INFO] checkpoint={CKPT.name}  thr_cls={thr:.4f}  device={device}")

    ckpt = torch.load(CKPT, map_location=device, weights_only=False)
    model = build_model(ckpt["params"], device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    _, df_holdout = _load_dataset()
    print(f"[INFO] holdout={len(df_holdout)} imagenes")

    loader = make_dataloader(df_holdout, training=False, seed=SEED, device=device)

    imgs_all, probs, pred_masks, gt_masks, labels = [], [], [], [], []
    with torch.no_grad():
        for imgs, lab, masks in loader:
            if imgs.device != device:
                imgs = imgs.to(device, non_blocking=True)
            out = model(imgs)
            p = torch.sigmoid(out["cls"].float()).cpu().numpy().reshape(-1)
            pm = (torch.sigmoid(out["mask"].float()) > 0.5).cpu().numpy()[:, 0]
            imgs_all.append(imgs.cpu().numpy())
            probs.append(p)
            pred_masks.append(pm)
            gt_masks.append(masks.cpu().numpy()[:, 0])
            labels.append(lab.cpu().numpy().reshape(-1))

    imgs_all = np.concatenate(imgs_all)          # (N,3,224,224) in [0,1]
    probs = np.concatenate(probs)                # (N,)
    pred_masks = np.concatenate(pred_masks)      # (N,224,224) bool
    gt_masks = np.concatenate(gt_masks)          # (N,224,224) {0,1}
    labels = np.concatenate(labels).astype(int)  # (N,)

    dice = np.array([_dice(pred_masks[i], gt_masks[i]) for i in range(len(labels))])
    gt_area = gt_masks.reshape(len(labels), -1).sum(1)

    is_attack = labels == 1
    pred_attack = probs >= thr

    # Marca las imagenes cuya region alterada incluye la foto (field_name == "face")
    def _has_face(rects_json: str) -> bool:
        try:
            rects = json.loads(rects_json or "[]")
        except Exception:
            return False
        return any((r.get("field_name") or "").strip().lower() == "face"
                   for r in rects)
    has_face = df_holdout["mask_rects_abs"].apply(_has_face).to_numpy()

    # Fila A — acierto: ataque con la FOTO alterada, detectado y bien localizado
    cand_face = np.where(is_attack & pred_attack & (gt_area > 0) & has_face)[0]
    idx_success = cand_face[np.argmax(dice[cand_face])]

    # Fila B — fallo de localizacion: ataque detectado pero Dice bajo (GT real)
    cand_all = np.where(is_attack & pred_attack & (gt_area > 0))[0]
    idx_locfail = cand_all[np.argmin(dice[cand_all])]

    # Fila C — falso positivo bonafide: label 0, predicho ataque, prob mas alta
    fp = np.where((labels == 0) & pred_attack)[0]
    if len(fp) == 0:
        fp = np.where(labels == 0)[0]
    idx_fp = fp[np.argmax(probs[fp])]

    rows = [
        ("Correct detection", idx_success),
        ("Localization failure", idx_locfail),
        ("Bona-fide false positive", idx_fp),
    ]
    for name, i in rows:
        print(f"  [{name:>26}] idx={i} label={labels[i]} "
              f"p_attack={probs[i]:.3f} dice={dice[i]:.3f} "
              f"gt_area={int(gt_area[i])} stem={df_holdout.iloc[i]['stem']}")

    # Figura 2x3 (filas: GT / Prediccion ; columnas: los 3 casos)
    gt_cmap = ListedColormap([(0, 0, 0, 0), (0.10, 0.70, 0.20, 0.45)])   # verde GT
    pr_cmap = ListedColormap([(0, 0, 0, 0), (0.90, 0.10, 0.10, 0.45)])   # rojo pred

    # Generate at the paper's full text width (7.16 in) so that, included at
    # \textwidth, the figure is 1:1 and its 9 pt text matches the body text.
    fig, axes = plt.subplots(2, 3, figsize=(7.16, 3.20))
    axes[0, 0].set_ylabel("Ground truth", fontsize=9)
    axes[1, 0].set_ylabel("Prediction", fontsize=9)

    for c, (name, i) in enumerate(rows):
        # Imagen en su proporcion original; mascaras 224x224 reescaladas a (W,H)
        img = np.asarray(Image.open(df_holdout.iloc[i]["img_path"]).convert("RGB"))
        H, W = img.shape[:2]
        gt_full = np.array(Image.fromarray((gt_masks[i] * 255).astype(np.uint8))
                           .resize((W, H), Image.NEAREST)) > 127
        pr_full = np.array(Image.fromarray((pred_masks[i] * 255).astype(np.uint8))
                           .resize((W, H), Image.NEAREST)) > 127

        axes[0, c].set_title(name, fontsize=9)
        axes[0, c].imshow(img)
        axes[0, c].imshow(gt_full.astype(float), cmap=gt_cmap, vmin=0, vmax=1)
        axes[1, c].imshow(img)
        axes[1, c].imshow(pr_full.astype(float), cmap=pr_cmap, vmin=0, vmax=1)

    for ax in axes.ravel():
        ax.set_xticks([])
        ax.set_yticks([])

    fig.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0.04, hspace=0)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(f"{OUT_STEM}.pdf", dpi=600, bbox_inches="tight", pad_inches=0)
    fig.savefig(f"{OUT_STEM}.png", dpi=600, bbox_inches="tight", pad_inches=0)
    print(f"[OK] figura guardada: {OUT_STEM}.pdf / .png")


if __name__ == "__main__":
    main()
