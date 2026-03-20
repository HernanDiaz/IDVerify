"""
03_eval_finetuned_comparison.py — Comparativa de tres sistemas sobre el holdout:
  1. TruFor zero-shot    (trufor_scores.csv)
  2. TruFor fine-tuneado (trufor_finetuned_scores.csv)
  3. DocVerify (ours)    (exports_hpo_pareto_nested/final_blind_test_multiseed.csv)

Genera:
  sota_comparison/full_comparison_table.csv
  sota_comparison/full_comparison_table.tex
"""

from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score, f1_score

_THIS_DIR   = Path(__file__).resolve().parent
_ROOT       = _THIS_DIR.parent

GT_CSV                = _THIS_DIR / "holdout_gt.csv"
ZEROSHOT_CSV          = _THIS_DIR / "trufor_scores.csv"
FINETUNED_CSV         = _THIS_DIR / "trufor_finetuned_scores.csv"
MASKS_DIR             = _THIS_DIR / "holdout_masks"
DOCVERIFY_CSV         = _ROOT / "exports_hpo_pareto_nested" / "final_blind_test_multiseed.csv"
OUT_CSV               = _THIS_DIR / "full_comparison_table.csv"
OUT_TEX               = _THIS_DIR / "full_comparison_table.tex"


# ── Métricas ──────────────────────────────────────────────────────────────

def cls_metrics(y_true, y_score):
    y_true  = np.asarray(y_true,  dtype=int)
    y_score = np.asarray(y_score, dtype=float)
    pr_auc  = float(average_precision_score(y_true, y_score))
    roc_auc = float(roc_auc_score(y_true, y_score))
    thresholds = np.linspace(0.001, 1.0, 501)
    best_f1, best_thr = 0.0, 0.5
    for thr in thresholds:
        f1 = float(f1_score(y_true, (y_score >= thr).astype(int),
                            average="macro", zero_division=0))
        if f1 > best_f1:
            best_f1, best_thr = f1, float(thr)
    f1_det = float(f1_score(y_true, (y_score >= 0.5).astype(int),
                            average="binary", zero_division=0))
    return {"pr_auc": pr_auc, "roc_auc": roc_auc,
            "f1_macro": best_f1, "f1_det_05": f1_det, "thr_opt": best_thr}


def dice_metrics(df_gt, df_scores, merge_key="unique_stem"):
    if merge_key not in df_gt.columns:
        merge_key = "stem"
    df = df_gt.merge(
        df_scores[["stem", "map_npy_path"]].rename(columns={"stem": merge_key}),
        on=merge_key, how="left"
    )
    TP = FP = FN = 0
    per_img = []
    for _, row in df.iterrows():
        gt_p = MASKS_DIR / f"{row['stem']}.npy"
        if not gt_p.exists():
            continue
        gt = (np.load(gt_p).astype(np.uint8) > 0).astype(np.uint8)

        mp = str(row.get("map_npy_path", "N/A"))
        if mp == "N/A" or not Path(mp).exists():
            pred = np.zeros_like(gt)
        else:
            m = np.load(mp).astype(np.float32)
            if m.max() > 1.0:
                m /= 255.0
            if m.shape != gt.shape:
                from PIL import Image
                m = np.array(
                    Image.fromarray((m * 255).astype(np.uint8)).resize(
                        (gt.shape[1], gt.shape[0]), Image.BILINEAR)
                ).astype(np.float32) / 255.0
            pred = (m >= 0.5).astype(np.uint8)

        tp = int((gt & pred).sum())
        fp = int(((1 - gt) & pred).sum())
        fn = int((gt & (1 - pred)).sum())
        TP += tp; FP += fp; FN += fn
        if gt.sum() > 0:
            per_img.append((2 * tp + 1e-6) / (gt.sum() + pred.sum() + 1e-6))

    dice_g = float((2 * TP + 1e-6) / (2 * TP + FP + FN + 1e-6))
    dice_m = float(np.mean(per_img)) if per_img else float("nan")
    return {"dice_global": dice_g, "dice_pos_mean": dice_m}


def eval_trufor(df_gt, scores_csv, label):
    df_s = pd.read_csv(scores_csv)
    mk = "unique_stem" if "unique_stem" in df_gt.columns else "stem"
    df = df_gt.merge(
        df_s[["stem", "trufor_score"]].rename(columns={"stem": mk}),
        on=mk, how="left"
    ).dropna(subset=["trufor_score"])

    y_true  = df["label"].values.astype(int)
    y_score = df["trufor_score"].values.astype(float)

    # Corregir inversión si es necesario
    if y_score[y_true == 1].mean() < y_score[y_true == 0].mean():
        print(f"  [{label}] Scores invertidos — aplicando 1-score")
        y_score = 1.0 - y_score

    print(f"  [{label}] Score range [{y_score.min():.3f}, {y_score.max():.3f}] "
          f"| attack_mean={y_score[y_true==1].mean():.3f} "
          f"| bonafide_mean={y_score[y_true==0].mean():.3f}")

    m = cls_metrics(y_true, y_score)
    d = dice_metrics(df_gt, df_s)
    return {**m, **d}


def load_docverify(variant="multitask"):
    df = pd.read_csv(DOCVERIFY_CSV)
    dv = df[df["variant"] == variant]
    out = {}
    for col in ["test_pr_auc", "test_roc_auc", "test_dice_global",
                "test_dice_pos_mean", "test_f1_macro", "test_f1_1"]:
        if col in dv.columns:
            k = col.replace("test_", "")
            out[f"{k}_mean"] = float(dv[col].mean())
            out[f"{k}_std"]  = float(dv[col].std())
    # F1 det @ 0.5 = test_f1_1 (binary)
    out["n_seeds"] = len(dv)
    return out


# ── Formateo ──────────────────────────────────────────────────────────────

def _fmt(v, decimals=3):
    return "N/A" if np.isnan(v) else f"{v:.{decimals}f}"

def _fmt_pm(mean, std, decimals=3):
    if np.isnan(mean): return "N/A"
    return f"{mean:.{decimals}f} ± {std:.{decimals}f}"

def _bold(v, ref, better="higher"):
    if np.isnan(v) or np.isnan(ref): return _fmt(v)
    is_better = v > ref if better == "higher" else v < ref
    s = f"{v:.3f}"
    return f"\\textbf{{{s}}}" if is_better else s


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print(" Comparativa: TruFor zero-shot / fine-tuned / DocVerify")
    print("=" * 60)

    for p, desc in [(GT_CSV, "holdout_gt.csv"),
                    (ZEROSHOT_CSV, "trufor_scores.csv"),
                    (FINETUNED_CSV, "trufor_finetuned_scores.csv"),
                    (DOCVERIFY_CSV, "final_blind_test_multiseed.csv")]:
        if not p.exists():
            print(f"[ERROR] Falta: {p}  ({desc})")
            sys.exit(1)

    df_gt = pd.read_csv(GT_CSV)

    print("\n[1/3] TruFor zero-shot...")
    zs = eval_trufor(df_gt, ZEROSHOT_CSV, "zero-shot")
    for k, v in zs.items(): print(f"       {k:22s}: {v:.4f}")

    print("\n[2/3] TruFor fine-tuneado (FantasyID, 30 épocas)...")
    ft = eval_trufor(df_gt, FINETUNED_CSV, "fine-tuned")
    for k, v in ft.items(): print(f"       {k:22s}: {v:.4f}")

    print("\n[3/3] DocVerify (multitask, 30 seeds)...")
    dv = load_docverify()
    print(f"       pr_auc          : {dv['pr_auc_mean']:.4f} ± {dv['pr_auc_std']:.4f}")
    print(f"       roc_auc         : {dv['roc_auc_mean']:.4f} ± {dv['roc_auc_std']:.4f}")
    print(f"       dice_global     : {dv['dice_global_mean']:.4f} ± {dv['dice_global_std']:.4f}")

    # ── CSV ──
    rows = [
        {"System": "TruFor (zero-shot)",      "Training": "876K pre-training",
         "PR-AUC": _fmt(zs["pr_auc"]), "ROC-AUC": _fmt(zs["roc_auc"]),
         "Dice (global)": _fmt(zs["dice_global"]), "Dice (pos)": _fmt(zs["dice_pos_mean"]),
         "F1-det@0.5": _fmt(zs["f1_det_05"])},
        {"System": "TruFor (fine-tuned)",     "Training": "FantasyID",
         "PR-AUC": _fmt(ft["pr_auc"]), "ROC-AUC": _fmt(ft["roc_auc"]),
         "Dice (global)": _fmt(ft["dice_global"]), "Dice (pos)": _fmt(ft["dice_pos_mean"]),
         "F1-det@0.5": _fmt(ft["f1_det_05"])},
        {"System": "DocVerify (ours)",        "Training": "FantasyID",
         "PR-AUC": _fmt_pm(dv["pr_auc_mean"], dv["pr_auc_std"]),
         "ROC-AUC": _fmt_pm(dv["roc_auc_mean"], dv["roc_auc_std"]),
         "Dice (global)": _fmt_pm(dv["dice_global_mean"], dv["dice_global_std"]),
         "Dice (pos)": _fmt_pm(dv.get("dice_pos_mean_mean", float("nan")),
                               dv.get("dice_pos_mean_std",  float("nan"))),
         "F1-det@0.5": _fmt_pm(dv.get("f1_1_mean", float("nan")),
                               dv.get("f1_1_std",  float("nan")))},
    ]
    df_out = pd.DataFrame(rows)
    df_out.to_csv(OUT_CSV, index=False)
    print(f"\n[OK] CSV -> {OUT_CSV}")
    print("\n" + df_out.to_string(index=False))

    # ── LaTeX ──
    dv_pr  = dv["pr_auc_mean"];  dv_roc = dv["roc_auc_mean"]
    dv_dg  = dv["dice_global_mean"]

    tex = (
        r"\begin{table}[!t]" + "\n"
        r"\centering" + "\n"
        r"\caption{Comparison of TruFor~\citep{trufor2023} (zero-shot and fine-tuned on" + "\n"
        r"FantasyID) against DocVerify on the internal holdout ($N=498$)." + "\n"
        r"DocVerify results are mean\,$\pm$\,std over 30 seeds.}" + "\n"
        r"\label{tab:trufor_comparison}" + "\n"
        r"\begin{tabular}{lccc}" + "\n"
        r"\toprule" + "\n"
        r"System & PR-AUC & ROC-AUC & Dice\textsubscript{global} \\" + "\n"
        r"\midrule" + "\n"
        f"TruFor zero-shot~\\citep{{trufor2023}} & "
        f"{_bold(zs['pr_auc'], dv_pr)} & "
        f"{_bold(zs['roc_auc'], dv_roc)} & "
        f"{_bold(zs['dice_global'], dv_dg)} \\\\\n"
        f"TruFor fine-tuned & "
        f"{_bold(ft['pr_auc'], dv_pr)} & "
        f"{_bold(ft['roc_auc'], dv_roc)} & "
        f"{_bold(ft['dice_global'], dv_dg)} \\\\\n"
        f"\\textbf{{DocVerify}} (ours) & "
        f"{_bold(dv_pr, max(zs['pr_auc'], ft['pr_auc']))} $\\pm$ {dv['pr_auc_std']:.3f} & "
        f"{_bold(dv_roc, max(zs['roc_auc'], ft['roc_auc']))} $\\pm$ {dv['roc_auc_std']:.3f} & "
        f"{_bold(dv_dg, max(zs['dice_global'], ft['dice_global']))} $\\pm$ {dv['dice_global_std']:.3f} \\\\\n"
        r"\bottomrule" + "\n"
        r"\end{tabular}" + "\n"
        r"\end{table}" + "\n"
    )
    OUT_TEX.write_text(tex, encoding="utf-8")
    print(f"[OK] LaTeX -> {OUT_TEX}")

    print("\n" + "=" * 60)
    print(" RESUMEN")
    print("=" * 60)
    print(f"  TruFor zero-shot  : PR-AUC={zs['pr_auc']:.4f}  Dice={zs['dice_global']:.4f}")
    print(f"  TruFor fine-tuned : PR-AUC={ft['pr_auc']:.4f}  Dice={ft['dice_global']:.4f}")
    print(f"  DocVerify (ours)  : PR-AUC={dv_pr:.4f}±{dv['pr_auc_std']:.4f}"
          f"  Dice={dv_dg:.4f}±{dv['dice_global_std']:.4f}")


if __name__ == "__main__":
    main()
