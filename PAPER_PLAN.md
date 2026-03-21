# DocVerify — Paper Writing Plan
**Date started:** 2026-03-14
**Last updated:** 2026-03-21

---

## 1. Working Title

> *"Multi-Objective Pareto Hyperparameter Optimization for Joint Detection and Localization of Forged Identity Documents"*

---

## 2. Core Contributions

1. **Pareto-based multi-objective HPO for multitask learning.**
   MOTPE (Optuna) navigates the PR-AUC / Dice trade-off space; final config selected
   by minimum Euclidean distance to the ideal point (1,1). Novel in document forensics.

2. **Full nested cross-validation with simultaneous HPO.**
   10 outer × 5 inner × 50 trials = 500 unique configurations (2,500 individual
   model trainings). Prevents HPO leakage into test estimates.

3. **Multitask CNN (Patel Encoder + U-Net Decoder).**
   Compact architecture jointly trained on binary classification and pixel-level
   segmentation. Trains exclusively on FantasyID — no external data.

4. **Ablation study with statistical testing.**
   Four variants × 30 seeds. Wilcoxon + Holm confirms each component contributes
   (multitask vs unweighted: p=1.3×10⁻⁴, Cohen's d=1.01, large effect).

5. **Competitive evaluation vs DeepID 2025 Challenge participants.**
   Outperforms all FantasyID-only systems on detection; surpasses challenge winner
   on localization F1 (0.807 vs 0.784).

---

## 3. Journal Targets

| Journal | Format | Limit | Status |
|---|---|---|---|
| **Pattern Recognition Letters** | elsarticle 5p double-col | 7p incl. refs | ✅ **Submitted — 2026-03-21** |
| **IEEE T-IFS** | IEEEtran double-col | 14p body + refs | Draft ready — 12 pages |
| IEEE TNNLS | IEEEtran | 14p | Fallback if T-IFS rejected |
| Pattern Recognition | elsarticle | Open | Fallback after PRL |

---

## 4. Paper Versions

### PRL (`paper/prltemplate/`) — 7 pages — SUBMITTED
Condensed version:
- Sec 4.4 removed (Pareto vs Scalar)
- Related Work condensed to ~0.7 pages
- Qualitative: attack-only example (bonafide removed)
- NCV table: ROC-AUC column removed
- Author-year citations (`\citep{}`/`\citet{}`)

### T-IFS (`paper/tifs/`) — 12 pages
Full version with all sections:
- Sec 4.4: Pareto vs Scalar HPO comparison (fig5)
- Full qualitative analysis (bonafide + attack examples)
- Full Related Work (~1.5 pages)
- NCV table with ROC-AUC column

---

## 5. Key Results

### Nested CV (10 folds):
| Metric | Value |
|---|---|
| PR-AUC | **0.9921 ± 0.0058** |
| Dice | **0.856 ± 0.030** |
| Selected w_mask | 2.39 ± 0.43 |

### Blind Test (30 seeds, multitask):
| Metric | Value |
|---|---|
| PR-AUC | 0.9967 ± 0.0021 |
| Dice | 0.875 ± 0.018 |
| F1 det (thr=0.5) | 0.969 ± 0.014 |
| F1 loc (per-image) | 0.807 ± 0.096 |

### Ablation (30 seeds):
| Variant | PR-AUC | Dice |
|---|---|---|
| **Multitask (ours)** | **0.9967 ± 0.0021** | **0.875 ± 0.018** |
| cls_only | 0.9503 ± 0.0744 | 0.002 ± 0.008 |
| seg_only | 0.7277 ± 0.0528 | 0.867 ± 0.023 |
| unweighted_losses | 0.9958 ± 0.0016 | 0.857 ± 0.016 |

### Challenge comparison (vs DeepID 2025):
| System | F1 det | F1 loc | Training data |
|---|---|---|---|
| Sunlight (1st) | 0.991 | 0.784 | 60K+ external |
| AG/EdgeDoc (3rd) | 0.958 | 0.686 | FantasyID only |
| **DocVerify (ours)** | **0.969 ± 0.014** | **0.807 ± 0.096** | FantasyID only |

*Evaluated on internal holdout — challenge deadline had already passed.*

---

## 6. Figures (all generated)

| File | Description | Used in |
|---|---|---|
| `fig0_architecture_compact.pdf` | DocVerify architecture | T-IFS + PRL |
| `fig1_pareto_front.pdf` | HPO objective space (500 configs) | T-IFS + PRL |
| `fig2_nested_cv.pdf` | Per-fold bar chart | T-IFS + PRL |
| `fig3_ablation.pdf` | Violin plots ablation | T-IFS + PRL |
| `fig4_challenge.pdf` | Challenge comparison | T-IFS + PRL |
| `fig5_scalar_vs_pareto.pdf` | Pareto vs scalar box plots | T-IFS only |

---

## 7. TODOs

### PRL — SUBMITTED ✅
- [x] Ecuaciones en Word corregidas con OOXML Math
- [x] Data Availability Statement: FantasyID → https://zenodo.org/records/17063366
- [x] Submission completada en Elsevier Editorial Manager (2026-03-21)

### T-IFS
- [ ] Add TruFor fine-tuned row to comparison table once inference completes
- [ ] Upload model weights to Zenodo → add DOI
- [ ] Run IEEE PDF checker / Xplore compliance tool
- [ ] Fill "Manuscript received" date

### Both
- [ ] DeepID Challenge: contact organizers for official test set evaluation
- [ ] Consider SIDTD as secondary dataset (currently excluded — poor results)
