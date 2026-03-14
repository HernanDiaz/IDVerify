# DocVerify — Paper Writing Plan
**Target:** Q1 journal (IEEE Transactions on Information Forensics and Security, T-IFS, or IEEE TNNLS)
**Date started:** 2026-03-14

---

## 1. Working Title

> *"DocVerify: Multi-Objective Hyperparameter Optimisation for Joint Document Forgery Detection and Localisation"*

Alternative:
> *"Pareto-Optimal Multitask Learning for Document Authenticity Verification"*

---

## 2. Core Contributions (what makes this publishable)

1. **Pareto-based multi-objective HPO for multitask learning.**
   Instead of collapsing detection and localisation into a single scalar loss via fixed λ,
   we frame the weight selection as a bi-objective optimisation problem (maximise PR-AUC
   AND Dice simultaneously). MOTPE (Multi-objective Tree-structured Parzen Estimator,
   Optuna) is used to navigate the trade-off space, and the final configuration is
   selected as the Pareto-optimal trial minimising Euclidean distance to the ideal point
   (1, 1). This is novel in the document forensics domain.

2. **Full nested cross-validation with simultaneous HPO.**
   10 outer folds × 5 inner folds × 50 Optuna trials = 500 trials. The nested design
   prevents information leakage from HPO into the test estimate, giving unbiased
   generalisation bounds. Most document forensics papers use a single train/val/test
   split and tune manually.

3. **Multitask CNN architecture (Patel Encoder + U-Net Decoder).**
   A compact model (8→16→32→64→128→256 filters) jointly trained on binary classification
   (pristine vs forged, sigmoid output) and pixel-level segmentation (mask head).
   Score convention: 1 − sigmoid(cls_logit) → 1.0 = pristine, 0.0 = tampered.

4. **Ablation study isolating each design choice.**
   Four variants (multitask, cls_only, seg_only, unweighted_losses) × 30 seeds with
   Wilcoxon + Holm statistical testing demonstrate that every component contributes.

5. **Competition evaluation on FantasyID (DeepID 2025 Challenge).**
   Holdout evaluation on unseen FantasyID images provides an external validity check
   beyond the academic dataset.

---

## 3. Suggested Journal Targets (in order of preference)

| Journal | Full name | Impact | Notes |
|---------|-----------|--------|-------|
| **IEEE T-IFS** | IEEE Transactions on Information Forensics and Security | Q1, IF ~6.8 | Best fit: document forensics + security |
| **IEEE TNNLS** | IEEE Transactions on Neural Networks and Learning Systems | Q1, IF ~10.4 | Stronger ML focus; emphasise the Pareto HPO contribution |
| **Pattern Recognition** | Pattern Recognition (Elsevier) | Q1, IF ~8.0 | Good fit if forensics angle is softened |
| **IJCV** | International Journal of Computer Vision | Q1 | Higher bar, more vision-focused |

**Recommendation:** Submit to **IEEE T-IFS** first. If rejected, revise for TNNLS
emphasising the multi-objective learning methodology.

---

## 4. Paper Structure (IEEE double-column format, ~8 pages)

### Abstract (~150 words)
- Problem: document forgery detection + localisation jointly
- Gap: existing methods use fixed scalar loss weights, ignore trade-off
- Method: Pareto-based MOTPE HPO inside nested CV
- Results: PR-AUC X.XX ± Y.YY, Dice X.XX ± Y.YY across 10 folds; ablation confirms each component
- Significance: outperforms scalar weighting, statistically significant (Wilcoxon p < 0.001)

### I. Introduction (~1 page)
- Motivation: document fraud, ID verification use case
- Challenge: detection alone is insufficient → localisation needed for forensic evidence
- Problem: multitask loss balancing is critical and dataset-dependent
- Proposed solution overview
- Contributions (numbered list, 4–5 bullets)
- Paper organisation

### II. Related Work (~1 page)
- Document forgery detection (traditional + deep learning)
- Multitask learning for forgery (if any)
- Multi-objective optimisation in neural network training
- Hyperparameter optimisation (SMAC, Optuna, MOTPE)
- Nested cross-validation in ML

### III. Methodology (~2 pages)
**III-A. Dataset**
- FantasyID dataset description (source, classes, splits)
- Preprocessing (resize to 224×224, normalise [0,1])

**III-B. Model Architecture**
- Patel CNN Encoder: 5 blocks, 8→16→32→64→128→256 filters, LeakyReLU(0.2) + BN
- U-Net Decoder: skip connections, upsampling blocks
- Classification head: global average pool → FC → sigmoid
- Segmentation head: 1×1 conv → sigmoid per pixel
- Dropout rate: tuned by HPO

**III-C. Training**
- Loss: L = λ · BCE_cls + (1−λ) · (1 − Dice_seg)
- Adam optimiser, learning rate tuned by HPO
- Decoder channels: tuned by HPO

**III-D. Multi-Objective HPO**
- Objectives: maximise PR-AUC (detection) AND Dice (localisation)
- Optimiser: MOTPE via Optuna
- 50 trials per inner fold
- Search space: dropout_rate, dec_ch, lr, λ
- Selection criterion: arg min_{p ∈ Pareto front} ||(1,1) − p||₂
- Reference Figure 1 (Pareto front plot)

**III-E. Nested Cross-Validation**
- 10 outer folds (StratifiedKFold)
- 5 inner folds for HPO
- 500 total trials
- Final model: best config from each outer fold, retrained on full outer train set
- Reference Figure 2 (nested CV bar chart)

### IV. Experiments (~1.5 pages)
**IV-A. Evaluation Metrics**
- PR-AUC (detection): area under precision-recall curve
- Dice (localisation): global pixel-level F1 on forgery mask
- Why PR-AUC not ROC-AUC: class imbalance

**IV-B. Ablation Study**
- Four variants: Multitask, cls_only, seg_only, unweighted_losses
- 30 seeds each
- Statistical testing: two-sided Wilcoxon signed-rank, Holm correction
- Reference Figure 3 (violin plots)

**IV-C. Comparison: Pareto vs Scalar HPO**
- Pareto HPO vs grid search over scalar λ values (by_prauc, by_dice)
- Shows Pareto dominates single-objective selection
- Reference Figure 5 (box plots)

**IV-D. Challenge Results**
- DeepID 2025 Challenge, Track 1 and Track 2
- Comparison with other participating teams
- Note: submitted after deadline (unofficial evaluation)
- Reference Figure 4 (horizontal bar chart)

### V. Discussion (~0.5 page)
- Why Pareto weighting helps: avoids degenerate solutions that maximise one metric at expense of the other
- Limitations: single architecture, single dataset
- Future work: larger architectures, multi-dataset evaluation, online HPO

### VI. Conclusion (~0.25 page)
- Summary of contributions
- Practical takeaway for practitioners

### References
- ~30–40 references (IEEE format)

---

## 5. Figures (all generated, in paper_figures/output/)

| File | Figure | Status |
|------|--------|--------|
| `fig1_pareto_front.pdf` | Pareto front — HPO objective space | ✓ Done |
| `fig2_nested_cv.pdf` | Nested CV — per-fold PR-AUC + Dice | ✓ Done (legend pending review) |
| `fig3_ablation.pdf` | Ablation study — violin plots | ✓ Done |
| `fig4_challenge.pdf` | Challenge comparison — horizontal bars | ✓ Done |
| `fig5_scalar_vs_pareto.pdf` | Pareto vs scalar HPO — box plots | ✓ Done |

Caption text files: fig1 ✓, fig3 ✓ | fig2, fig4, fig5 pending

---

## 6. Key Numbers to Report in the Paper

### Main results (Multitask, 10-fold nested CV):
- PR-AUC: **0.9967 ± 0.0021**
- Dice (global): **0.8750 ± 0.0180**
- Selected config distance to ideal: **d = 0.316**

### Ablation (30 seeds, blind test):
| Variant | PR-AUC (mean ± std) | Dice (mean ± std) |
|---------|--------------------|--------------------|
| Multitask | 0.9967 ± 0.0021 | 0.8750 ± 0.0180 |
| cls_only | 0.9503 ± 0.0744 | 0.0018 ± 0.0079 |
| seg_only | 0.7277 ± 0.0528 | 0.8672 ± 0.0232 |
| unweighted_losses | 0.9958 ± 0.0016 | 0.8572 ± 0.0165 |

### Statistical significance (vs Multitask, Wilcoxon + Holm):
| Comparison | PR-AUC p | Dice p |
|------------|----------|--------|
| vs cls_only | 1.49e-8 (***) | 1.30e-8 (***) |
| vs seg_only | 1.49e-8 (***) | 0.288 (ns) |
| vs unweighted | 0.083 (ns) | 1.28e-4 (***) |

---

## 7. Writing Order (recommended)

1. **Methods (Section III)** — write from code, most objective section
2. **Figures + captions** — already generated, write captions in LaTeX
3. **Results (Section IV)** — fill in numbers from table above
4. **Abstract** — write last, summarise everything
5. **Introduction** — write after abstract, frame contributions
6. **Related Work** — fill gaps identified during introduction
7. **Discussion + Conclusion** — write last

---

## 8. LaTeX Notes

- Template: `IEEEtran` class, `\documentclass[journal]{IEEEtran}`
- Figures: `\includegraphics[width=\columnwidth]{fig1_pareto_front.pdf}`
  or `width=\textwidth` for double-column figures
- Math: use `\text{PR-AUC}`, `\text{Dice}`, `\lambda`, `d = \|\mathbf{1} - \mathbf{p}\|_2`
- Significance: use `$p < 0.001$` inline, not stars in running text
- Line width for double column: 3.5 inches (single), 7.16 inches (double)

---

## 9. Files of Interest

| Path | Description |
|------|-------------|
| `exports_hpo_pareto_nested/` | All experiment CSVs (HPO trials, nested CV results, stats) |
| `exports_hpo_pareto_nested/models/model_multitask_seed47.pt` | Selected model checkpoint |
| `paper_figures/` | All figure generation code (SOLID architecture) |
| `paper_figures/output/` | Generated PDFs + caption TXT files |
| `baseline-docker/` | FastAPI competition submission container |
| `train.py` | Training loop |
| `evaluate.py` | Evaluation metrics |
| `model.py` | DocVerify architecture |
| `config.py` | Experiment configuration |
