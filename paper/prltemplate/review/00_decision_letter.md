# Decision Letter — PRLETTERS-D-26-00496

**Manuscript:** Multi-Objective Pareto hyperparameter optimization for joint detection and localization of forged identity documents
**Author:** Hernán Díaz Rodríguez
**Journal:** Pattern Recognition Letters
**Decision:** Major Revision
**Editor-in-Chief:** Jiwen Lu, Ph.D.
**Revision deadline:** 2026-07-17
**Submission portal:** https://www.editorialmanager.com/prletters/ (username: HRodríguez-288)

> NOTE: Revised submission must include **source files** (LaTeX), not only PDF.
> A **list of changes / point-by-point rebuttal** must accompany the revision.
> Revised submission may be re-reviewed.

---

## Associate Editor (AE)

The reviewers find the work relevant, but raise several important concerns regarding
**result transparency, baseline comparisons, and the strength of the main claims**.
Revise to address these, in particular:
- clarifying Table 5,
- ensuring complete and consistent reporting,
- moderating or strengthening the conclusions.

---

## Reviewer 1 — Major Revision

Read the manuscript, examined the public code repo (github.com/HernanDiaz/IDVerify),
verified references against original PDFs, inspected the full CSV results in the repo.

### Major Issues

**R1.1 — EdgeDoc performance in Table 5 (attribution error).**
Table 5 reports AG/EdgeDoc with F1@0.5 = 0.958 on FantasyID. Per the original EdgeDoc
paper (George & Marcel, 2025, Table 2), EdgeDoc *alone* achieves F1 = 0.43 on FantasyID.
The value 0.96 corresponds to the **EdgeDoc+TruFor fusion**, not EdgeDoc in isolation.
If correct value is 0.43, DocVerify outperforms EdgeDoc by 2.25×, not the marginal
0.969 vs 0.958 currently suggested.
→ Clarify the exact origin of 0.958. If from the fusion, either report EdgeDoc standalone
(0.43) or rename the row to "EdgeDoc+TruFor (fusion)" with an explicit note.

**R1.2 — Selective omission of SIDTD results.**
Repo includes full SIDTD support (dataset_sidtd.py, eval_sidtd.py, finetune_sidtd.py).
Internal docs (PROJECT_CONTEXT.md) state: "SIDTD — Tested and implemented but excluded
from the paper due to poor results". SIDTD has real-world document images — exactly the
validation that would show generalization beyond synthetic data. Excluding unfavorable
results = file-drawer problem.
→ Strongly recommend including SIDTD results, even if lower. A candid discussion of the
domain gap would strengthen the paper.

**R1.3 — TruFor fine-tuned results also omitted.**
Repo contains run_trufor_finetune.py, trufor_finetuned_scores.csv, full_comparison_table.csv.
These show TruFor fine-tuned achieves F1-det@0.5 = 0.856 — competitive on detection (0.969)
but much lower localization (Dice = 0.132 vs 0.875). Table 5 only reports TruFor zero-shot
(F1 = 0.807, Dice = 0.056).
→ Either add the TruFor fine-tuned row to Table 5 or explain in text why it was excluded.

**R1.4 — No data augmentation on a small synthetic dataset.**
Trained on ~2,800 synthetic images without augmentation; justification is technical
("VRAM caching prevents augmentation"), not scientific. Likely explains poor SIDTD results.
→ Add standard augmentation (moderate rotation, brightness variation, Gaussian blur) and
report results with and without it. If augmentation degrades performance, that is itself
informative.

**R1.5 — Weak evidence that MOTPE/Pareto is superior to scalar optimization.**
Section 4.4: "No statistically significant differences... (Wilcoxon + Holm, all p ≥ 0.05)".
Only 5 repetitions per condition → very low statistical power (std ±0.030 Dice, ±0.006 PR-AUC).
The "Pareto needs no a priori commitment" argument is theoretical, not an empirical advantage.
→ Increase repetitions to at least 15 (preferably 30) per condition, report CIs and effect
sizes (Cohen's d). If still non-significant, reframe honestly. "We applied MOTPE and found it
equivalent to scalar search" is more valuable than implying unsupported superiority.

**R1.6 — Comparison with challenge participants not on same test set.**
Docs clarify "Evaluation is on an internal holdout, not the official test set". DocVerify's
metrics are not directly comparable with challenge entries. Footnote acknowledges this, but
Table 5 formatting (bolded DocVerify, side-by-side metrics) invites direct comparison.
→ Either move the challenge comparison to a clearly-labeled appendix, or add a boldface
caveat directly in the table caption stating evaluation sets differ.

### Moderate Issues

**R1.7 — Architecture not state-of-the-art.**
Patel CNN (2019, ~5M params), no pre-trained backbone. Contribution is the optimization
method, not architecture. Showing MOTPE works with a modern backbone (e.g. ResNet-18 +
ImageNet) would strengthen the architecture-agnostic claim.
→ Consider adding at least one pre-trained backbone as a baseline.

**R1.8 — Qualitative analysis shows only a single success case.**
Figure 4 shows one successful localization → cherry-picking given F1_loc std ±0.096.
→ Include at least three examples: one success, one localization failure, one false positive
on a bonafide document.

### Minor Issues

**R1.9 — Citation precision.**
Of 27 verified refs, 21 fully accurate, 6 minor imprecisions (e.g. "3,284 images" for
FantasyID not in original paper; "876,000 images" for TruFor is approximate — original
reports ~828K). No fabricated refs.
→ Verify numerical claims against original sources; add page numbers / direct quotes.

**R1.10 — Computational cost vs benefit.**
~59 GPU-hours (2,500 trials) to tune a 5M-param CNN. A random search of ~20 configs would
likely reach similar conclusions far faster.
→ Add a brief discussion of whether the computational investment was proportional to the
gain, especially given Section 4.4 shows no significant Pareto-vs-scalar difference.

### Positive Aspects (noted by R1)
- Nested CV (10×5) unusually rigorous for PRLetters.
- Statistical testing (Wilcoxon + Holm-Bonferroni + Cohen's d, 30 seeds) above journal norm.
- Open-source, well-organized code with all trial CSVs — exemplary.
- Clear, well-structured writing.
- Application of multi-objective HPO to document forensics is novel.

---

## Reviewer 2 — Major Revision

### 1. General Comments

**R2.1.1** — All experiments on FantasyID (synthetic, single institute). Severely limits
generalizability; no cross-domain validation; practical applicability undemonstrated on
real-world documents.

**R2.1.2** — Table 5 needs more caution. DocVerify on internal 15% holdout vs challenge
systems on official test set (FantasyID test + private out-of-domain PXL Vision set).
Competitive conclusions (e.g. "competitive with Sunlight despite using no external data")
not statistically justified.

**R2.1.3** — NCV and blind holdout results reported separately, but unclear whether the 15%
holdout was defined before or after the NCV protocol was designed. If used during development
(e.g. for early-stopping patience tuning), blind test results may be optimistically biased.

**R2.1.4** — Dataset imbalance (71.6% attack, 28.4% bonafide) "preserved in all splits and not
corrected". No justification provided, no analysis of impact.

**R2.1.5** — "No data augmentation" applied — unusual for ~2,791 training images. No ablation
or justification.

### 2. Major Concerns

**R2.2.1** — Most serious: insufficient cross-domain generalization evidence. Must either
provide results on at least one additional benchmark (e.g. MIDV-2020, DocTamper, or any
real-world document dataset) OR substantially revise scope of claims to explicitly restrict
to FantasyID domain, with appropriate hedging throughout.

**R2.2.2** — Table 5 comparison methodologically unsound (non-equivalent evaluation sets;
challenge used private out-of-domain PXL Vision partition, DocVerify used in-distribution
internal holdout). Add a clearly formatted disclaimer box within the table itself (not only
text) and remove competitive language from conclusions.

**R2.2.3** — Class imbalance strategy requires justification and ablation. BCEWithLogits
without pos_weight → model may bias toward majority class. Provide theoretical/empirical
justification, and include ablation/sensitivity analysis on class imbalance correction and
its impact on PR-AUC and Dice.

**R2.2.4** — "Pareto vs Scalar" (Table 4) does not support claimed advantage (all p ≥ 0.05).
If primary novelty is Pareto HPO and scalar achieves equivalent results, provide stronger
argument for structural/practical advantages beyond predictive accuracy (reduced sensitivity
to scalar-weight choice, better trade-off coverage, improved robustness across seeds).

**R2.2.5** — Architecture description incomplete for reproducibility. Missing/ambiguous:
- Exact number/config of FC layers in classification head (256→32→16→16→1 mentioned but
  dropout rates and activations between layers not fully specified).
- Decoder block structure: # conv layers per decoder block, kernel sizes, whether BN applied
  in decoder.
- Skip connection concatenation mechanism (stated extracted "prior to pooling" but concat
  mechanism in decoder not described).
- Early stopping criterion: patience=12 mentioned, but monitored metric not specified
  (val loss? PR-AUC? Dice?).

### 3. Minor Concerns

**R2.3.1** — Equations must be explicitly cited in the text.

**R2.3.2** — "F1@0.5" used in Table 5 and throughout without formal definition. Define
explicitly (F1 at classification threshold 0.5) at first occurrence.

### 4. Recommendation: Major Revision
Critical issues to address: single-dataset evaluation (most significant); Table 5 unsound
comparison; Pareto HPO contribution undermined by Table 4; architecture + blind holdout
protocol gaps preventing reproducibility. Highlights adequate. No graphical abstract provided.
