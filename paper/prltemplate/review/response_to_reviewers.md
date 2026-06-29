# Response to Reviewers — PRLETTERS-D-26-00496

**Manuscript:** Multi-Objective Pareto hyperparameter optimization for joint detection and
localization of forged identity documents
**Author:** Hernán Díaz Rodríguez
**Decision:** Major Revision · **Deadline:** 2026-07-17

---

We thank the Associate Editor and both reviewers for their careful and constructive reading.
We are especially grateful for Reviewer 1's verification of our code repository and references,
and for both reviewers' recognition of the statistical rigor and open-source nature of the work.
Below we respond to each point individually. Reviewer comments are quoted in *italics*; our
responses follow. Manuscript changes are highlighted in **blue** in the revised PDF, with
section/line references given here.

---

## Cross-cutting themes (both reviewers)

Several concerns are raised by both reviewers and will be addressed jointly:

| Theme | Reviewer 1 | Reviewer 2 | Planned action |
|-------|-----------|-----------|----------------|
| Cross-domain / 2nd dataset (SIDTD) | R1.2 | R2.1.1, R2.2.1 | **DONE** — new Generalization subsection (§4.6): SIDTD zero-shot reported candidly (PR-AUC 0.555, near chance) + in-domain retraining recovery (PR-AUC 0.878, ROC-AUC 0.826, domain-shift not arch limit) + explicit FantasyID-domain scope restriction; Conclusion future-work reframed |
| Challenge comparison not on same test set | R1.6 | R2.1.2, R2.2.2 | **DONE** — removed the challenge comparison table (reviewer's first option); §4.5 condensed to one paragraph, all caveats inline (differing sets, no superiority claim, training-data disparity) |
| Pareto vs scalar — weak evidence | R1.5 | R2.1 (implied), R2.2.4 | **DONE** — 30-seed blind-test reframe (CIs + Cohen's d, Holm); Pareto significantly > both scalars; merged into ablation Table 3 |
| Data augmentation | R1.4 | R2.1.5 | DONE |
| Class imbalance | — | R2.1.4, R2.2.3 | **DONE** — justification (positive=majority, mild ratio) + pos_weight sensitivity sweep (30 seeds, all p_Holm>0.05); note in Sec. 3.1, full table in this letter |
| TruFor fine-tuned omitted | R1.3 | — | **DONE** — reviewer correct; rather than add a fine-tuned row, removed the whole table (no controlled comparison possible) and condensed §4.5, giving the section less weight |
| EdgeDoc attribution error | R1.1 | — | **DONE** — corrected attribution now in §4.5 prose (EdgeDoc standalone 0.43 vs EdgeDoc+TruFor fusion 0.958, cited to challenge leaderboard); challenge table since removed (see R1.6) |
| Reproducibility / architecture detail | — | R2.2.5 | DONE |
| Blind-holdout protocol timing | — | R2.1.3 | **DONE** — Sec. 3.1 sentence + 3-leg evidence (code isolation, git chronology, threshold from dev-internal val) |
| Pre-trained backbone baseline | R1.7 | — | **DONE** — re-ran identical MOTPE/Pareto protocol on ImageNet ResNet-18 (n=30): PR-AUC 0.994, Dice 0.889; neither model dominates; script `revision_experiments/resnet18_motpe.py`, Discussion sentence added |
| Qualitative: failure cases | R1.8 | — | **DONE** — replaced single-example figure with 2×3 figure (success / localization failure / bona-fide false positive) from blind-test seed 42; script `revision_experiments/qualitative_examples.py`, §4.7 rewritten |
| Citation precision | R1.9 | — | DONE |
| Compute cost vs benefit | R1.10 | — | DONE |
| Equation citing / F1@0.5 definition | — | R2.3.1, R2.3.2 | DONE |

---

# Reviewer 1

## Major Issues

### R1.1 — EdgeDoc attribution error in Table 5
> *Table 5 reports AG/EdgeDoc with F1@0.5 = 0.958... EdgeDoc alone achieves F1 = 0.43 on
> FantasyID. The value 0.96 corresponds to the EdgeDoc+TruFor fusion... Please clarify the
> exact origin of the 0.958 value.*

**Response:** We thank the reviewer for catching this. The reviewer is correct. We verified
the values against the DeepID 2025 leaderboard: EdgeDoc **standalone** achieves F1 = 0.43
(George & Marcel, 2025, Table 2), TruFor achieves F1 = 0.81, and the **EdgeDoc+TruFor fusion**
achieves F1 = 0.96 (Korshunov et al., 2025, Table 2). Our original "AG/EdgeDoc = 0.958" row
conflated the third-place team's *fusion* submission with the EdgeDoc architecture in
isolation. We corrected the attribution in the text: the EdgeDoc+TruFor fusion (0.958, cited
only to the challenge leaderboard, its actual source) is distinguished from EdgeDoc standalone
(0.43; George & Marcel, 2025, Table 2), noting that the fusion relies on the pre-trained TruFor
backbone whereas EdgeDoc standalone is the FantasyID-only comparator. The challenge comparison
table has since been removed (see R1.6), so this corrected attribution now appears in the
running prose of Sec. 4.5. Consistent with our response to R1.6/R2.1.2/R2.2.2, we do not draw
competitive conclusions from these numbers, since DocVerify is evaluated on a different
(internal) test set.

**Changes:** Sec. 4.5: corrected the EdgeDoc attribution in the prose (fusion 0.958 vs
standalone 0.43; George & Marcel, 2025, Table 2), with the 0.958 citation pointing only to the
challenge leaderboard (Korshunov et al., 2025). The challenge comparison table was subsequently
removed (see R1.6), so the correction now lives only in text; competitive language removed.

---

### R1.2 — Selective omission of SIDTD results
> *Running the experiments... and then excluding them because they were unfavorable is a
> classic file-drawer problem. I strongly recommend including SIDTD results... A candid
> discussion of the domain gap would strengthen the paper.*

**Response:** We agree and have brought the SIDTD result into the paper. The Experiments
section now contains a dedicated **Generalization** subsection (Sec. 4.6) whose
*Across datasets* paragraph reports the zero-shot transfer to the out-of-domain SIDTD
dataset (templates subset; 1,000 bonafide, 1,222 attack) candidly, including the
unfavorable numbers: the 30 blind-test models fall to near chance (PR-AUC 0.555 ± 0.011,
ROC-AUC 0.504 ± 0.009). To distinguish a domain-shift problem from an architectural one, we
also retrained the *same* architecture directly on the SIDTD training partition (25 seeds):
performance recovers to PR-AUC 0.878 ± 0.016, ROC-AUC 0.826 ± 0.023 (few-shot at 10–400 shots
stays near chance; only full in-domain training recovers). We therefore state in Sec. 4.6 that
the gap reflects domain shift rather than architectural capacity, and that the *zero-shot*
scope of the current FantasyID-trained system is bounded to its training domain, motivating
multi-source training as future work. The per-seed SIDTD CSVs (zero-shot, few-shot and full
retraining) are archived externally (Zenodo); the scripts are in the repository
(`revision_experiments/sidtd_generalization.py`, `finetune_sidtd.py`). This is consistent with
the cross-domain augmentation check reported under R1.4, which independently confirms the
zero-shot gap is too wide for standard augmentation to close.

**Changes:** Sec. 4.6 (new **Generalization** subsection): added an *Across datasets*
paragraph reporting the SIDTD zero-shot null result (PR-AUC 0.555, ROC-AUC 0.504), the
in-domain retraining recovery (PR-AUC 0.878, ROC-AUC 0.826), and a candid domain-gap
discussion; the former "Single dataset" limitation paragraph is subsumed by this concrete
result.

---

### R1.3 — TruFor fine-tuned results omitted
> *Please either add the TruFor fine-tuned row to Table 5 or explain in the text why it was
> excluded.*

**Response:** The reviewer is correct: the original comparison did not include a fine-tuned
TruFor result. Rather than add a single fine-tuned row, we removed the challenge comparison
table entirely (Sec. 4.5). Because DocVerify is evaluated on our internal 15% holdout while
every challenge figure comes from the official test set (FantasyID test partition plus a
private out-of-domain PXL Vision set), no direct or controlled comparison is possible; a
detailed table, with or without a fine-tuned TruFor row, would overstate its rigour. We
therefore give this section less weight: Sec. 4.5 is now a short paragraph that situates
DocVerify among related systems with explicit caveats and no superiority claim.

**Changes:** Sec. 4.5: removed the challenge comparison table and condensed the section to a
single paragraph (see R1.6).

---

### R1.4 — No data augmentation
> *Please add standard augmentation (moderate rotation, brightness variation, Gaussian blur)
> and report results with and without it. If augmentation degrades performance, that is itself
> informative.*

**Response:** We added the requested ablation. Using the paper's Pareto configuration and the
identical blind-test protocol (30 seeds), we trained a second set of models with five standard
augmentations applied **only** to the training loader (rotation ±10°, brightness ×[0.8,1.2],
contrast ×[0.8,1.2], Gaussian blur k=3 σ∈[0.1,2.0], JPEG compression q∈[50,95]); the
validation and holdout sets remain un-augmented. The "no aug" condition reuses the paper's
blind test, so the comparison is paired by seed and analysed with Wilcoxon + Holm + Cohen's d.

| Metric | No aug | With aug | Δ (aug−noaug) | Cohen's d | p (Holm) |
|---|---|---|---|---|---|
| PR-AUC | 0.9967 ± 0.0021 | 0.9577 ± 0.0457 | −0.039 | −0.84 | 1.5e−8 |
| Dice | 0.8750 ± 0.0180 | 0.7797 ± 0.0396 | −0.095 | −2.10 | 9.3e−9 |
| BAcc | 0.9573 ± 0.0184 | 0.8302 ± 0.1007 | −0.127 | −1.20 | 5.6e−9 |
| F1₁ | 0.9651 ± 0.0152 | 0.8593 ± 0.1061 | −0.106 | −0.97 | 1.1e−8 |
| F1-macro | 0.9423 ± 0.0234 | 0.7998 ± 0.1136 | −0.142 | −1.19 | 7.5e−9 |

Augmentation degrades all five metrics significantly (Holm-corrected p < 1e−7 in every case)
and roughly doubles-to-quintuples their variance. As the reviewer anticipated ("if augmentation
degrades performance, that is itself informative"), this confirms our design choice: FantasyID is
a clean, synthetic dataset, so the un-augmented holdout penalises the train–test distribution
mismatch that augmentation introduces. The experiment script is included in the code repository
(`revision_experiments/augmentation_ablation.py`); the per-seed CSVs are archived at [Zenodo].

To rule out the alternative hypothesis that augmentation might trade in-domain accuracy for
*cross-domain* robustness, we additionally evaluated both sets of 30 models zero-shot on the
out-of-domain SIDTD dataset (templates subset; 1000 bonafide, 1222 attack), paired by seed:

| Metric (SIDTD zero-shot) | No aug | With aug | Δ (aug−noaug) | Cohen's d | p (Holm) |
|---|---|---|---|---|---|
| PR-AUC | 0.5554 ± 0.0110 | 0.5639 ± 0.0096 | +0.0085 | +0.55 | 0.077 (n.s.) |
| ROC-AUC | 0.5041 ± 0.0094 | 0.5089 ± 0.0060 | +0.0048 | +0.41 | 0.199 (n.s.) |

Both conditions remain essentially at chance on SIDTD (PR-AUC ≈ 0.55–0.56, ROC-AUC ≈ 0.50–0.51):
the FantasyID→SIDTD domain gap is too large for standard augmentation to close. The small PR-AUC
gain in favour of augmentation does not survive Holm correction, so we make **no claim** that
augmentation improves cross-domain generalization; we report this only for completeness, and it
reinforces that FantasyID is a narrow synthetic domain (consistent with our Limitations
paragraph). Script: `revision_experiments/sidtd_generalization.py`; CSVs archived at [Zenodo].

**Changes:** Sec. 4.1 (Experimental Setup): added one sentence justifying the no-augmentation
protocol with the ablation result, citing the external archive. No additional paper text for the
cross-domain check (null result; reported here for completeness only).

---

### R1.5 — Weak evidence Pareto > scalar (statistical power)
> *Increase the number of repetitions to at least 15 (preferably 30) per condition, report
> confidence intervals and effect sizes... if differences remain non-significant, reframe the
> paper honestly.*

**Response:** We agree the original n=10 comparison was underpowered. We elevated the
Pareto-vs-scalar comparison to the same 30-seed blind-test protocol used elsewhere in the
paper, reporting 95% CIs and Cohen's d with Holm correction across all 10 contrasts
(2 comparisons × 5 metrics). Selection uses the same 500 MOTPE trials: each scalar rule picks
the trial with the highest validation objective (`by_prauc` → PR-AUC; `by_dice` → Dice); the
global winner is retrained over 30 seeds and evaluated on the 15% holdout. The script is in the
repository (`revision_experiments/pareto_vs_scalar_blindtest.py`, non-invasive; writes only to
`results/scalar30/`). Means ± std (n=30):

| Criterion | PR-AUC | Dice | BAcc | F1 (attack) | F1-macro |
|---|---|---|---|---|---|
| Pareto (ours) | 0.9967±0.0021 | 0.8750±0.0180 | 0.9573±0.0184 | 0.9651±0.0152 | 0.9423±0.0234 |
| by_prauc | 0.9953±0.0021 | 0.8589±0.0230 | 0.9481±0.0146 | 0.9569±0.0122 | 0.9294±0.0184 |
| by_dice | 0.9948±0.0027 | 0.8566±0.0147 | 0.9463±0.0170 | 0.9564±0.0121 | 0.9283±0.0184 |

Paired Wilcoxon + Holm (Δ = Pareto − scalar; positive = Pareto higher):

| Comparison | Metric | Δ | 95% CI | d | p_holm | Sig. |
|---|---|---|---|---|---|---|
| Pareto vs by_prauc | PR-AUC | +0.0014 | [0.0003, 0.0024] | 0.49 | 0.062 | ns |
| | Dice | +0.0161 | [0.0060, 0.0262] | 0.59 | 0.043 | * |
| | BAcc | +0.0092 | [0.0006, 0.0178] | 0.40 | 0.061 | ns |
| | F1 (attack) | +0.0082 | [0.0014, 0.0151] | 0.45 | 0.066 | ns |
| | F1-macro | +0.0129 | [0.0024, 0.0234] | 0.46 | 0.050 | * |
| Pareto vs by_dice | PR-AUC | +0.0019 | [0.0008, 0.0030] | 0.64 | 0.007 | ** |
| | Dice | +0.0185 | [0.0116, 0.0254] | 1.00 | 0.0004 | *** |
| | BAcc | +0.0110 | [0.0019, 0.0201] | 0.45 | 0.040 | * |
| | F1 (attack) | +0.0087 | [0.0014, 0.0160] | 0.44 | 0.066 | ns |
| | F1-macro | +0.0140 | [0.0028, 0.0251] | 0.47 | 0.043 | * |

At this power Pareto significantly outperforms both scalar criteria after Holm correction: it
improves localization over `by_prauc` (Dice +0.016, d=0.59) and improves *both* objectives over
`by_dice` (PR-AUC +0.0019, d=0.64; Dice +0.018, d=1.00, large). Each single-objective criterion
sacrifices the objective it does not target, whereas Pareto balances both. This reverses the
original underpowered "no significant difference" finding into a demonstrated advantage, while
preserving Pareto's structural benefit of requiring no a priori objective weighting.

**Changes:** Sec. 4.3 (renamed "Ablation and Selection-Criterion Study"): replaced the n=10 NCV
Table 4 with the 30-seed blind-test results, merged into the single full-width ablation Table 3;
rewrote the text. Script committed to the repository; per-seed CSVs archived on Zenodo.

---

### R1.6 — Challenge comparison not on same test set
> *Either remove the challenge comparison from Table 5 and move it to a clearly-labeled
> appendix, or add a boldface caveat directly in the table caption.*

**Response:** The reviewer offered two options: remove the comparison or add a boldface caveat.
We chose the first. The challenge comparison table has been removed from the main text, and
Sec. 4.5 is now a single short paragraph stating explicitly that:
1. **Evaluation sets differ.** Our internal 15% holdout versus the official challenge test set
   (FantasyID test partition plus a private out-of-domain PXL Vision set), so the comparison is
   indicative rather than controlled.
2. **No superiority claim.** We make no superiority claim and imply no ranking; the numbers only
   situate DocVerify in context.
3. **Training data differ markedly.** Sunlight adds over 60,000 external identity images and the
   TruFor-based entries use a backbone pre-trained on ~828,000 manipulation images, whereas
   EdgeDoc standalone, UAM-Biometrics and DocVerify train on FantasyID alone.

This removes the competitive framing the reviewers (R1.6, R2.1.2, R2.2.2) objected to.

**Changes:** Sec. 4.5: removed the challenge comparison table; condensed the prose to one
paragraph with all caveats inline; removed competitive/ranking language.

---

## Moderate Issues

### R1.7 — Architecture not state-of-the-art (pre-trained backbone)
> *Consider adding at least one pre-trained backbone as a baseline (e.g. ResNet-18 with
> ImageNet weights).*

**Response:** We thank the reviewer. To verify that our MOTPE/Pareto HPO protocol is not tied to our lightweight encoder, we re-ran the *identical* methodology (50 MOTPE trials, 5 inner folds, Pareto selection by distance to the ideal point, 30-seed blind test) on an **ImageNet-pretrained ResNet-18** backbone with a U-Net decoder. To keep the additional cost tractable we used a single MOTPE study rather than the full 10-fold nested protocol. The complete, self-contained experiment script is provided in the repository at `revision_experiments/resnet18_motpe.py` (non-invasive: it reuses the original training pipeline without modifying any weights or results, and writes only to `revision_experiments/results/resnet18/`). Results on the blind holdout (n=30):

| Model | PR-AUC | Dice | BAcc | F1 (attack) | F1-macro |
|---|---|---|---|---|---|
| DocVerify | 0.9967±0.0021 | 0.8750±0.0180 | 0.9573±0.0184 | 0.9651±0.0152 | 0.9423±0.0234 |
| ResNet18-MOTPE | 0.9942±0.0025 | 0.8894±0.0111 | 0.9629±0.0108 | 0.9779±0.0069 | 0.9613±0.0114 |

Paired Wilcoxon + Holm: DocVerify higher PR-AUC (p=3e-4, d=0.75); ResNet-18 higher Dice (p=8e-3, d=−0.60), F1 (p=4e-3) and F1-macro (p=4e-3); BAcc not significant (p=0.29). MOTPE selected `loss_w_mask=2.69` for ResNet-18, close to the 2.46 selected for DocVerify, showing the Pareto balance is consistent across architectures. Since DocVerify is better on detection while ResNet-18 is better on localization, neither model dominates the other, which confirms that the methodology transfers cleanly to a modern pretrained backbone. Full code and per-seed CSVs are also archived at [Zenodo DOI].

**Changes:** Sec. 4.6 (new **Generalization** subsection): the *Across architectures* paragraph reports this result (ResNet-18 PR-AUC 0.994, Dice 0.889; neither model dominates), framing the contribution as the multi-objective selection methodology rather than the specific architecture; experiment script committed to the repository (`revision_experiments/resnet18_motpe.py`) and archived on Zenodo.

---

### R1.8 — Qualitative analysis: only one success case
> *Please include at least three examples: one success, one localization failure, and one
> false positive on a bonafide document.*

**Response:** We agree. The original single-example figure showed only a near-perfect
case and gave an overly optimistic impression of the model's behaviour. We replaced it
with a figure contrasting the three requested outcomes, all drawn from the same blind-test
model (seed 42, no retraining) so the comparison is internally consistent:
1. **Correct detection** — a forged photograph correctly flagged (p_attack=0.96) and
   tightly localized (Dice 0.96).
2. **Localization failure** — a forgery correctly flagged (p_attack=1.00) but whose
   predicted mask drifts onto the portrait (Dice 0.22), the dominant failure mode for Dice.
3. **Bona-fide false positive** — a genuine document wrongly flagged as an attack
   (p_attack=0.96) with a spurious mask, illustrating the rare false positives discussed
   in the Discussion.

Cases were selected programmatically (highest-Dice face-swap attack, lowest-Dice detected
attack, and highest-confidence bona-fide false positive) to be representative rather than
cherry-picked; the generation script is committed for reproducibility. This makes the
localization variance and false-positive behaviour visible directly in the paper, with the
quantitative variance already reported in Tables 2 and 3.

**Changes:** Replaced the single-example qualitative figure with a 2×3 figure
(Fig. `fig:qualitative`, Sec. 4.7) showing ground truth (top) vs. DocVerify prediction
(bottom) for the three cases above, and rewrote the accompanying paragraph to describe both
failure modes. Generation script committed (`revision_experiments/qualitative_examples.py`),
which reuses the blind-test checkpoint and its saved decision threshold (no retraining).

---

## Minor Issues

### R1.9 — Citation precision
> *Verify these numerical claims against the original sources... "3,284 images" for FantasyID
> does not appear in the original paper; "876,000 images" for TruFor... original reports ~828K.*

**Response:** We thank the reviewer for the careful check and have verified every
numerical claim against its original source, correcting the inaccuracies.

1. **FantasyID "3,284 images".** The reviewer is correct that this figure is not a
   number reported in the FantasyID paper; it is the size of the release we actually
   process. To avoid attributing our own count to the dataset, Sec. 3.1 now reads
   "The FantasyID release we process contains 3,284 labeled images" and states the
   split explicitly (2,791 development + 493 holdout = 3,284).

2. **TruFor "876,000 images".** Corrected to ~828,000 to match the figure reported by
   the original TruFor paper; the related-work description (Sec. 2) now reads
   "~828,000 pre-training images", consistent with the "~828,000 manipulation images"
   already stated in Sec. 4.5.

3. **Bibliography encoding.** While verifying the references we also found and fixed
   two malformed ampersand entries (`&amp;` HTML encoding) in the bibliography, in
   the Optuna KDD'19 and Miettinen records, which now use the correct LaTeX `\&`.

**Changes:** Sec. 3.1 wording (FantasyID count attributed to our processing, explicit
split); Sec. 2 TruFor pre-training figure 876,000 → ~828,000; `references.bib`
`&amp;` → `\&` (two entries).

---

### R1.10 — Computational cost vs benefit
> *The paper would benefit from a brief discussion of whether the computational investment was
> proportional to the gain.*

**Response:** We agree, and we now make explicit what the ~59-hour budget actually pays for.
That figure is the cost of the **full scientific protocol**, not the cost of *using* the
method. It breaks down (measured wall-clock, single RTX 5060 Ti, VRAM-cached) as follows:

| Phase | What it is | Trainings | Wall-clock |
|---|---|---|---|
| Nested HPO search | 10 outer folds × 50 MOTPE trials × 5 inner splits | 2 500 | ~50 h |
| Outer final models | one retrain per outer fold (unbiased CV estimate) | 10 | ~1 h |
| Blind-test ablation | 30 seeds × 4 variants | 120 | ~12 h |
| **Total scientific protocol** | | | **~59–63 h** |
| **Practical deployment** | one fixed config, one seed | 1 | **7.7 ± 1.0 min** |

The point is the contrast in the last two rows. The multi-hour budget buys *scientific rigor*:
an unbiased nested 10-fold cross-validation estimate plus 30-seed statistical robustness for the
ablation. A practitioner who only needs the deployable artifact does **not** pay this: once the
Pareto-selected configuration is fixed, retraining the single deployed model from scratch (up to
100 epochs with early stopping) takes only **7.7 ± 1.0 min** per seed (range 6.0–10.2 min,
measured over the 30 blind-test seeds). The lightweight encoder is the design choice that keeps
this marginal cost at minute scale, and the Pareto loss weighting that the search yields adds
**no inference-time cost** while delivering a statistically significant localization gain
(Dice, d = 1.01 over equal weights; Sec. 4.3). The investment is therefore amortized once and is
proportional to the gain.

**Changes:** Sec. 4.1 (Experimental Setup): added one sentence clarifying that the 59-hour budget
covers the full scientific protocol, whereas retraining the single deployed model from scratch
takes 7.7 ± 1.0 min. Full breakdown reported here in the response letter to respect the 7-page
limit; per-seed training times are released in the reproducibility archive (Zenodo DOI).

---

# Reviewer 2

## General Comments

### R2.1.1 — Single synthetic dataset limits generalizability
> *This severely limits the generalizability of the claims... leaving the practical
> applicability of DocVerify entirely undemonstrated on real-world documents.*

**Response:** We have addressed this on two fronts. (1) *Cross-domain evidence:* the new
**Generalization** subsection (Sec. 4.6, *Across datasets*) now reports the zero-shot SIDTD
evaluation (PR-AUC 0.555, ROC-AUC 0.504), so the practical out-of-domain behavior is no
longer undemonstrated; it is shown to be near chance and discussed candidly (see R1.2).
(2) *Scope restriction:* we explicitly bound the operational scope to the synthetic FantasyID
domain in Sec. 4.6 and reframed the Conclusion's future-work statement around the cross-domain
gap exposed on SIDTD. The challenge comparison (Sec. 4.5) is already hedged as indicative only.

**Changes:** Sec. 4.6 *Across datasets* paragraph (SIDTD result + scope-bounding statement);
Conclusion future-work sentence reframed around the SIDTD cross-domain gap (Sec. 4.6).

---

### R2.1.2 — Table 5 needs more caution
> *Despite acknowledging this discrepancy, the author draws competitive conclusions... that
> are not statistically justified.*

**Response:** Addressed jointly with R1.6 (and R2.2.2). We removed the challenge comparison
table entirely and condensed Sec. 4.5 to a single paragraph that states the evaluation sets
differ and the results are not directly comparable. We no longer claim superiority over any
challenge entry.

**Changes:** See R1.6 — challenge comparison table removed; Sec. 4.5 condensed with all caveats
inline and competitive language removed.

---

### R2.1.3 — Blind holdout protocol timing
> *Does not clarify whether the 15% holdout was defined before or after the NCV protocol was
> designed. If the holdout was used at any point during development... the blind test results
> may be optimistically biased.*

**Response:** We thank the reviewer for raising this. The 15% holdout was carved off **once, at
the very start**, before any hyperparameter search or model-design decision, and was accessed
only for the final 30-seed blind evaluation. This is verifiable in the public repository on
three independent grounds:

1. **Structural (code).** In `main.py`, `load_and_prepare_data()` applies a single
   `GroupShuffleSplit(test_size=0.15, random_state=42)` to separate the holdout *first*; the
   nested cross-validation (`run_nested_cv`) receives only the 85% development partition, and
   `run_blind_test` is invoked exactly once at the end. Anti-leakage assertions by document
   identity (`stem`) guarantee the partitions are disjoint. Crucially, the classification
   threshold is **not** tuned on the holdout: in `train.py` it is selected by a sweep over a
   development-internal validation split (`loader_sel`, itself a `GroupShuffleSplit` within the
   development set) and then applied unchanged to the holdout. No hyperparameter, model, or
   decision threshold is informed by the holdout.

2. **Chronological (git history).** The holdout split predates all experiments: it is present
   from the first PyTorch commit, before the HPO ranges were refined and before the blind test
   was run. Configuration changes made *after* the blind test did not touch `test_size`,
   early-stopping patience, epoch budgets, or HPO ranges.

3. **Discipline.** No design choice (early-stopping patience, epoch budget, HPO ranges,
   classification threshold) was at any point informed by holdout performance.

**Changes:** Sec. 3.1: added an explicit sentence (highlighted) stating that the split was
performed once, before any hyperparameter search or model-design decision, and that the holdout
was accessed only for the final blind evaluation. The structural/chronological evidence above is
provided here for the reviewer; the repository (`main.py`, `train.py`) allows full verification.

---

### R2.1.4 — Class imbalance not justified
> *No justification is provided for this design choice, and no analysis of its impact is
> presented.*

**Response:** Addressed jointly with R2.2.3 below, where we add both a justification and a
sensitivity analysis. In brief: the imbalance is mild (2.5:1), the positive (attack) class is
the *majority* (so PR-AUC, our primary detection metric, is robust to it), and the new
sensitivity analysis shows that correcting the imbalance has no statistically significant effect
on PR-AUC or Dice.

**Changes:** Sec. 3.1 (justification + sensitivity note); full analysis and table in R2.2.3 below.

---

### R2.1.5 — No data augmentation, no justification
> *An unusual and potentially limiting choice for a dataset of only ~2,791 training images.
> No ablation or justification for this decision is provided.*

**Response:** We now provide both an ablation and a justification (see R1.4 for the full
30-seed paired comparison). In short, applying five standard augmentations significantly degrades
every metric on the clean synthetic FantasyID holdout (e.g. Dice 0.875→0.780, PR-AUC
0.997→0.958; all Holm-corrected p < 1e−7), confirming that the no-augmentation protocol is a
deliberate, empirically-grounded choice rather than an oversight. Sec. 4.1 now states this with
a one-sentence justification citing the external archive.

**Changes:** See R1.4 (augmentation ablation added to Sec. 4.1; full 30-seed table in this
letter under R1.4).

---

## Major Concerns

### R2.2.1 — Insufficient cross-domain generalization evidence
> *Must either provide results on at least one additional benchmark (e.g. MIDV-2020,
> DocTamper, or any real-world document dataset) or substantially revise the scope of the
> claims to explicitly restrict them to the FantasyID domain, with appropriate hedging.*

**Response:** We chose both options the reviewer offers, not one. We *provide* a cross-domain
result, the zero-shot SIDTD evaluation now reported in the new **Generalization** subsection
(Sec. 4.6; PR-AUC 0.555, ROC-AUC 0.504, near chance), *and* we explicitly restrict the scope:
Sec. 4.6 states that performance bounds the system to the synthetic FantasyID domain, and the
Conclusion's future-work statement is reframed around closing this gap via multi-source
training. All competitive language relative to the challenge was already removed (see R1.6).
SIDTD was used as the additional benchmark because it is a public ID-document dataset directly
relevant to our task; MIDV-2020/DocTamper target different settings (capture variation, generic
document tampering) and are left for the multi-source future work.

**Changes:** Sec. 4.6 *Across datasets* paragraph (SIDTD cross-domain result + explicit
FantasyID-domain scope restriction); Conclusion future-work reframed (see R1.2, R2.1.1).

---

### R2.2.2 — Table 5 methodologically unsound
> *Add a clearly formatted disclaimer box within the table itself (not only in the text) and
> remove any competitive language from the conclusions.*

**Response:** Addressed jointly with R1.6 and R2.1.2. Rather than add a disclaimer box inside the
table, we removed the challenge comparison table entirely, which eliminates the methodological
concern at its source: there is no longer a table that could be read as a direct comparison.
Sec. 4.5 is now a single paragraph that states the evaluation sets differ and the results are
indicative only. All competitive language has been removed from the comparison text and from the
Conclusion, which now states only that DocVerify is "broadly competitive... indicative only,
since challenge systems used a different test set."

**Changes:** See R1.6 — challenge comparison table removed; Sec. 4.5 condensed with all caveats
inline and competitive language removed from Sec. 4.5 and the Conclusion.

---

### R2.2.3 — Class imbalance: justification + ablation required
> *The model may be biased toward predicting the majority class. Provide a theoretical or
> empirical justification... and include an ablation or sensitivity analysis on class
> imbalance correction and its impact on PR-AUC and Dice.*

**Response:** We thank the reviewer and have added both a justification and a sensitivity
analysis.

*Justification.* Two points make the uncorrected imbalance benign in our setting. (i) The ratio
is **mild** (2.5:1; 71.6% attack / 28.4% bonafide), far from the regime where loss reweighting
is typically required. (ii) Crucially, the **positive class (label = 1) is *attack*, the
*majority***. The conventional `pos_weight` correction up-weights the positive term, which here
would up-weight the *already-majority* class — the opposite of imbalance correction. A correct
imbalance correction in our case requires `pos_weight = N_neg/N_pos ≈ 0.40` (down-weighting the
majority). We therefore evaluate the full direction of the correction, not just one value.
Moreover, our primary detection metric is **PR-AUC**, which is threshold-free and robust to prior
class balance.

*Sensitivity analysis (new).* We retrained the selected final configuration with
`pos_weight ∈ {0.40, 1.0, 2.52}` — correction toward the minority (bonafide), no correction
(the paper's setting), and the naive formula that *aggravates* the imbalance — using the same
blind-test protocol and **30 seeds per value** (the `pos_weight = 1.0` column reuses the paper's
existing 30-seed blind test; the two new values were run identically). Results:

| `pos_weight` | PR-AUC | Dice | bACC | recall (attack) | recall (bonafide) |
|---|---|---|---|---|---|
| 0.40 (→ minority) | 0.9973 ± 0.0018 | 0.879 ± 0.017 | 0.961 | 0.943 | **0.980** |
| 1.0 (paper) | 0.9967 ± 0.0021 | 0.875 ± 0.018 | 0.957 | 0.944 | 0.971 |
| 2.52 (→ majority) | 0.9967 ± 0.0019 | 0.869 ± 0.024 | 0.955 | 0.944 | 0.967 |

Paired Wilcoxon + Holm–Bonferroni vs. the `pos_weight = 1.0` baseline shows **no statistically
significant difference** on any metric (all $p_{\text{Holm}} > 0.05$; $|d| < 0.26$). Two
observations: (a) PR-AUC and Dice are flat across the grid, so the no-correction choice costs
nothing; (b) the **minority (bonafide) recall is consistently *higher* than the majority (attack)
recall** (0.971 vs. 0.944 at baseline), directly refuting the concern that the model is biased
toward the majority class. The small, non-significant monotonic trend (correcting toward the
minority marginally raises bonafide recall and Dice; aggravating lowers them) confirms the
analysis is sensitive enough to detect the *direction* of the effect, while its *magnitude* is
negligible — exactly as expected for a mild ratio with a near-saturated detector.

**Changes:** Sec. 3.1: added a sentence noting the mild ratio, that the positive (attack) class
is the majority, PR-AUC robustness, and that a `pos_weight` sweep confirms no significant effect.
To respect the journal's 7-page limit, the full per-class table is provided here in this response
letter rather than in the manuscript. New experiment script and outputs are in the repository
(`revision_experiments/pos_weight_sensitivity.py`,
`revision_experiments/results/pos_weight/`).

---

### R2.2.4 — Pareto vs Scalar does not support claimed advantage
> *Provide a stronger argument for the structural or practical advantages of Pareto selection
> beyond predictive accuracy (e.g. reduced sensitivity to the choice of scalar weight, better
> coverage of the trade-off landscape, or improved robustness across seeds).*

**Response:** Addressed jointly with R1.5: at n=30 Pareto significantly outperforms both scalar
criteria after Holm correction (`by_dice`: PR-AUC d=0.64, Dice d=1.00; `by_prauc`: Dice d=0.59,
F1-macro significant). Beyond predictive accuracy, Pareto requires no a priori commitment to a
primary objective and exposes the full trade-off landscape (Fig. 3), consistent with the DeepID
protocol that ranks detection and localization independently.

**Changes:** See R1.5.

---

### R2.2.5 — Architecture description incomplete (reproducibility)
> *Missing/ambiguous: FC-layer config in classification head (dropout, activations); decoder
> block structure (# conv layers, kernel sizes, BN); skip-connection concatenation mechanism;
> early-stopping monitored metric.*

**Response:** We have completed the architecture description with all four missing details,
matching the released implementation (`model.py`, `train.py`; archived at the Zenodo DOI):

1. **Classification head (Sec. 3.2.2).** Global average pooling of the 256-channel
   bottleneck feeds four fully-connected layers (256→32→16→16→1). A dropout layer with
   rate `p` (an HPO parameter) precedes *each* linear layer, and LeakyReLU (α=0.2) follows
   each *hidden* layer; the final layer is linear, producing the classification logit.

2. **Decoder block structure (Sec. 3.2.3).** Each of the five U-Net decoder stages performs
   bilinear ×2 upsampling, channel-wise concatenation of the matching-resolution encoder
   skip, then **two 3×3 convolutions** with LeakyReLU (α=0.2) and **no batch normalization**
   (in contrast to the encoder, which does use BN). Channel width halves at each stage from
   `C_dec` down to `C_dec/16`, and a final 1×1 convolution emits the segmentation logit.

3. **Skip-connection mechanism (Sec. 3.2.3).** Encoder skips are tapped *before* pooling at
   resolutions 224/112/56/28/14 and concatenated channel-wise (not summed) with the
   upsampled decoder feature at the matching resolution.

4. **Early-stopping monitored metric (Sec. 3.4).** The default multitask model is monitored
   on the validation distance to the ideal point, √((1−PR-AUC)²+(1−Dice)²) (Eq. 4) — the
   same criterion as Pareto selection — with patience 12; ReduceLROnPlateau steps on the
   same monitor. (The single-task ablations `cls_only`/`seg_only` monitor 1−PR-AUC and
   1−Dice respectively.)

**Changes:** Sec. 3.2.2 (classification head: dropout/activation/final-linear detail),
Sec. 3.2.3 (decoder: two 3×3 convs, LeakyReLU, no BN; skip tapped before pooling and
concatenated channel-wise), Sec. 3.4 (early-stopping monitored metric tied to Eq. 4).
To stay within the 7-page limit, the qualitative figure was slightly reduced in size with
no loss of content.

---

## Minor Concerns

### R2.3.1 — Equations must be explicitly cited in the text
**Response:** Every numbered equation is now referred to by number at the point where
its symbols are used. The combined loss (Eq. 1), the segmentation/Dice loss (Eq. 2),
the multi-objective HPO formulation (Eq. 3), and the ideal-point selection criterion
(Eq. 4) are each cited explicitly in the surrounding text; the previously unlabeled
Dice loss now carries a label and is referenced.

**Changes:** Sec. 3.3 (Loss Functions) and Sec. 3.4 (Multi-Objective HPO): added explicit
`Eq.~\ref{}` citations for all four display equations; added a label to the Dice loss.

---

### R2.3.2 — F1@0.5 must be formally defined at first occurrence
**Response:** We now define the metric at its first occurrence, including the base F1
formula. F1@0.5 is the F1 score (the harmonic mean of precision and recall,
F1 = 2PR/(P+R)) of the bona-fide/attack decision taken at classification threshold 0.5,
and F1_loc is the pixel-level F1 of the predicted mask. With the challenge comparison
table removed (see R1.3), F1@0.5 first appears in Sec. 4.5, where all definitions are now
given inline.

**Changes:** Sec. 4.5: parenthetical definitions at first use of F1@0.5 ("the F1 score,
2PR/(P+R), of the bona-fide/attack decision at classification threshold 0.5") and F1_loc
("pixel-level F1 of the predicted mask"), now spelling out the base F1 harmonic-mean
formula.
