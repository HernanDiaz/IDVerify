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
| Cross-domain / 2nd dataset (SIDTD) | R1.2 | R2.1.1, R2.2.1 | _TBD_ |
| Table 5 comparison not on same test set | R1.6 | R2.1.2, R2.2.2 | **DONE** — boldface caveat in Table 5 caption; competitive language removed from Sec. 4.5 + Conclusion |
| Pareto vs scalar — weak evidence | R1.5 | R2.1 (implied), R2.2.4 | _TBD_ |
| Data augmentation | R1.4 | R2.1.5 | _TBD_ |
| Class imbalance | — | R2.1.4, R2.2.3 | **DONE** — justification (positive=majority, mild ratio) + pos_weight sensitivity sweep (30 seeds, all p_Holm>0.05); note in Sec. 3.1, full table in this letter |
| TruFor fine-tuned omitted | R1.3 | — | _TBD_ |
| EdgeDoc attribution error | R1.1 | — | **DONE** — Table 5 split into fusion (0.958) + standalone (0.43) rows; loc 0.621→0.686 typo fixed |
| Reproducibility / architecture detail | — | R2.2.5 | _TBD_ |
| Blind-holdout protocol timing | — | R2.1.3 | **DONE** — Sec. 3.1 sentence + 3-leg evidence (code isolation, git chronology, threshold from dev-internal val) |
| Pre-trained backbone baseline | R1.7 | — | _TBD_ |
| Qualitative: failure cases | R1.8 | — | _TBD_ |
| Citation precision | R1.9 | — | _TBD_ |
| Compute cost vs benefit | R1.10 | — | _TBD_ |
| Equation citing / F1@0.5 definition | — | R2.3.1, R2.3.2 | _TBD_ |

---

# Reviewer 1

## Major Issues

### R1.1 — EdgeDoc attribution error in Table 5
> *Table 5 reports AG/EdgeDoc with F1@0.5 = 0.958... EdgeDoc alone achieves F1 = 0.43 on
> FantasyID. The value 0.96 corresponds to the EdgeDoc+TruFor fusion... Please clarify the
> exact origin of the 0.958 value.*

**Response:** We thank the reviewer for catching this. The reviewer is correct. We verified
the value against George & Marcel (2025, Table 2): on the FantasyID partition of the DeepID
2025 leaderboard, EdgeDoc **standalone** achieves F1 = 0.43, TruFor achieves F1 = 0.81, and
the **EdgeDoc+TruFor fusion** achieves F1 = 0.96. Our original "AG/EdgeDoc = 0.958" row
conflated the third-place team's *fusion* submission with the EdgeDoc architecture in
isolation. We have corrected Table 5 to report both rows explicitly — EdgeDoc+TruFor fusion
(0.96) and EdgeDoc standalone (0.43) — and we now note that the fusion relies on the
pre-trained TruFor backbone, whereas EdgeDoc standalone is the FantasyID-only comparator.
With the corrected attribution, DocVerify (0.969) substantially outperforms the only other
FantasyID-only architectures (EdgeDoc standalone 0.43; UAM-Biometrics 0.712).

**Changes:** Table 5 (Sec. 4.5): replaced the single "AG/EdgeDoc (3rd) 0.958" row with two
rows — "EdgeDoc+TruFor fusion (3rd) 0.96" and "EdgeDoc standalone 0.43", both citing
George & Marcel (2025, Table 2). Updated the two surrounding text passages (training-data
disparity paragraph and the Track-1 comparison sentence) accordingly. Competitive language
moderated.

**Note (internal, not for reviewer):** the fusion row's localization value was corrected from
0.621 (LaTeX transcription typo in the previous draft) to **0.686**, verified against
Korshunov et al. (2025), Tables 2 (detection, AG FantasyID = 0.958) and 3 (localization,
AG FantasyID = 0.686), and consistent with the repository code
(`evaluate_challenge_metrics.py`, `challenge.py`). Resolved.

---

### R1.2 — Selective omission of SIDTD results
> *Running the experiments... and then excluding them because they were unfavorable is a
> classic file-drawer problem. I strongly recommend including SIDTD results... A candid
> discussion of the domain gap would strengthen the paper.*

**Response:** _TBD_

**Changes:** _TBD_

---

### R1.3 — TruFor fine-tuned results omitted
> *Please either add the TruFor fine-tuned row to Table 5 or explain in the text why it was
> excluded.*

**Response:** _TBD_

**Changes:** _TBD_

---

### R1.4 — No data augmentation
> *Please add standard augmentation (moderate rotation, brightness variation, Gaussian blur)
> and report results with and without it.*

**Response:** _TBD_

**Changes:** _TBD_

---

### R1.5 — Weak evidence Pareto > scalar (statistical power)
> *Increase the number of repetitions to at least 15 (preferably 30) per condition, report
> confidence intervals and effect sizes... if differences remain non-significant, reframe the
> paper honestly.*

**Response:** _TBD_

**Changes:** _TBD_

---

### R1.6 — Challenge comparison not on same test set
> *Either remove the challenge comparison from Table 5 and move it to a clearly-labeled
> appendix, or add a boldface caveat directly in the table caption.*

**Response:** We kept Table 5 in the main text, as it gives readers useful context, but
strengthened the caveats and removed all competitive language (addressing R1.6, R2.1.2 and
R2.2.2 jointly):
1. **Boldface caveat in the caption.** The Table 5 caption now states, in bold, that *the
   evaluation sets differ and the results are not directly comparable*: the $\dagger$ entries use
   the official challenge test set (FantasyID test partition plus a private out-of-domain
   PXL Vision set), whereas DocVerify is evaluated on our internal 15% holdout; no ranking is
   implied.
2. **Competitive language removed.** The sentence that claimed DocVerify "substantially exceeds"
   and "approaches" specific systems has been replaced by an explicit disclaimer that we do not
   claim superiority over any challenge entry and report the numbers only to situate DocVerify in
   context.
3. The surrounding text still explains the protocol difference and the training-data disparity
   (e.g. Sunlight augments FantasyID with external identity documents; TruFor is pre-trained on
   general-purpose manipulation images).

**Changes:** Sec. 4.5, Table 5 caption: added the boldface "Evaluation sets differ and results
are not directly comparable" caveat. Sec. 4.5 text: removed competitive/ranking language and
replaced it with a non-superiority disclaimer. Fixed the fusion-row citation to point only to
the challenge leaderboard (the source of the 0.958 figure).

---

## Moderate Issues

### R1.7 — Architecture not state-of-the-art (pre-trained backbone)
> *Consider adding at least one pre-trained backbone as a baseline (e.g. ResNet-18 with
> ImageNet weights).*

**Response:** _TBD_

**Changes:** _TBD_

---

### R1.8 — Qualitative analysis: only one success case
> *Please include at least three examples: one success, one localization failure, and one
> false positive on a bonafide document.*

**Response:** _TBD_

**Changes:** _TBD_

---

## Minor Issues

### R1.9 — Citation precision
> *Verify these numerical claims against the original sources... "3,284 images" for FantasyID
> does not appear in the original paper; "876,000 images" for TruFor... original reports ~828K.*

**Response:** _TBD_

**Changes:** _TBD_

---

### R1.10 — Computational cost vs benefit
> *The paper would benefit from a brief discussion of whether the computational investment was
> proportional to the gain.*

**Response:** _TBD_

**Changes:** _TBD_

---

# Reviewer 2

## General Comments

### R2.1.1 — Single synthetic dataset limits generalizability
> *This severely limits the generalizability of the claims... leaving the practical
> applicability of DocVerify entirely undemonstrated on real-world documents.*

**Response:** _TBD_ (see also R2.2.1, R1.2)

**Changes:** _TBD_

---

### R2.1.2 — Table 5 needs more caution
> *Despite acknowledging this discrepancy, the author draws competitive conclusions... that
> are not statistically justified.*

**Response:** Addressed jointly with R1.6 (and R2.2.2). We removed the competitive conclusions
and added a boldface caveat in the Table 5 caption stating that the evaluation sets differ and
the results are not directly comparable. We no longer claim superiority over any challenge entry.

**Changes:** See R1.6 — Table 5 caption caveat + removal of competitive language in Sec. 4.5.

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

**Response:** _TBD_ (see also R1.4)

**Changes:** _TBD_

---

## Major Concerns

### R2.2.1 — Insufficient cross-domain generalization evidence
> *Must either provide results on at least one additional benchmark (e.g. MIDV-2020,
> DocTamper, or any real-world document dataset) or substantially revise the scope of the
> claims to explicitly restrict them to the FantasyID domain, with appropriate hedging.*

**Response:** _TBD_ (see also R1.2)

**Changes:** _TBD_

---

### R2.2.2 — Table 5 methodologically unsound
> *Add a clearly formatted disclaimer box within the table itself (not only in the text) and
> remove any competitive language from the conclusions.*

**Response:** Addressed jointly with R1.6 and R2.1.2. As requested, the disclaimer is now *inside
the table* (a boldface caveat in the Table 5 caption stating the evaluation sets differ and the
results are not directly comparable), not only in the running text. All competitive language has
been removed from the comparison text and from the Conclusion, which now states only that
DocVerify is "broadly competitive... indicative only, since challenge systems used a different
test set."

**Changes:** See R1.6 — boldface caveat in the Table 5 caption + removal of competitive language
in Sec. 4.5 and the Conclusion.

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

**Response:** _TBD_ (see also R1.5)

**Changes:** _TBD_

---

### R2.2.5 — Architecture description incomplete (reproducibility)
> *Missing/ambiguous: FC-layer config in classification head (dropout, activations); decoder
> block structure (# conv layers, kernel sizes, BN); skip-connection concatenation mechanism;
> early-stopping monitored metric.*

**Response:** _TBD_

**Changes:** _TBD_

---

## Minor Concerns

### R2.3.1 — Equations must be explicitly cited in the text
**Response:** _TBD_

**Changes:** _TBD_

---

### R2.3.2 — F1@0.5 must be formally defined at first occurrence
**Response:** _TBD_

**Changes:** _TBD_
