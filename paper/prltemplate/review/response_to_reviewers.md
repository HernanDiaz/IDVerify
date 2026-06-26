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
| Table 5 comparison not on same test set | R1.6 | R2.1.2, R2.2.2 | _TBD_ |
| Pareto vs scalar — weak evidence | R1.5 | R2.1 (implied), R2.2.4 | _TBD_ |
| Data augmentation | R1.4 | R2.1.5 | _TBD_ |
| Class imbalance | — | R2.1.4, R2.2.3 | _TBD_ |
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

**Response:** _TBD_

**Changes:** _TBD_

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

**Response:** _TBD_ (see also R2.2.2, R1.6)

**Changes:** _TBD_

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

**Response:** _TBD_ (see also R2.2.3)

**Changes:** _TBD_

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

**Response:** _TBD_ (see also R1.6)

**Changes:** _TBD_

---

### R2.2.3 — Class imbalance: justification + ablation required
> *The model may be biased toward predicting the majority class. Provide a theoretical or
> empirical justification... and include an ablation or sensitivity analysis on class
> imbalance correction and its impact on PR-AUC and Dice.*

**Response:** _TBD_

**Changes:** _TBD_

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
