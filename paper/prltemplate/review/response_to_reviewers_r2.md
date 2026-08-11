# Response to Reviewers — Round 2 (Minor Revision)

**Manuscript:** PRLETTERS-D-26-00496R1
**Title:** Multi-Objective Pareto hyperparameter optimization for joint detection and localization of forged identity documents
**Author:** Hernán Díaz Rodríguez

We thank the Associate Editor and both reviewers for their positive assessment and
for the concrete, constructive suggestions. We are grateful to Reviewer 1 for
recommending acceptance and verifying the Zenodo archive, and to Reviewer 2 for the
careful reading that sharpened the statistical claims. All requested changes are
editorial/interpretive; no new experiments were required. Round-2 changes are
highlighted in blue in the revised PDF (round-1 changes are now shown as normal
text). Section references below point to the revised manuscript.

---

## Associate Editor

> *Please make a minor editorial revision to ensure that the discussion of the Pareto
> versus scalar selection results accurately reflects the statistical significance
> reported in Table 3 and briefly acknowledge the limited zero-shot generalization to
> SIDTD in the abstract and discussion. No additional experiments are required.*

**Response:** All three requests are addressed and no new experiments were needed.
(i) The Pareto-vs-scalar discussion in Sec. 4.4 has been rewritten to match the
significance actually reported (see R2.1). (ii) The abstract now acknowledges the
bounded in-domain scope and the near-chance zero-shot SIDTD result (see R2.3-abstract).
(iii) The Discussion (Sec. 5.2, Generalization) already noted the near-chance zero-shot
transfer to SIDTD and now also states that standard augmentation does not close the gap
(see R2.2).

**Changes:** Sec. 4.4 recalibrated; abstract sentence added; Discussion generalization
paragraph extended. All changes are highlighted in blue.

---

## Editor-in-Chief checklist

1. **Title (<=10–15 words, grammatical).** Compliant: the title is 13 words and unchanged.
2. **Conclusions reflect strengths and weaknesses, differ from the abstract, longer.**
   Done. The Conclusion now adds a paragraph that contrasts the method's strengths (no
   manual loss-weight tuning; the MOTPE protocol transfers to a ResNet-18 backbone, so the
   contribution is methodological not architecture-specific; the evaluation is fully
   reproducible with openly archived per-seed results) against its limitations (the
   Pareto-over-scalar advantage is partial and concentrated in localization; performance is
   bounded to the in-domain FantasyID setting, with near-chance zero-shot on SIDTD not closed
   by augmentation; the nested protocol is computationally heavy). The Pareto-vs-scalar
   statement in finding (i) was recalibrated to match Sec. 4.4.
3. **Bibliography well organized (~30 items, not excessive arXiv / single series).**
   Done. We reduced the visible arXiv presence without dropping references. Two items
   previously cited as arXiv preprints are now cited by their published venue: TruFor
   (Guillaro et al., CVPR 2023) and FantasyID (Korshunov et al., IJCB 2025). For five
   further entries that were already published but carried a residual arXiv identifier
   (EdgeNeXt, FakeIDet, FatFormer, MMFusion, UniFD), we removed the arXiv tag so they cite
   only the published venue. Only one arXiv-only item now remains (the EdgeDoc challenge
   report, which has no formal venue). The list holds 32 items, no single conference series
   dominates, and grouped citations are discussed individually in the text. After adding the
   three recent Pattern Recognition Letters references requested under item 4, we removed three
   non-essential citations (a threat report, a regulatory reference, and one of a pair of
   general-forensics examples) so the list holds 32 items, close to the recommended size.
4. **Cite recent Pattern Recognition Letters work.**
   Done. Sec. 2.1 now cites three recent Pattern Recognition Letters papers on
   deepfake/face-manipulation detection, each commented individually: a two-stream network
   with hierarchical supervision (Liang et al., 2023, vol. 172), RGB-depth feature fusion
   (Leporoni et al., 2024, vol. 181), and a self-attention discriminator (Wang et al., 2024,
   vol. 183). This situates our document-forgery work within the journal's forensics
   literature and connects it to the PRL readership.
5. **Format: single-spaced, double-column, <=8 pages.** Compliant: the paper uses the
   elsarticle two-column (5p) format and fits within 8 pages including references.

---

## Reviewer 1 (recommends acceptance)

> *TruFor fine-tuned (R1.3): ... A footnote in Sec. 4.5 would improve completeness.*

**Response:** We thank Reviewer 1 for recommending acceptance and verifying the Zenodo
archive. As suggested, we added a footnote in Sec. 4.5 noting that a controlled comparison
against a FantasyID-fine-tuned TruFor, evaluated on the same internal holdout, is released
with the code (`trufor_finetuned_scores.csv`), which improves the completeness of the section.

**Changes:** Sec. 4.5: added a footnote pointing to the released fine-tuned-TruFor
controlled comparison.

---

## Reviewer 2 (minor revision)

### R2.1 — Recalibrate Sec. 4.4 Pareto-vs-scalar claims

> *The blanket phrasing overstates what is, in fact, a partial and metric-specific
> advantage ... state precisely which metrics were significant and which were not ...
> discuss the practical magnitude of the gains alongside the statistical significance.*

**Response:** We agree, and we have rewritten the passage in Sec. 4.4 to report the
significance and effect sizes precisely rather than as a blanket claim. The revised
text states that, against `by_dice`, Pareto is significantly better (Holm-corrected)
on four of five metrics (PR-AUC p=0.007, Dice p<0.001, balanced accuracy p=0.040,
F1-macro p=0.043), with only the Dice gain reaching a large effect (d=1.00); and
that against `by_prauc` only Dice (p=0.043) and F1-macro (p=0.050, exactly at the
threshold) are significant, while PR-AUC, balanced accuracy and F1-attack are not
(p>0.06). We further note that the remaining effect sizes are small-to-medium
(d≈0.4–0.65) and the absolute gains modest (Δ≲0.02), so the practical benefit is
concentrated in localization quality, the broader value being that distance-to-ideal
selection attains this without committing a priori to a primary objective. The full
five-metric paired statistics are given in Supplementary Table S4.

**Changes:** Sec. 4.4: replaced "Pareto significantly outperforms both scalar
criteria after Holm correction ..." with a metric-specific statement of which
comparisons are significant and which are not, their effect sizes, and their
practical magnitude, with a pointer to Supplementary Table S4.

### R2.2 — Integrate the cross-domain augmentation result (Table S8) into the main text

> *... deserves at least a brief mention in the main text rather than only in the supplement.*

**Response:** We have brought this result into the main text. The Discussion (Sec. 5.2,
Generalization) now states that standard augmentation does not close the cross-domain
gap: augmentation-trained models remain at near-chance zero-shot on SIDTD
(Supplementary Table S8), indicating the gap reflects domain shift rather than limited
input variability.

**Changes:** Sec. 5.2 (Generalization): added a sentence integrating the
augmentation-vs-SIDTD null result (Supplementary Table S8) into the main text.

### R2.3 — Move response-letter-only justifications into the supplementary material

> *computational cost breakdown, full pos_weight sensitivity table ... should be
> incorporated ... into the supplementary material ... for long-term traceability.*

**Response:** Both justifications now live in the supplementary material rather than only
in the review correspondence. The full pos_weight sensitivity table is already present
(Supplementary Sec. S2, Table S2). We have additionally re-added the computational-cost
breakdown as a new supplementary section (Sec. S9, Table S9), reporting the wall-clock
cost per phase (nested HPO ~50 h, outer final models ~1 h, blind-test ablation ~12 h;
~59–63 h total) and the practical deployment cost of 7.7 ± 1.0 min per seed. Long-term
traceability is thus independent of the response letter.

**Changes:** Supplementary: added Sec. S9 "Computational cost breakdown" (Table S9). The
full pos_weight sensitivity table was already present as Table S2.

### R2.3 (abstract) — Acknowledge bounded in-domain scope

> *The manuscript's abstract should more clearly acknowledge the bounded, in-domain
> scope ... given the near-chance zero-shot cross-domain result.*

**Response:** We have added a sentence to the abstract stating that the reported
figures characterize in-domain performance on FantasyID and that zero-shot transfer
to the out-of-domain SIDTD benchmark falls to near chance (PR-AUC 0.555), so the
accuracy should be read as bounded to the training domain.

**Changes:** Abstract: added a closing sentence acknowledging the bounded in-domain
scope and the near-chance zero-shot SIDTD result.
