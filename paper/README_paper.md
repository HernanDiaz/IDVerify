# DocVerify — Paper Build Instructions

Target journal: **IEEE Transactions on Information Forensics and Security (T-IFS)**

## File structure

```
paper/
├── main.tex                  ← master document (IEEE journal format)
├── references.bib            ← bibliography (verify TODOs before submission)
├── setup_figures.sh          ← copies/converts figures into figures/
├── sections/
│   ├── 01_introduction.tex
│   ├── 02_related_work.tex
│   ├── 03_methodology.tex
│   ├── 04_experiments.tex
│   ├── 05_discussion.tex
│   └── 06_conclusion.tex
└── figures/                  ← populated by setup_figures.sh (gitignored)
    ├── fig0_architecture.pdf
    ├── fig1_pareto_front.pdf
    ├── fig2_nested_cv.pdf
    ├── fig3_ablation.pdf
    ├── fig4_challenge.pdf
    └── fig5_scalar_vs_pareto.pdf
```

## Quick start

### 1. Set up figures

```bash
cd paper
bash setup_figures.sh     # Linux/Mac
# On Windows: run manually (see comments in setup_figures.sh)
```

For Windows, copy the PDFs manually:
```
paper_figures/output/fig1_pareto_front.pdf      → paper/figures/
paper_figures/output/fig2_nested_cv.pdf         → paper/figures/
paper_figures/output/fig3_ablation.pdf          → paper/figures/
paper_figures/output/fig4_challenge.pdf         → paper/figures/
paper_figures/output/fig5_scalar_vs_pareto.pdf  → paper/figures/
```

Convert architecture SVG to PDF (Inkscape or https://cloudconvert.com/svg-to-pdf):
```
paper_figures/output/fig0_architecture_compact.svg → paper/figures/fig0_architecture.pdf
```

### 2. Compile

```bash
cd paper
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

Or with `latexmk`:
```bash
latexmk -pdf main.tex
```

## TODOs before submission

### References (references.bib)
- [ ] Fill in `fantasyid2024` — authors, title, venue from https://www.idiap.ch/paper/fantasyid/
- [ ] Fill in `dokforensics2023` — actual EdgeDoc/AG team paper citation
- [ ] Add `TODO_idfraud_report` — a government or industry report on ID document fraud
- [ ] Add `TODO_sidtd` — SIDTD dataset citation (if second dataset is added)
- [ ] Add `TODO_vit` — ViT or similar transformer citation for Discussion section
- [ ] Verify all `TODO:` journal/year/page entries

### Paper content
- [ ] Fill in author names and affiliations in `main.tex`
- [ ] Add funding acknowledgments in `06_conclusion.tex`
- [ ] Add `TODO_idfraud_report` citation in Introduction (line 1 context reference)
- [ ] Add visual examples figure (bonafide + attack + GT mask + predicted mask)
      — use FantasyID/examples/ images plus a model inference pass
- [ ] Double-check all numerical values against final CSV exports

### Optional enhancements for Q1 acceptance
- [ ] Add SIDTD as a second evaluation dataset
- [ ] Add parameter count table for DocVerify model
- [ ] Consider data augmentation ablation (currently no augmentation)
- [ ] Submit to DeepID official test set for a controlled comparison

## Estimated page count (IEEE double-column)

| Section | Est. pages |
|---------|-----------|
| Title + Abstract + Keywords | 0.3 |
| Introduction | 1.5 |
| Related Work | 1.5 |
| Methodology | 3.5 |
| Experiments & Results | 3.5 |
| Discussion | 1.0 |
| Conclusion | 0.5 |
| Figures (6) | 2.5 |
| Tables (5) | 1.5 |
| References | 1.0 |
| **Total** | **~17** |

> Trim if needed: shorten Related Work to 1 page, merge Discussion into Experiments.
> IEEE T-IFS limit is typically 14 pages for the text body (excluding references).
