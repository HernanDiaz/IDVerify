# DocVerify — Paper

**Author:** Hernán Díaz Rodríguez — Department of Computer Science, University of Oviedo
**Funding:** Instituto Nacional de Ciberseguridad (INCIBE) / Cátedra INCIBE de Ciberseguridad e IA

---

## Versiones

### `prltemplate/` — Pattern Recognition Letters (Elsevier) — ENVIADO
- **Formato:** elsarticle 5p, doble columna
- **Límite:** 7 páginas incluyendo referencias
- **Estado:** ✅ **Enviado — 2026-03-21**
- **Citar como:** DocVerify (versión compacta)

### `tifs/` — IEEE Transactions on Information Forensics and Security
- **Formato:** IEEEtran journal, doble columna
- **Límite:** 14 páginas de texto + referencias
- **Estado:** Draft v1 — 12 páginas, compila sin errores
- **Citar como:** DocVerify (versión extendida)

---

## Estructura de carpetas

```
paper/
├── README_paper.md        ← este archivo
├── tifs/                  ← versión IEEE T-IFS
│   ├── main.tex
│   ├── references.bib
│   ├── compile.ps1
│   ├── sections/
│   │   ├── 01_introduction.tex
│   │   ├── 02_related_work.tex
│   │   ├── 03_methodology.tex
│   │   ├── 04_experiments.tex
│   │   ├── 05_discussion.tex
│   │   └── 06_conclusion.tex
│   └── figures/
│       ├── fig0_architecture_compact.pdf
│       ├── fig1_pareto_front.pdf
│       ├── fig2_nested_cv.pdf
│       ├── fig3_ablation.pdf
│       ├── fig4_challenge.pdf
│       ├── fig5_scalar_vs_pareto.pdf
│       └── fig_qual_*.png   (ejemplos cualitativos)
└── prltemplate/           ← versión Pattern Recognition Letters — ENVIADO
    ├── main.tex
    ├── references.bib
    ├── main.bbl           ← bibliografía compilada (incluida para reproducibilidad)
    ├── elsarticle.cls     ← plantilla oficial Elsevier
    ├── prletters.sty      ← estilo PRL
    ├── sections/
    │   ├── 01_introduction.tex
    │   ├── 02_related_work.tex  (condensado ~50%)
    │   ├── 03_methodology.tex
    │   ├── 04_experiments.tex   (sin Sec. 4.4 Pareto vs Scalar)
    │   ├── 05_discussion.tex
    │   └── 06_conclusion.tex
    └── figures/           ← figuras usadas en el paper (PDF + PNG)
        ├── fig0_architecture_compact.pdf / .png
        ├── fig1_pareto_front.pdf / .png
        └── fig_qual_attack_*.png
```

---

## Diferencias entre versiones

| Elemento | T-IFS | PRL |
|---|---|---|
| Formato | IEEEtran | elsarticle 5p |
| Páginas | 12 | 7 |
| Citas | `\cite{}` numérico | `\citep{}`/`\citet{}` autor-año |
| Sec. 4.4 Pareto vs Scalar | ✅ incluida | ❌ eliminada |
| Figura violines ablation | `figure*` | `figure*` |
| Análisis cualitativo | 2 ejemplos (bonafide + attack) | 1 ejemplo (solo attack) |
| Tabla NCV | con ROC-AUC | sin ROC-AUC |
| Related Work | ~1.5 páginas | ~0.7 páginas |

---

## Compilar

### T-IFS (Windows/MiKTeX)
```powershell
cd paper/tifs
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
# → main.pdf (12 páginas)
```

### PRL (Windows/MiKTeX)
```powershell
cd paper/prltemplate
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
# → main.pdf (7 páginas)
```

---

## Regenerar figuras

Las figuras se generan desde `paper_figures/`:
```powershell
cd E:\PycharmProjects\DocVerify
venv\Scripts\python.exe paper_figures\generate_all.py
```

Para exportar versiones de alta resolución (EPS/TIFF para la revista):
```powershell
venv\Scripts\python.exe paper_figures\export_hires.py
# → paper_figures/output/hires/
```

| Figura | Script | Usado en |
|---|---|---|
| `fig0_architecture_compact.pdf` | `architecture_svg_compact.py` | T-IFS + PRL |
| `fig1_pareto_front.pdf` | `pareto_front.py` | T-IFS + PRL |
| `fig2_nested_cv.pdf` | `nested_cv.py` | T-IFS + PRL |
| `fig3_ablation.pdf` | `ablation.py` | T-IFS + PRL |
| `fig4_challenge.pdf` | `challenge.py` | T-IFS + PRL |
| `fig5_scalar_vs_pareto.pdf` | `scalar_vs_pareto.py` | T-IFS solo |
