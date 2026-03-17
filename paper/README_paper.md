# DocVerify — Paper

**Author:** Hernán Díaz Rodríguez — Department of Computer Science, University of Oviedo
**Funding:** Instituto Nacional de Ciberseguridad (INCIBE) / Cátedra INCIBE de Ciberseguridad e IA

---

## Versiones

### `tifs/` — IEEE Transactions on Information Forensics and Security
- **Formato:** IEEEtran journal, doble columna
- **Límite:** 14 páginas de texto + referencias
- **Estado:** Draft v1 — 12 páginas, compila sin errores
- **Citar como:** DocVerify (versión extendida)

### `prl/` — Pattern Recognition Letters (Elsevier)
- **Formato:** elsarticle 5p, doble columna
- **Límite:** 7 páginas incluyendo referencias
- **Estado:** Draft v1 — 7 páginas, compila sin errores
- **Citar como:** DocVerify (versión compacta)

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
└── prl/                   ← versión Pattern Recognition Letters
    ├── main.tex
    ├── highlights.tex     ← Research Highlights (fichero separado, requisito PRL)
    ├── references.bib
    ├── elsarticle.cls     ← plantilla oficial Elsevier
    ├── prletters.sty      ← estilo PRL
    ├── model2-names.bst   ← bibliografía autor-año
    ├── sections/
    │   ├── 01_introduction.tex
    │   ├── 02_related_work.tex  (condensado ~50%)
    │   ├── 03_methodology.tex
    │   ├── 04_experiments.tex   (sin Sec. 4.4 Pareto vs Scalar)
    │   ├── 05_discussion.tex
    │   └── 06_conclusion.tex
    └── figures/           ← mismas figuras que tifs/ (sin fig5)
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
| Research Highlights | No requerido | `highlights.tex` (requisito PRL) |
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
cd paper/prl
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

| Figura | Script | Usado en |
|---|---|---|
| `fig0_architecture_compact.pdf` | `architecture_svg_compact.py` | T-IFS + PRL |
| `fig1_pareto_front.pdf` | `pareto_front.py` | T-IFS + PRL |
| `fig2_nested_cv.pdf` | `nested_cv.py` | T-IFS + PRL |
| `fig3_ablation.pdf` | `ablation.py` | T-IFS + PRL |
| `fig4_challenge.pdf` | `challenge.py` | T-IFS + PRL |
| `fig5_scalar_vs_pareto.pdf` | `scalar_vs_pareto.py` | T-IFS solo |

---

## Estado de referencias

| Clave | Estado |
|---|---|
| `fantasyid2024` | ✅ Korshunov et al., IJCB 2025, arXiv:2507.20808 |
| `deepid2025challenge` | ✅ Korshunov et al., ICCVW 2025, pp. 510–519 |
| `trufor2023` | ✅ Guillaro et al., CVPR 2023 |
| `TODO_idfraud_report` | ✅ Europol "Facing Reality? Deepfakes", 2022 |
| `TODO_euaiact` | ✅ EU AI Act, Reg. 2024/1689 |
| `TODO_sidtd` | ✅ Boned et al., Scientific Data 11:1356, 2024 |
| `TODO_vit` | ✅ Dosovitskiy et al., ICLR 2021 |
| `dokforensics2023` | ⚠️ AG/EdgeDoc — pendiente de publicación independiente |

> Los prefijos `TODO_` se conservan por trazabilidad; todos tienen datos reales.

---

## TODOs antes de envío

### T-IFS
- [ ] Añadir tabla comparativa TruFor fine-tuneado vs DocVerify (pendiente de inferencia)
- [ ] Subir pesos del modelo a Zenodo → añadir DOI
- [ ] Verificar valores numéricos contra `exports_hpo_pareto_nested/*.csv`
- [ ] Rellenar fecha "Manuscript received" en `main.tex`
- [ ] Ejecutar IEEE PDF checker / IEEE Xplore compliance tool

### PRL
- [ ] Añadir tabla comparativa TruFor fine-tuneado vs DocVerify
- [ ] Rellenar `\received{}`, `\finalform{}`, `\accepted{}` en frontmatter
- [ ] Rellenar CPU/RAM TODO en Sec. 4.1 ← **ya hecho** (AMD Ryzen 5 3400G, 32 GB DDR4-3200)
- [ ] Verificar con la plantilla oficial que el layout de primera página es correcto

### Ambas versiones
- [ ] DeepID Challenge: contactar organizadores para evaluación en test set oficial
- [ ] Considerar añadir SIDTD como dataset secundario (actualmente excluido por malos resultados)
