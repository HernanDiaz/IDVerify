# DocVerify — Contexto del Proyecto
**Última actualización:** 2026-03-21

---

## Descripción General

Sistema de detección y localización de documentos de identidad falsificados mediante una red neuronal multi-tarea (clasificación binaria + segmentación de regiones alteradas).

**Contribución principal:** Framework de optimización multi-objetivo (frente de Pareto / MOTPE) que optimiza simultáneamente PR-AUC (clasificación) y Dice (segmentación), en lugar de una única métrica escalar como hacen la mayoría de trabajos del estado del arte.

---

## Estado Actual del Proyecto (2026-03-21)

### Experimentos completados
- ✅ **Pipeline principal v3** — Nested CV 10×5×50, blind test 30 seeds, scalar experiment
- ✅ **TruFor zero-shot** — Evaluado sobre holdout, métricas en `sota_comparison/trufor_scores.csv`
- ✅ **TruFor fine-tuning** — 30 épocas sobre FantasyID, mejor avg_det_bacc=0.8763 (época 29)
- ✅ **SIDTD** — Testado e implementado pero excluido del paper por malos resultados

### Paper
- ✅ **PRL version** — `paper/prltemplate/`, 7 páginas, elsarticle — **ENVIADO 2026-03-21**
- ✅ **T-IFS version** — `paper/tifs/`, 12 páginas, IEEEtran, compila sin errores (pendiente envío)

### Repositorio
- ✅ Git limpio — `paper/prl/` migrado a `paper/prltemplate/`, resultados CSV subidos, `.gitignore` actualizado

---

## Dataset

- **FantasyID** (~3.284 imágenes, documentos de identidad sintéticos)
  - `bonafide` (genuinos) y `attack` (face swap / text inpainting)
  - Anotaciones JSON con `region_provenance: altered`
  - Split: 85% desarrollo (~2.791 imgs) / 15% holdout (~493 imgs, fijo con random_state=42)
  - Publicado por Idiap Research Institute para el DeepID 2025 Challenge
  - **URL:** https://zenodo.org/records/17063366

- **SIDTD** — Testado pero excluido. Resultados muy inferiores a FantasyID; diferencias de dominio demasiado grandes.

---

## Arquitectura del Modelo

- **Encoder:** Patel CNN (6 bloques convolucionales, 8→16→32→64→128→256 filtros)
- **Decoder:** U-Net con skip connections (5 etapas de upsampling)
- **Cabeza clasificación:** GlobalAvgPool + 4×Dense+Dropout → logit
- **Cabeza segmentación:** Decoder U-Net → Conv 1×1 → logit por píxel
- **Activaciones:** LeakyReLU (α=0.2 fijo) + BatchNorm
- **Salidas:** logits (sigmoid se aplica en la pérdida, no en el forward)

---

## Pipeline ML

### Pérdidas
- Clasificación: `BCEWithLogitsLoss`
- Segmentación: `BCEWithLogitsLoss + DiceLoss`
- Total: `L_cls + λ_mask · L_seg` (λ_mask optimizado por HPO)

### HPO Multi-Objetivo (Optuna)
- Sampler: MOTPE con fallback a NSGA-II
- Objetivos: maximizar (PR-AUC, Dice) simultáneamente
- Selección: `argmin_{p ∈ Pareto} ||(1,1) − p||₂`
- **Espacio de búsqueda:**
  - `lr`: log-uniform [5e-5, 9e-4]
  - `weight_decay`: log-uniform [1e-7, 1e-4]
  - `dropout_rate`: uniform [0.1, 0.4]
  - `dec_ch`: categórico {96, 128, 192, 256}
  - `loss_w_mask`: uniform [0.5, 3.0]

### Nested Cross-Validation
- **10 outer folds × 5 inner folds × 50 trials**
- 500 configuraciones únicas evaluadas
- 2.500 entrenamientos individuales de HPO + 10 modelos finales
- Early stopping: patience=12, ReduceLROnPlateau (factor=0.5, patience=3)

### Test Ciego
- 4 variantes ablación × 30 seeds
- Estadística: Wilcoxon + Holm-Bonferroni + Cohen's d

---

## Resultados Clave

### Nested CV (10 folds):
| Métrica | Valor |
|---|---|
| PR-AUC | **0.9921 ± 0.0058** |
| Dice | **0.856 ± 0.030** |
| w_mask seleccionado | 2.39 ± 0.43 (mayoría entre 2.3–2.7) |

### Blind Test (30 seeds, multitask):
| Métrica | Valor |
|---|---|
| PR-AUC | 0.9967 ± 0.0021 |
| Dice | 0.875 ± 0.018 |
| F1 det (thr=0.5) | 0.969 ± 0.014 |
| F1 loc (per-image) | 0.807 ± 0.096 |

### Ablación (30 seeds):
| Variante | PR-AUC | Dice |
|---|---|---|
| **Multitask (ours)** | **0.9967 ± 0.0021** | **0.875 ± 0.018** |
| cls_only | 0.9503 ± 0.0744 | 0.002 ± 0.008 |
| seg_only | 0.7277 ± 0.0528 | 0.867 ± 0.023 |
| unweighted_losses | 0.9958 ± 0.0016 | 0.857 ± 0.016 |

**Clave:** multitask vs unweighted_losses: Dice p=1.3e-4, d=1.01 (efecto grande).

### Comparativa DeepID 2025 (holdout interno):
| Sistema | F1 det | F1 loc | Datos |
|---|---|---|---|
| Sunlight (1º) | 0.991 | 0.784 | 60K+ externas |
| AG/EdgeDoc (3º) | 0.958 | 0.686 | Solo FantasyID |
| TruFor baseline | 0.807 | 0.590 | 876K preentrenado |
| **DocVerify (ours)** | **0.969 ± 0.014** | **0.807 ± 0.096** | Solo FantasyID |

*Nota: el proyecto se desarrolló después de que cerrara el plazo del challenge; la evaluación es sobre holdout interno, no el test set oficial.*

---

## Stack Tecnológico

- **Framework:** PyTorch 2.10.0+cu130
- **Hardware:** AMD Ryzen 5 3400G (3.70 GHz), 32 GB DDR4-3200, NVIDIA RTX 5060 Ti 16 GB VRAM
- **SO:** Windows 11, CUDA 13.0
- **Python:** 3.11 (venv)
- **HPO:** Optuna 3.x con SQLite backend
- **IDE:** PyCharm / IntelliJ

---

## Estructura de Archivos Relevantes

```
DocVerify/
├── config.py                        — Configuración global (DATASET_ROOT, SIDTD_ROOT, etc.)
├── dataset.py                       — Indexación FantasyID, VRAMCache, DataLoader
├── model.py                         — Arquitectura DocVerify, pérdidas
├── evaluate.py                      — PR-AUC, Dice, BACC, etc.
├── train.py                         — Nested CV, HPO, blind test, estadística
├── main.py                          — Punto de entrada
├── scalar_experiment.py             — Experimento escalarización (baseline)
├── evaluate_challenge_metrics.py    — Métricas DeepID Challenge
├── dataset_sidtd.py                 — Parser SIDTD (excluido del paper)
├── exports_hpo_pareto_nested/       — Resultados del experimento (en repo)
│   ├── nested_outer_results.csv     — Métricas por fold (Tabla 2)
│   ├── optuna_trials_nested.csv     — 2.500 trials HPO
│   ├── optuna_nested_outer.sqlite3  — BD Optuna completa
│   ├── final_blind_test_multiseed.csv — Ablación 30 seeds (Tabla 3)
│   ├── scalar_experiment/           — Comparativa Pareto vs scalar (Tabla 4)
│   ├── challenge_metrics*.csv       — Métricas challenge (Tabla 5)
│   └── stat_tests.csv               — Tests estadísticos
├── paper/
│   ├── tifs/                        — Versión T-IFS (12 páginas)
│   └── prltemplate/                 — Versión PRL (7 páginas) — ENVIADO 2026-03-21
├── paper_figures/                   — Scripts generación figuras
└── sota_comparison/
    ├── trufor_scores.csv            — TruFor zero-shot sobre holdout
    └── holdout_gt.csv               — Ground truth holdout
```

---

## Historial de Experimentos

### v1 — Descartada
Bug sigmoid faltante → umbrales inválidos. Resultados no fiables.

### v2 — Completada (N_OUTER=5, N_FINAL_SEEDS=20)
Bugs corregidos. N_OUTER=5 insuficiente para potencia estadística Q1 (p_min=0.0625).

### v3 — Completada (N_OUTER=10, N_FINAL_SEEDS=30) ← **versión actual**
- Nested CV: PR-AUC=0.9921±0.0058, Dice=0.856±0.030
- Blind test multitask: PR-AUC=0.9967±0.0021, Dice=0.875±0.018
- Tiempo: ~59h en RTX 5060 Ti

---

## Bugs Históricos Corregidos

1. **sigmoid faltante** antes de threshold_sweep → umbral=0.0, predicción trivial
2. **umbral 0.0** en linspace → cambiado a `np.linspace(0.001, 1, 501)`
3. **BCELoss incompatible con AMP** → reemplazado por BCEWithLogitsLoss
4. **torch.compile en Windows** → Triton no disponible, desactivado por defecto
5. **GradScaler deprecado** → API nueva `torch.amp.GradScaler("cuda")`

---

## Orientación hacia Publicación

**Estado:**
- **Pattern Recognition Letters** (Q1) — ✅ **ENVIADO 2026-03-21**
- **IEEE T-IFS** (Q1, IF ~6.8) — Draft listo, pendiente de envío

**Limitaciones conocidas para revisores:**
- Dataset único (SIDTD excluido por malos resultados)
- Arquitectura Patel CNN no es estado del arte (posibles peticiones de comparativa con ViT)
- Evaluación challenge sobre holdout interno, no test set oficial
