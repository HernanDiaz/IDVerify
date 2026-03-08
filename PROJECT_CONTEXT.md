# DocVerify — Contexto del Proyecto

## Descripción General

Sistema de detección de documentos de identidad falsificados mediante una red neuronal
multi-tarea (clasificación binaria + segmentación de regiones alteradas).

**Contribución principal:** Framework de optimización multi-objetivo (frente de Pareto)
que optimiza simultáneamente PR-AUC (clasificación) y Dice (segmentación), en lugar de
una única métrica como hacen la mayoría de trabajos del estado del arte.

---

## Dataset

- **Nombre:** FantasyID (dataset local, no en Drive)
- **Tamaño:** ~3.284 imágenes con anotaciones JSON
- **Clases:** bonafide / attack
- **Estructura:**
  ```
  FantasyID/
    train/
      bonafide/<device>/<stem>.jpg + <stem>.json
      attack/<attack_type>/<device>/<stem>.jpg + <stem>.json
    test/
      (misma estructura)
  ```
- **Anotaciones:** JSON con regiones rectangulares, campo `region_provenance: altered`
  indica las zonas falsificadas que forman la máscara de segmentación.

---

## Arquitectura del Modelo

- **Encoder:** Patel CNN (6 bloques convolucionales, 8→16→32→64→128→256 filtros)
- **Decoder:** U-Net con skip connections (5 bloques de upsampling)
- **Cabeza clasificación:** GlobalAvgPool + 4 capas Dense con Dropout → logit
- **Cabeza segmentación:** Decoder U-Net → Conv 1×1 → logit
- **Activaciones:** LeakyReLU (alpha=0.2 fijo) + BatchNorm
- **Salidas:** logits (sigmoid se aplica en la pérdida, no en el forward)

---

## Pipeline ML

### Pérdidas
- Clasificación: `BCEWithLogitsLoss`
- Segmentación: `BCEWithLogitsLoss + DiceLoss` (ambas sobre logits)

### HPO Multi-Objetivo (Optuna) — v2
- Sampler: MOTPE con fallback a NSGA-II
- Objetivos: maximizar (PR-AUC, Dice) simultáneamente
- Selección del mejor trial: mínima distancia euclídea al punto ideal (1, 1)
- Frente de Pareto exportado a CSV para análisis
- **Espacio de búsqueda (4 dimensiones efectivas):**
  - `lr`: log-uniform [5e-5, 9e-4]
  - `weight_decay`: log-uniform [1e-7, 1e-4]
  - `dropout_rate`: uniform [0.1, 0.4]
  - `dec_ch`: categórico [96, 128, 192, 256]
  - `loss_w_mask`: uniform [0.5, 3.0]
  - `alpha`: fijo en 0.2 (eliminado del HPO — sin señal en 250 trials)

### Validación
- **Nested CV:** 5 folds externos × 5 folds internos × 50 trials = 1.250 entrenamientos HPO
- **Early stopping:** patience=12, monitor=distancia al punto ideal
- **Scheduler:** ReduceLROnPlateau (factor=0.5, patience=3)
- **Gradient clipping:** 1.0

### Test Ciego
- 4 variantes de ablación: multitask, cls_only, seg_only, unweighted_losses
- 5 seeds por variante = 20 entrenamientos sobre holdout nunca visto

### Estadística
- Test de Wilcoxon pareado + corrección Holm-Bonferroni
- t-test pareado como referencia
- Cohen's d para tamaño del efecto

---

## Stack Tecnológico

- **Framework:** PyTorch 2.10.0+cu130
- **GPU:** NVIDIA RTX 5060 Ti 16GB VRAM, CUDA 13.0
- **Python:** 3.11 (venv_torch)
- **HPO:** Optuna 3.x con SQLite backend
- **IDE:** PyCharm

### Migración relevante
El proyecto comenzó en TensorFlow/Keras (Google Colab) y fue migrado a PyTorch
por incompatibilidad de TensorFlow con CUDA 13.0 / arquitectura Blackwell.

---

## Optimizaciones de Rendimiento Implementadas

1. **Cache VRAM por outer fold:** Las imágenes se cargan en VRAM una sola vez al
   inicio de cada fold externo. Los inner folds hacen slices por índice sobre los
   tensores ya en GPU → sin transferencia CPU→GPU durante el HPO.
   - Antes: 12 min/trial (carga en cada trial)
   - Después: ~3.5 min/trial

2. **Precisión mixta (AMP):** `torch.autocast` con float16 en operaciones GPU.
   Compatible con `BCEWithLogitsLoss` (se evitó `BCELoss` que es unsafe con AMP).

3. **GradScaler:** `torch.amp.GradScaler("cuda")` (API nueva, no la deprecada).

4. **torch.compile:** Desactivado en Windows por falta de soporte Triton.
   Activar en Linux con `USE_COMPILE=1`.

5. **Barra de progreso:** tqdm por epoch mostrando loss, PR-AUC, Dice, patience.

6. **Guardado de modelos:** Cada fold externo y cada variante del test ciego
   guarda su `.pt` en `exports_hpo_pareto_nested/models/`.

---

## Bugs Corregidos

1. **sigmoid faltante antes de threshold_sweep** — los logits crudos se pasaban
   directamente al barrido de umbrales [0,1], causando umbral=0.0 y predicción
   trivial (todo ataque) en 3/5 folds del nested CV y 3/5 seeds del blind test.
   Fix: `torch.sigmoid()` aplicado antes de `threshold_sweep` en los dos sitios
   donde se llama (`run_nested_cv` y `_train_final_model`).

2. **umbral 0.0 en threshold_sweep** — `np.linspace(0, 1, 501)` incluía 0.0,
   que predice todo positivo y puede ser elegido espuriamente como óptimo.
   Fix: cambiado a `np.linspace(0.001, 1, 501)`.

3. **BCELoss incompatible con AMP** — reemplazado por `BCEWithLogitsLoss` en
   train.py y `binary_cross_entropy_with_logits` en model.py. Los sigmoids
   de las salidas del modelo fueron eliminados.

4. **torch.compile falla en Windows** — Triton no disponible. `USE_COMPILE`
   desactivado por defecto. Activar en Linux con variable de entorno.

5. **GradScaler deprecado** — `torch.cuda.amp.GradScaler` reemplazado por
   `torch.amp.GradScaler("cuda")`.

---

## Estructura de Archivos

```
DocVerify/
├── config.py          — Configuración global (rutas, hiperparámetros, flags)
├── dataset.py         — Indexación, parsing JSON, VRAMCache, DataLoader
├── model.py           — Arquitectura, pérdidas (logits), factory
├── evaluate.py        — Métricas completas (PR-AUC, Dice, mIoU, etc.)
├── train.py           — Nested CV, HPO, early stopping, test ciego, estadística
├── main.py            — Punto de entrada
├── requirements.txt   — Dependencias (torch cu124, optuna, sklearn, etc.)
├── .gitignore         — Excluye FantasyID/, venv_torch/, exports/, *.pt
├── PROJECT_CONTEXT.md — Este archivo
└── exports_hpo_pareto_nested/
    ├── optuna_nested_outer.sqlite3       — Base de datos Optuna (DB Browser para ver)
    ├── optuna_trials_nested.csv          — Todos los trials HPO
    ├── pareto_front_trials.csv           — Solo trials en frente de Pareto
    ├── nested_outer_results.csv          — Métricas outer test por fold
    ├── final_blind_test_multiseed.csv    — Métricas test ciego por variante/seed
    ├── stat_tests.csv                    — Wilcoxon + Holm + Cohen's d
    └── models/                           — Modelos .pt guardados
```

---

## CSVs Generados (resumen)

| CSV | Cuándo se escribe | Para qué sirve |
|-----|-------------------|----------------|
| `optuna_trials_nested.csv` | Durante HPO | Registro de todos los trials |
| `pareto_front_trials.csv` | Durante HPO | Solo trials no dominados |
| `nested_outer_results.csv` | Tras cada outer fold | Métricas de generalización |
| `final_blind_test_multiseed.csv` | Test ciego | Rendimiento final por variante |
| `stat_tests.csv` | Al final | Significancia estadística |

---

## Configuración de Producción (v2 — ejecución actual)

```python
N_OUTER              = 5
N_INNER              = 5
N_TRIALS             = 50
MAX_EPOCHS_TRIAL     = 15     # v1: 10
MAX_EPOCHS_FINAL     = 100
MAX_EPOCHS_ABLATION  = 50
BATCH_SIZE           = 64
PATCH_SIZE           = 224
USE_AMP              = True
USE_COMPILE          = False  # Windows no soporta Triton
GRAD_CLIP            = 1.0
N_FINAL_SEEDS        = 20
patience_final       = 12     # v1: 8
```

**Tiempo real:** ~24 horas en RTX 5060 Ti 16GB.

---

## Configuración de Producción (v3 — completada)

```python
N_OUTER              = 10     # v2: 5 — necesario para potencia estadística Q1
N_INNER              = 5
N_TRIALS             = 50
MAX_EPOCHS_TRIAL     = 15
MAX_EPOCHS_FINAL     = 100
MAX_EPOCHS_ABLATION  = 50
BATCH_SIZE           = 64     # subir a 256 si se usa A100/H100
PATCH_SIZE           = 224
USE_AMP              = True
USE_COMPILE          = False  # activar con USE_COMPILE=1 en Linux
GRAD_CLIP            = 1.0
N_FINAL_SEEDS        = 30     # v2: 20
patience_final       = 12
```

**Tiempo real:** ~59h en RTX 5060 Ti 16GB (Windows).

---

## Historial de Ejecuciones

### v1 (primera ejecución — descartada)
- MAX_EPOCHS_TRIAL=10, patience=8
- Espacio HPO: 6 dimensiones (incluía alpha y dropout amplio)
- Bug: sigmoid faltante → umbrales inválidos en 3/5 folds
- Resultados válidos: PR-AUC=0.990±0.005, Dice=0.830±0.023 (métricas de umbral no fiables)
- Observaciones clave:
  - Trade-off PR-AUC vs Dice confirmado empíricamente
  - lr<5e-5 produce Dice≈0 en 10 epochs → eliminado del rango
  - alpha LeakyReLU sin señal → fijado en 0.2
  - dropout sin señal fuera de [0.1, 0.4]
  - dec_ch=192 dominante en Pareto → añadido 256

### v2 (completada — N_OUTER=5, N_FINAL_SEEDS=20)
- Todos los bugs corregidos
- Espacio HPO refinado a 4 dimensiones efectivas
- MAX_EPOCHS_TRIAL=15, patience=12
- Nested CV: PR-AUC=0.9904±0.0013, Dice=0.829±0.017
- Blind test (20 seeds): multitask PR-AUC=0.9970±0.0009, Dice=0.868±0.006
- Scalar experiment: completado, diferencias no significativas (n=5 insuficiente)
- Limitación: N_OUTER=5 → Wilcoxon p mínimo=0.0625, potencia estadística insuficiente para Q1

### v3 (completada — N_OUTER=10, N_FINAL_SEEDS=30)
- N_OUTER duplicado para alcanzar potencia estadística Q1 (Wilcoxon p mínimo=0.002)
- N_FINAL_SEEDS: 20→30
- Exports en: `exports_hpo_pareto_nested/`

**Nested CV (10 folds):**

| Fold | PR-AUC | Dice | ROC-AUC | BACC | w_mask sel. |
|------|--------|------|---------|------|-------------|
| 1 | 0.9939 | 0.861 | 0.9848 | 0.9491 | 2.46 |
| 2 | 0.9769 | 0.871 | 0.9615 | 0.9319 | 2.70 |
| 3 | 0.9993 | 0.909 | 0.9981 | 0.9470 | 2.47 |
| 4 | 0.9915 | 0.858 | 0.9773 | 0.8986 | 2.43 |
| 5 | 0.9948 | 0.827 | 0.9869 | 0.9318 | 1.41 |
| 6 | 0.9936 | 0.874 | 0.9830 | 0.9319 | 2.89 |
| 7 | 0.9940 | 0.861 | 0.9834 | 0.9479 | 2.67 |
| 8 | 0.9967 | 0.887 | 0.9911 | 0.9557 | 2.29 |
| 9 | 0.9896 | 0.807 | 0.9722 | 0.9023 | 1.95 |
| 10 | 0.9909 | 0.800 | 0.9765 | 0.9294 | 2.65 |
| **Media** | **0.9921±0.0058** | **0.856±0.030** | **0.9815±0.0110** | **0.9326±0.0192** | **2.39±0.43** |

Observación: Pareto selecciona consistentemente w_mask en rango 2.3–2.7.
Folds 5 y 9 (w_mask=1.41, 1.95) coinciden con los peores Dice.

**Blind test (30 seeds, 4 variantes):**

| Variante | PR-AUC | Dice |
|----------|--------|------|
| multitask | 0.9967±0.0021 | 0.875±0.018 |
| cls_only | 0.9503±0.0744 | 0.002±0.008 |
| seg_only | 0.7277±0.0528 | 0.867±0.023 |
| unweighted_losses | 0.9958±0.0016 | 0.857±0.016 |

**Tests estadísticos (Wilcoxon + Holm, n=30):**

| Comparación | Métrica | p_holm | Significativo | Cohen's d |
|-------------|---------|--------|---------------|-----------|
| multitask vs cls_only | PR-AUC | 1.5e-08 | ✅ | — |
| multitask vs cls_only | Dice | 1.3e-08 | ✅ | — |
| multitask vs seg_only | PR-AUC | 1.5e-08 | ✅ | — |
| multitask vs seg_only | Dice | 0.288 | ❌ (esperado) | — |
| multitask vs unweighted | Dice | 1.3e-04 | ✅ | d=1.01 (grande) |
| multitask vs unweighted | PR-AUC | 0.083 | ❌ (esperado) | — |
| multitask vs unweighted | BACC | 0.121 | ❌ (esperado) | — |

Resultado clave: multitask vs unweighted_losses significativo en Dice (p=1.3e-4, d=1.01).
Valida la contribución del HPO multi-objetivo. PR-AUC no significativo — correcto,
ambas variantes clasifican bien.

**Scalar experiment (10 folds, grid loss_w_mask ∈ {0.5, 1.0, 1.5, 2.0, 2.5, 3.0}):**

Grid completo (media ± std, n=10):

| w_mask | PR-AUC | Dice |
|--------|--------|------|
| 0.5 | 0.9911±0.0030 | 0.8007±0.0183 |
| 1.0 | 0.9938±0.0023 | 0.8289±0.0353 |
| 1.5 | 0.9888±0.0084 | 0.8237±0.0465 |
| 2.0 | 0.9943±0.0027 | 0.8498±0.0241 |
| 2.5 | 0.9926±0.0059 | 0.8480±0.0194 |
| 3.0 | 0.9923±0.0035 | 0.8552±0.0227 |

Selección por criterio (media ± std, n=10 folds):

| Método | PR-AUC | Dice | Dice std |
|--------|--------|------|----------|
| by_prauc | 0.9959±0.0015 | 0.8574±0.0166 | 0.0166 |
| by_dice | 0.9936±0.0033 | 0.8630±0.0147 | 0.0147 |
| multiobjective (Pareto) | 0.9921±0.0060 | 0.8555±0.0345 | 0.0345 |

Tests estadísticos (Wilcoxon + Holm, n=10): ninguna comparación significativa.
El Pareto tiene mayor std en Dice (0.0345 vs 0.0166) porque explora el espacio
continuo y en folds 5/9 selecciona w_mask bajos (1.41, 1.95); los escalares
colapsan hacia w_mask altos (by_dice: μ=2.60, 6/10 folds eligen w=3.0).

Argumento para el paper: la ventaja del Pareto no es superar estadísticamente
a los escalares (n=10 insuficiente para detectar diferencias pequeñas), sino
(a) evitar la decisión arbitraria del criterio mono-objetivo, (b) búsqueda
continua en lugar de grid discreto, y (c) la diferencia significativa ya
establecida frente a unweighted_losses en el blind test (p=1.3e-4, d=1.01).

---

## Orientación hacia Publicación

**Target realista:** IEEE TIFS / Pattern Recognition (Q1) si se añade SIDTD como
segundo dataset. Workshop CVPR/ECCV/ICCV como opción más rápida con solo FantasyID.

**Narrativa del paper:**
"Framework de optimización multi-objetivo para sistemas de verificación documental
que explicita el trade-off entre clasificación y localización de alteraciones,
permitiendo selección del punto de operación según el contexto de despliegue."

**Contexto competitivo — DeepID Challenge (ICCV 2025):**
- Nuestro F1 de clase 1 (~0.93–0.96 según fold) comparable con 3er puesto del
  challenge (AG/EdgeDoc: F1=0.958) entrenando solo con FantasyID.
- Sunlight (1º) usa 60K+ imágenes externas + 2 etapas de preentrenamiento.
- 6/7 equipos que superan el baseline dependen de TruFor preentrenado en 876K imgs.
- Nuestro sistema es el único con HPO multi-objetivo explícito y arquitectura propia.
- Métricas del challenge calculables sin reentrenar (F1 a umbral 0.5 fijo +
  F1 per-image de localización): pendiente de implementar en evaluate.py.
- Dataset privado de PXL Vision (20K IDs reales): pendiente solicitar a Idiap
  para comparación cross-domain.

**Limitaciones conocidas:**
- Dataset único hasta SIDTD — revistas Q1 exigen ≥2 datasets
- Arquitectura Patel CNN no es estado del arte (revisores pedirán comparativa con ViT)
- 50 trials de Optuna razonable pero no exhaustivo

**Elementos completados para el paper:**
- ✅ Análisis visual del frente de Pareto (fig1_pareto_front.png)
- ✅ Comparativa con escalarización clásica — scalar experiment v3
- ✅ Análisis de consistencia de w_mask por fold (fig6_wmask_per_fold.png)
- ✅ Comparativa 4 variantes ablación + tests estadísticos (fig2, fig4)
- ✅ Trade-off clasificación/segmentación (fig3_scalar_tradeoff.png)
- ✅ Comparativa con baselines del paper FantasyID (fig5_baseline_comparison.png)

**Elementos pendientes para el paper:**
- ⏳ SIDTD dataset (segundo dataset para Q1) — implementación parser pendiente
- ⏳ Métricas del challenge (F1@0.5, F1 per-image localización) — 2 funciones en evaluate.py
- ⏳ Solicitud dataset privado PXL Vision a equipo Idiap
- ⏳ Redacción paper

---

## Mejoras Pendientes Identificadas (no implementadas)

- Data augmentation (flips, rotaciones, variaciones de brillo) — solo en train
- Cosine annealing con warm restarts en lugar de ReduceLROnPlateau
- torch.compile en Linux (20-40% adicional de velocidad)
