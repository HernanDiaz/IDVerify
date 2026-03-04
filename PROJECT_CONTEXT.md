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
N_FINAL_SEEDS        = 5
patience_final       = 12     # v1: 8
```

**Tiempo estimado:** ~24 horas en RTX 5060 Ti 16GB.

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

### v2 (ejecución actual)
- Todos los bugs corregidos
- Espacio HPO refinado a 4 dimensiones efectivas
- MAX_EPOCHS_TRIAL=15, patience=12
- Pendiente: resultados

---

## Orientación hacia Publicación

**Target realista:** Workshop CVPR/ECCV/ICCV especializado en biometría/seguridad
documental, o IEEE TIFS / Pattern Recognition si se añade un segundo dataset.

**Narrativa del paper:**
"Framework de optimización multi-objetivo para sistemas de verificación documental
que explicita el trade-off entre clasificación y localización de alteraciones,
permitiendo selección del punto de operación según el contexto de despliegue."

**Limitaciones conocidas:**
- Dataset único (FantasyID) — revistas primer nivel exigen ≥2 datasets
- Arquitectura Patel CNN no es estado del arte (revisores pedirán comparativa con ViT)
- 50 trials de Optuna razonable pero no exhaustivo

**Elementos pendientes para el paper:**
- Análisis visual del frente de Pareto
- Comparativa con escalarización clásica (barrido manual de loss_w_mask)
- Análisis de patrones de hiperparámetros en el frente de Pareto

---

## Mejoras Pendientes Identificadas (no implementadas)

- Data augmentation (flips, rotaciones, variaciones de brillo) — solo en train
- Cosine annealing con warm restarts en lugar de ReduceLROnPlateau
- torch.compile en Linux (20-40% adicional de velocidad)
