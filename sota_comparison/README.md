# Comparación SOTA: DocVerify vs TruFor

Directorio de scripts para comparar DocVerify directamente con TruFor
sobre el mismo holdout de FantasyID (**N=498** imágenes, 141 bonafide + 357 attack).

## Justificación académica

TruFor es el único método con **pesos públicos entrenados por sus autores**,
lo que permite una comparación genuinamente zero-shot sobre nuestro holdout.
EdgeDoc (ICCV 2025) no tiene pesos públicos; se cita en el paper con nota
de incomparabilidad directa.

---

## Prerequisitos

1. **Docker Desktop** instalado y corriendo
   - Descarga: https://www.docker.com/products/docker-desktop/
   - Para GPU: asegúrate de que NVIDIA Container Toolkit está configurado

2. **Python del proyecto** (venv activo)

---

## Flujo de ejecución

El pipeline está dividido en pasos cortos y pasos largos (Docker).
Los pasos marcados con ⚡ se ejecutan desde **PyCharm Run Configurations**.
Los marcados con 🖱️ son **batch files de doble clic**.

### ✅ PASO 1 — Exportar holdout (completado)

> **Ya ejecutado.** El holdout está en `holdout_images/` y `holdout_gt.csv`.

Para re-ejecutar si fuera necesario:
```
PyCharm → Run Config: sota_00_export_holdout
```

---

### ⏳ PASO 2a — Build Docker de TruFor (~15-25 min)

> **Ya iniciado en background** (PID 48292).
> Monitoriza en **Docker Desktop** — cuando la imagen `trufor_docverify` aparezca, continúa.

Para relanzar si falló:
```
🖱️  STEP2a_build_docker.bat
```

---

### PASO 2b — Inferencia TruFor sobre el holdout (~5-10 min GPU)

Una vez que `trufor_docverify` aparezca en Docker Desktop:
```
🖱️  STEP2b_run_inference.bat
```

Genera `trufor_output/` con un `.npz` por imagen.

**Si no tienes GPU:** edita el .bat y elimina la línea `--gpus all`.

---

### PASO 2c — Parsear output .npz → CSV (⚡ rápido, <1 min)

```
PyCharm → Run Config: sota_01_parse_output
```

Genera `trufor_scores.csv` con columnas `stem, trufor_score, map_npy_path`.

---

### PASO 3 — Calcular métricas y tabla LaTeX (⚡ rápido, <1 min)

```
PyCharm → Run Config: sota_02_eval_comparison
```

Genera:
- `trufor_metrics.csv` — métricas TruFor detalladas
- `comparison_table.csv` — tabla comparativa legible
- `comparison_table.tex` — tabla LaTeX lista para el paper

---

## Estructura de outputs de TruFor

Cada imagen genera un `.npz` en `trufor_output/` con:
| Campo    | Tipo       | Descripción                              |
|----------|------------|------------------------------------------|
| `score`  | float [0,1]| Score de detección (1=forjado)           |
| `map`    | array H×W  | Mapa de localización de anomalías        |
| `conf`   | array H×W  | Mapa de confianza (fiabilidad del mapa)  |
| `imgsize`| tuple      | Tamaño original de la imagen             |

---

## Troubleshooting

### Docker: "No GPU available"
```bat
REM En STEP2b_run_inference.bat, eliminar la línea:
--gpus all
```

### Docker: Error de rutas en Windows
Docker Desktop en Windows con WSL2 usa rutas `/c/...` no `C:\...`.
El script convierte automáticamente. Si falla, comprueba que en Docker Desktop
→ Settings → Resources → WSL Integration está activado.

### TruFor: error de módulo mmcv
El Dockerfile instala mmcv-full 1.5.3 con wheels pre-compilados para
PyTorch 1.11 + CUDA 11.3. Si la descarga falla por timeout, relanza
`STEP2a_build_docker.bat` (Docker usa caché de las capas ya descargadas).

### Scores invertidos (attacks con score bajo)
`02_eval_comparison.py` detecta automáticamente si los scores están
invertidos y los corrige. Verás el mensaje:
`[NOTA] Los scores de TruFor parecen INVERTIDOS — Invirtiendo scores`

---

## Archivos del directorio

```
sota_comparison/
│
│  ── SCRIPTS ──────────────────────────────────────────────
├── 00_export_holdout.py       ← Paso 1 (completado ✓)
├── launch_docker_build.py     ← Lanza docker build en background
├── 01_run_trufor.py           ← Parsea output .npz → trufor_scores.csv
├── 02_eval_comparison.py      ← Calcula métricas + tabla LaTeX
├── Dockerfile                 ← Imagen Docker para TruFor
│
│  ── BATCH FILES ───────────────────────────────────────────
├── STEP2a_build_docker.bat    ← Build Docker (~15-25 min)
├── STEP2b_run_inference.bat   ← Inferencia Docker (~5-10 min)
├── RUN_COMPARISON.bat         ← Pipeline completo (todo en uno)
│
│  ── GENERADOS POR PASO 1 ──────────────────────────────────
├── holdout_gt.csv             ← Ground truth 498 imágenes ✓
├── holdout_images/            ← Imágenes copiadas ✓
│   ├── bonafide/ (141 imágenes)
│   └── attack/  (357 imágenes)
├── holdout_masks/             ← Máscaras GT .npy ✓
│
│  ── GENERADOS POR PASO 2 ──────────────────────────────────
├── trufor_repo/               ← Código TruFor (clonado) ✓
├── trufor_weights/            ← Pesos trufor.pth.tar (descargado) ✓
├── trufor_output/             ← .npz por imagen (pendiente)
├── trufor_scores.csv          ← Scores consolidados (pendiente)
│
│  ── GENERADOS POR PASO 3 ──────────────────────────────────
├── trufor_metrics.csv         ← Métricas TruFor (pendiente)
├── comparison_table.csv       ← Tabla comparativa (pendiente)
└── comparison_table.tex       ← LaTeX para paper (pendiente)
```
