"""
config.py — Configuración global del proyecto DocVerify.
Modifica este archivo para ajustar rutas, hiperparámetros y comportamiento del experimento.
"""

import os
from pathlib import Path

# ============================================================
# RUTAS DEL DATASET
# ============================================================

# Directorio raíz donde está descomprimido FantasyID.
# Puede ser una ruta absoluta o relativa al directorio de ejecución.
# Ejemplos:
#   DATASET_ROOT = Path("./FantasyID")
#   DATASET_ROOT = Path("C:/Datasets/FantasyID")
#   DATASET_ROOT = Path("/home/user/data/FantasyID")
DATASET_ROOT = Path(os.getenv("DATASET_ROOT", "./FantasyID"))

# ============================================================
# DIRECTORIO DE EXPORTACIÓN DE RESULTADOS
# ============================================================
EXPORT_DIR = Path(os.getenv("EXPORT_DIR", "./exports_hpo_pareto_nested"))

# ============================================================
# IMAGEN Y MODELO
# ============================================================
PATCH_SIZE = 224          # Resolución de entrada al modelo (alto y ancho)
IMG_EXTS   = {".jpg"}     # Extensiones de imagen aceptadas

# ============================================================
# DATALOADER (PyTorch)
# ============================================================
# Con precarga en RAM, los workers no son necesarios.
NUM_WORKERS = int(os.getenv("NUM_WORKERS", "0"))

# pin_memory acelera la transferencia CPU→GPU. Activar solo con GPU.
PIN_MEMORY  = (os.getenv("PIN_MEMORY", "1") == "1")

# persistent_workers mantiene los procesos de carga vivos entre epochs.
# Solo tiene efecto si NUM_WORKERS > 0.
PERSISTENT_WORKERS = NUM_WORKERS > 0

# ============================================================
# ACELERACIÓN GPU
# ============================================================

# Precisión mixta float16/float32 (duplica throughput en GPU moderna).
USE_AMP = (os.getenv("USE_AMP", "1") == "1")

# torch.compile: no disponible en Windows (requiere Triton).
# Activar solo en Linux/Mac.
USE_COMPILE = (os.getenv("USE_COMPILE", "0") == "1")

# Gradient clipping: evita explosión de gradientes.
GRAD_CLIP = float(os.getenv("GRAD_CLIP", "1.0"))

# ============================================================
# MODELOS GUARDADOS
# ============================================================
MODELS_DIR = EXPORT_DIR / "models"

# ============================================================
# ENTRENAMIENTO
# ============================================================
BATCH_SIZE  = int(os.getenv("BATCH_SIZE",  "64"))
THR_MASK    = float(os.getenv("THR_MASK",  "0.5"))   # Umbral de binarización de la máscara

# ============================================================
# NESTED CROSS-VALIDATION Y HPO
# ============================================================
SEED_BASE = 42

N_OUTER  = int(os.getenv("N_OUTER",  "5"))    # Folds del loop externo        (prueba: 2)
N_INNER  = int(os.getenv("N_INNER",  "5"))    # Folds del loop interno (HPO)  (prueba: 2)
N_TRIALS = int(os.getenv("N_TRIALS", "50"))   # Trials de Optuna por fold     (prueba: 2)

# Epochs (subir para runs de producción)
MAX_EPOCHS_TRIAL    = int(os.getenv("MAX_EPOCHS_TRIAL",    "10"))  # prueba: 1-3
MAX_EPOCHS_FINAL    = int(os.getenv("MAX_EPOCHS_FINAL",    "100")) # prueba: 5-15
MAX_EPOCHS_ABLATION = int(os.getenv("MAX_EPOCHS_ABLATION", "50"))  # prueba: 5-10

# Validación durante los trials de HPO
TRIAL_VALIDATION_FREQ  = int(os.getenv("TRIAL_VALIDATION_FREQ",  "2"))
TRIAL_PATIENCE         = int(os.getenv("TRIAL_PATIENCE",          "1"))
TRIAL_STEPS_PER_EPOCH  = int(os.getenv("TRIAL_STEPS_PER_EPOCH",   "0"))  # 0 = sin límite
TRIAL_VAL_STEPS        = int(os.getenv("TRIAL_VAL_STEPS",          "0"))  # 0 = sin límite

# ============================================================
# TEST CIEGO Y ABLACIÓN
# ============================================================
N_FINAL_SEEDS        = int(os.getenv("N_FINAL_SEEDS", "5"))
FINAL_SEEDS          = [SEED_BASE + i for i in range(N_FINAL_SEEDS)]

RUN_FINAL_BLIND_TEST = (os.getenv("RUN_FINAL_BLIND_TEST", "1") == "1")
RUN_ABLATIONS        = (os.getenv("RUN_ABLATIONS",        "1") == "1")
RUN_STATS_TESTS      = (os.getenv("RUN_STATS_TESTS",      "1") == "1")

ABLATION_VARIANTS = ["multitask", "cls_only", "seg_only", "unweighted_losses"]

# ============================================================
# ETIQUETA DE EJECUCIÓN (para distinguir múltiples runs)
# ============================================================
_tag      = os.getenv("RUN_TAG", "").strip()
RUN_TAG   = f"_{_tag}" if _tag else ""

# ============================================================
# RUTAS DE EXPORTACIÓN (derivadas)
# ============================================================
TRIALS_CSV     = EXPORT_DIR / f"optuna_trials_nested{RUN_TAG}.csv"
OUTER_CSV      = EXPORT_DIR / f"nested_outer_results{RUN_TAG}.csv"
PARETO_CSV     = EXPORT_DIR / f"pareto_front_trials{RUN_TAG}.csv"
FINAL_TEST_CSV = EXPORT_DIR / f"final_blind_test_multiseed{RUN_TAG}.csv"
STATS_CSV      = EXPORT_DIR / f"stat_tests{RUN_TAG}.csv"
SQLITE_PATH    = EXPORT_DIR / f"optuna_nested_outer{RUN_TAG}.sqlite3"
