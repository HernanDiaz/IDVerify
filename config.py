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
# ENTRENAMIENTO
# ============================================================
BATCH_SIZE  = int(os.getenv("BATCH_SIZE",  "32"))
THR_MASK    = float(os.getenv("THR_MASK",  "0.5"))   # Umbral de binarización de la máscara

# ============================================================
# NESTED CROSS-VALIDATION Y HPO
# ============================================================
SEED_BASE = 42

N_OUTER  = int(os.getenv("N_OUTER",  "3"))    # Folds del loop externo        (prueba: 2)
N_INNER  = int(os.getenv("N_INNER",  "2"))    # Folds del loop interno (HPO)  (prueba: 2)
N_TRIALS = int(os.getenv("N_TRIALS", "5"))    # Trials de Optuna por fold     (prueba: 2)

# Epochs (subir para runs de producción)
MAX_EPOCHS_TRIAL    = int(os.getenv("MAX_EPOCHS_TRIAL",    "3"))   # producción: 5-10
MAX_EPOCHS_FINAL    = int(os.getenv("MAX_EPOCHS_FINAL",    "15"))  # producción: 30-100
MAX_EPOCHS_ABLATION = int(os.getenv("MAX_EPOCHS_ABLATION", "10"))  # producción: 20

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
