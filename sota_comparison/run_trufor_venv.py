"""
run_trufor_venv.py — Corre TruFor directamente en el venv del proyecto
(Python 3.11 + PyTorch 2.x + RTX 5060 Ti sm_120, sin Docker).

Por qué sin Docker:
  - El Docker usa PyTorch 1.11 que no soporta sm_120 (RTX 5060 Ti)
  - TruFor NO usa mmcv ni mmsegmentation (son puro PyTorch + timm + yacs)
  - Podemos correrlo directamente en el venv existente

Prerrequisitos (se instalan automáticamente si faltan):
  - timm        (backbone SegFormer)
  - yacs        (config YACS)
  - Pillow, numpy, tqdm (ya deberían estar en el venv DocVerify)

Uso (PyCharm run config 'sota_trufor_venv'):
    python sota_comparison/run_trufor_venv.py
"""
import subprocess
import sys
import os
from pathlib import Path

_DIR = Path(__file__).resolve().parent
_TRUFOR = _DIR / "trufor_repo" / "TruFor_train_test"
_WEIGHTS = _DIR / "trufor_weights" / "trufor.pth.tar"
_IMAGES = _DIR / "holdout_images"
_OUTPUT = _DIR / "trufor_output"

_OUTPUT.mkdir(parents=True, exist_ok=True)

# ─── 1. Instalar dependencias que falten ─────────────────────────────────────
print("=" * 60)
print("Verificando dependencias...")
print("=" * 60)

missing = []
try:
    import timm
    print(f"  [OK] timm {timm.__version__}")
except ImportError:
    missing.append("timm")

try:
    import yacs
    print(f"  [OK] yacs")
except ImportError:
    missing.append("yacs")

try:
    from PIL import Image
    print(f"  [OK] Pillow")
except ImportError:
    missing.append("Pillow")

if missing:
    print(f"\nInstalando: {missing}")
    subprocess.check_call([sys.executable, "-m", "pip", "install"] + missing)
    print("  [OK] Instalación completada")

# ─── 2. Verificar archivos necesarios ────────────────────────────────────────
print(f"\nVerificando archivos...")
assert _TRUFOR.exists(), f"TruFor repo no encontrado en {_TRUFOR}"
assert _WEIGHTS.exists(), f"Pesos no encontrados en {_WEIGHTS}"
assert _IMAGES.exists(), f"Imágenes no encontradas en {_IMAGES}"

img_count = len(list(_IMAGES.rglob("*.jpg")) + list(_IMAGES.rglob("*.png")))
print(f"  [OK] {img_count} imágenes en holdout_images/")
print(f"  [OK] Pesos: {_WEIGHTS}")

# ─── 3. Verificar GPU ─────────────────────────────────────────────────────────
import torch
print(f"\nGPU disponible: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"  Dispositivo: {torch.cuda.get_device_name(0)}")
    print(f"  CUDA version: {torch.version.cuda}")
    gpu_flag = "-g"
    gpu_id = "0"
else:
    print("  [WARN] Sin GPU — usando CPU (lento)")
    gpu_flag = "-g"
    gpu_id = "-1"

# ─── 4. Lanzar test.py directamente con el venv ───────────────────────────────
print("\n" + "=" * 60)
print(f"Lanzando TruFor inference ({img_count} imágenes)...")
print("=" * 60)

cmd = [
    sys.executable,
    str(_TRUFOR / "test.py"),
    gpu_flag, gpu_id,
    "-in",  str(_IMAGES),
    "-out", str(_OUTPUT),
    "-exp", "trufor_ph3",
    "TEST.MODEL_FILE", str(_WEIGHTS),
]

print(f"Comando: {' '.join(cmd)}")
print(f"PYTHONPATH: {_TRUFOR}\n")

env = os.environ.copy()
env["PYTHONPATH"] = str(_TRUFOR) + os.pathsep + env.get("PYTHONPATH", "")

# Lanzar en background (Popen) para no bloquear PyCharm más de 120s
proc = subprocess.Popen(cmd, cwd=str(_TRUFOR), env=env)
print(f"[OK] Proceso iniciado — PID {proc.pid}")
print(f"     Output: {_OUTPUT}")
print(f"\nMonitoriza el progreso con run config: sota_check_inference")
print(f"Cuando termine, ejecuta: sota_01_parse_output")
