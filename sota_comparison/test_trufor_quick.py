"""
test_trufor_quick.py — Test BLOCKING en 3 imágenes para verificar que funciona.
Muestra stdout/stderr en tiempo real para detectar errores.

Uso (PyCharm run config 'sota_trufor_quick_test'):
    python sota_comparison/test_trufor_quick.py
"""
import subprocess
import sys
import os
import shutil
from pathlib import Path

_DIR = Path(__file__).resolve().parent
_TRUFOR = _DIR / "trufor_repo" / "TruFor_train_test"
_WEIGHTS = _DIR / "trufor_weights" / "trufor.pth.tar"
_SAMPLE_IN = _DIR / "trufor_sample_input"
_SAMPLE_OUT = _DIR / "trufor_quick_output"

# Usar 3 imágenes de muestra
if _SAMPLE_OUT.exists():
    shutil.rmtree(_SAMPLE_OUT)
_SAMPLE_OUT.mkdir()

if not _SAMPLE_IN.exists() or len(list(_SAMPLE_IN.rglob("*.jpg"))) < 3:
    _SAMPLE_IN.mkdir(exist_ok=True)
    all_imgs = sorted((_DIR / "holdout_images").rglob("*.jpg"))[:3]
    for img in all_imgs:
        shutil.copy(img, _SAMPLE_IN / img.name)

n = len(list(_SAMPLE_IN.glob("*.jpg")))
print(f"Testing con {n} imágenes en: {_SAMPLE_IN}")
print(f"Output dir: {_SAMPLE_OUT}")
print()

cmd = [
    sys.executable,
    str(_TRUFOR / "test.py"),
    "-g", "0",              # GPU 0
    "-in",  str(_SAMPLE_IN),
    "-out", str(_SAMPLE_OUT),
    "-exp", "trufor_ph3",
    "TEST.MODEL_FILE", str(_WEIGHTS),
]

env = os.environ.copy()
env["PYTHONPATH"] = str(_TRUFOR) + os.pathsep + env.get("PYTHONPATH", "")

print(f"Cmd: {' '.join(cmd)}")
print(f"CWD: {_TRUFOR}")
print(f"PYTHONPATH: {env['PYTHONPATH'][:100]}")
print("=" * 60)

# Blocking — muestra output en tiempo real
result = subprocess.run(cmd, cwd=str(_TRUFOR), env=env)
print("=" * 60)
print(f"Exit code: {result.returncode}")

npz = list(_SAMPLE_OUT.rglob("*.npz"))
print(f"Archivos .npz generados: {len(npz)}")
for f in npz:
    print(f"  {f.name}")
