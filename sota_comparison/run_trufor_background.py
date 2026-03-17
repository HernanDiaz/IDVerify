"""
run_trufor_background.py — Lanza TruFor inference en background real (sin PIPE).

Escribe stdout+stderr a fichero en disco para que el proceso hijo siga
corriendo aunque este script termine (sin PIPE = sin BrokenPipeError).

Monitoriza progreso con: sota_check_inference
"""

import subprocess
import sys
import os
from pathlib import Path

_DIR     = Path(__file__).resolve().parent
_TRUFOR  = _DIR / "trufor_repo" / "TruFor_train_test"
_WEIGHTS = _DIR / "trufor_weights" / "trufor.pth.tar"
_IMAGES  = _DIR / "holdout_images"
_OUTPUT  = _DIR / "trufor_output"
_LOGFILE = _DIR / "trufor_run.log"
_OUTPUT.mkdir(parents=True, exist_ok=True)

env = os.environ.copy()
env["PYTHONPATH"] = str(_TRUFOR) + os.pathsep + env.get("PYTHONPATH", "")
# Permite al allocator de CUDA usar segmentos expandibles → evita OOM por fragmentación
env["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

cmd = [
    sys.executable,
    str(_TRUFOR / "test.py"),
    "-g", "0",
    "-in",       str(_IMAGES),
    "-out",      str(_OUTPUT),
    "-exp",      "trufor_ph3",
    "-max_size", "1024",          # 3200px → 1024px; evita OOM con mit_b2 (tokens cuadráticos)
    "TEST.MODEL_FILE", str(_WEIGHTS),
]

print(f"[INFO] Log  : {_LOGFILE}")
print(f"[INFO] Cmd  : {' '.join(cmd[:6])} ...")
print(f"[INFO] max_size=1024  (3200px -> 1024px; evita OOM con mit_b2 en 16GB VRAM)")
print()

# Usar fichero en disco en modo binario — el hijo escribe directamente sin buffering Python
logf = open(_LOGFILE, "wb")
logf.write(f"CMD: {' '.join(cmd)}\n\n".encode("utf-8"))
logf.flush()

proc = subprocess.Popen(
    cmd,
    cwd=str(_TRUFOR),
    env=env,
    stdout=logf,    # fichero binario en disco, NO PIPE
    stderr=logf,    # stderr también al mismo fichero
)
logf.close()        # cierra handle del padre; el hijo conserva el suyo

print(f"[OK] PID {proc.pid} iniciado en background")
print(f"     Los kernels CUDA ya están compilados — debería procesar ~498 imgs en <10 min")
print(f"     Monitoriza con: sota_check_inference")
print(f"     Log en tiempo real: type trufor_run.log (o tail -f en Linux)")
