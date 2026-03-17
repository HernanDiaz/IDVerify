"""
run_trufor_logged.py — Corre TruFor con log completo a fichero.
Útil para diagnosticar errores cuando el proceso crashea.

Lanza test.py como proceso hijo y redirige stdout+stderr a trufor_run.log.
Luego sigue ejecutando y muestra el log en tiempo real (hasta 120s para PyCharm).
El proceso hijo continúa aunque este script termine.
"""
import subprocess
import sys
import os
from pathlib import Path
import threading

_DIR = Path(__file__).resolve().parent
_TRUFOR = _DIR / "trufor_repo" / "TruFor_train_test"
_WEIGHTS = _DIR / "trufor_weights" / "trufor.pth.tar"
_IMAGES  = _DIR / "holdout_images"
_OUTPUT  = _DIR / "trufor_output"
_LOGFILE = _DIR / "trufor_run.log"
_OUTPUT.mkdir(parents=True, exist_ok=True)

env = os.environ.copy()
env["PYTHONPATH"] = str(_TRUFOR) + os.pathsep + env.get("PYTHONPATH", "")

cmd = [
    sys.executable,
    str(_TRUFOR / "test.py"),
    "-g", "0",
    "-in",  str(_IMAGES),
    "-out", str(_OUTPUT),
    "-exp", "trufor_ph3",
    "TEST.MODEL_FILE", str(_WEIGHTS),
]

print(f"Log: {_LOGFILE}")
print(f"Cmd: {' '.join(cmd[:6])} ...")
print()

with open(_LOGFILE, "w", encoding="utf-8") as logf:
    logf.write(f"CMD: {' '.join(cmd)}\n\n")
    logf.flush()

    proc = subprocess.Popen(
        cmd,
        cwd=str(_TRUFOR),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    print(f"[OK] PID {proc.pid} iniciado — salida en {_LOGFILE.name}")

    # Leer output en tiempo real (hasta timeout de PyCharm ~120s)
    # El proceso hijo continúa aunque el script termine
    import time
    start = time.time()
    lines_shown = 0
    try:
        for line in proc.stdout:
            logf.write(line)
            logf.flush()
            print(line, end="", flush=True)
            lines_shown += 1
            if time.time() - start > 110:  # Near PyCharm timeout
                print(f"\n[INFO] Límite de tiempo alcanzado — proceso sigue en background (PID {proc.pid})")
                print(f"       Monitoriza con: sota_check_pid y sota_check_inference")
                break
    except Exception as e:
        print(f"[WARN] Error leyendo output: {e}")

print(f"\n[INFO] {lines_shown} líneas escritas en {_LOGFILE}")
