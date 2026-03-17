"""
kill_and_relaunch.py — Mata procesos zombie de TruFor y relanza inferencia limpia.
"""
import subprocess
import sys
import os
from pathlib import Path

_DIR = Path(__file__).resolve().parent
_TRUFOR = _DIR / "trufor_repo" / "TruFor_train_test"
_WEIGHTS = _DIR / "trufor_weights" / "trufor.pth.tar"
_IMAGES  = _DIR / "holdout_images"
_OUTPUT  = _DIR / "trufor_output"
_OUTPUT.mkdir(parents=True, exist_ok=True)

# 1. Identificar y matar procesos test.py y test_trufor_quick.py
print("=== Buscando procesos Python de TruFor ===")
r = subprocess.run(
    ["wmic", "process", "where", "name='python.exe'",
     "get", "ProcessId,CommandLine"],
    capture_output=True, text=True, timeout=10
)

pids_to_kill = []
for line in r.stdout.splitlines():
    line = line.strip()
    if not line or "CommandLine" in line:
        continue
    # La línea tiene CommandLine + ProcessId al final
    # Buscamos líneas con test.py o test_trufor_quick
    if "test.py" in line or "test_trufor_quick" in line:
        # El PID está al final de la línea
        parts = line.rsplit(None, 1)
        if len(parts) == 2:
            try:
                pid = int(parts[-1])
                pids_to_kill.append(pid)
                print(f"  [KILL] PID {pid}: {parts[0][:80]}...")
            except ValueError:
                pass

if pids_to_kill:
    for pid in pids_to_kill:
        r_kill = subprocess.run(["taskkill", "/F", "/PID", str(pid)],
                                capture_output=True, text=True)
        print(f"  taskkill PID {pid}: {r_kill.stdout.strip()} {r_kill.stderr.strip()}")
    print(f"[OK] {len(pids_to_kill)} proceso(s) terminado(s)")
else:
    print("  (sin procesos de TruFor activos)")

print()

# 2. Breve pausa para que se libere VRAM
import time
time.sleep(3)

# 3. Relanzar inferencia completa
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

print(f"Relanzando inferencia sobre {len(list(_IMAGES.rglob('*.jpg')))} imágenes...")
print(f"  Cmd: {' '.join(cmd[:6])} ...")
proc = subprocess.Popen(cmd, cwd=str(_TRUFOR), env=env)
print(f"[OK] Proceso iniciado — PID {proc.pid}")
print("     Monitoriza con: sota_check_inference")
print("     Cuando termine: sota_01_parse_output")
