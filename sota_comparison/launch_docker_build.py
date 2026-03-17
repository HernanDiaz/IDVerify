"""
launch_docker_build.py — Lanza 'docker build' en background y sale inmediatamente.
El build continúa en Docker Desktop aunque este script termine.

Uso (desde PyCharm run config 'sota_docker_build'):
    python sota_comparison/launch_docker_build.py

Cuando termine el build (ver Docker Desktop), ejecuta STEP2b_run_inference.bat
"""
import subprocess
import sys
from pathlib import Path

_THIS_DIR = Path(__file__).resolve().parent

cmd = [
    "docker", "build",
    "-t", "trufor_docverify",
    "-f", str(_THIS_DIR / "Dockerfile"),
    str(_THIS_DIR),
]
print("Lanzando docker build en background...")
print(f"  $ {' '.join(cmd)}")
print()
print("El build tarda ~15-25 minutos. Puedes monitorizarlo en Docker Desktop.")
print("Cuando termine, ejecuta: STEP2b_run_inference.bat")
print()

# Popen: lanza y NO espera — el proceso Python termina inmediatamente
proc = subprocess.Popen(
    cmd,
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    text=True,
)
print(f"[OK] Build iniciado — PID {proc.pid}")
print("Saliendo del script. Docker sigue construyendo en background.")
