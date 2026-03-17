"""
check_inference_status.py — Monitoriza el progreso de TruFor inference.
Funciona tanto con Docker como con proceso venv (run_trufor_venv.py).

Uso (desde PyCharm run config 'sota_check_inference'):
    python sota_comparison/check_inference_status.py
"""
import subprocess
from pathlib import Path

_DIR = Path(__file__).resolve().parent
output_dir = _DIR / "trufor_output"

# 1. Contar .npz generados (en cualquier subdirectorio)
npz_files = list(output_dir.rglob("*.npz")) if output_dir.exists() else []
total_expected = 498  # holdout size
print(f"[NPZ] {len(npz_files)} / {total_expected} archivos .npz generados en trufor_output/")

# 2. Comprobar si hay proceso Python test.py corriendo (venv, sin Docker)
r_proc = subprocess.run(
    ["tasklist", "/FI", "IMAGENAME eq python.exe", "/FO", "CSV", "/NH"],
    capture_output=True, text=True
)
trufor_proc_running = "python.exe" in r_proc.stdout

# 3. Comprobar Docker container
r_docker = subprocess.run(
    ["docker", "ps", "--filter", "name=trufor_inference", "--format", "{{.Names}}\t{{.Status}}"],
    capture_output=True, text=True
)
container_running = "trufor_inference" in r_docker.stdout

# 4. Estado global
if container_running:
    print(f"[RUN]  Contenedor Docker 'trufor_inference' EN EJECUCIÓN")
elif trufor_proc_running:
    print(f"[RUN]  Proceso Python (venv) EN EJECUCIÓN — PID(s) activos:")
    for line in r_proc.stdout.strip().splitlines():
        if "python" in line.lower():
            print(f"       {line}")
else:
    if len(npz_files) >= total_expected:
        print(f"[DONE] Inferencia COMPLETADA — {len(npz_files)} archivos generados")
        print("[NEXT] Ejecuta run config: sota_01_parse_output")
    elif len(npz_files) > 0:
        print(f"[INFO] {len(npz_files)} .npz generados. Puede que la inferencia haya terminado parcialmente.")
        if len(npz_files) >= total_expected * 0.95:
            print("[NEXT] Suficiente para análisis — ejecuta: sota_01_parse_output")
    else:
        print("[WAIT] Sin proceso activo y sin .npz.")
        print("       Lanza la inferencia con: sota_trufor_venv")

# 5. Mostrar últimos 5 archivos generados
if npz_files:
    npz_sorted = sorted(npz_files, key=lambda f: f.stat().st_mtime, reverse=True)
    print(f"\nÚltimos archivos generados ({min(5, len(npz_files))} de {len(npz_files)}):")
    for f in npz_sorted[:5]:
        print(f"  {f.relative_to(output_dir)}")
    # Mostrar breakdown por subdirectorio
    subdirs = set(f.parent for f in npz_files)
    if len(subdirs) > 1:
        print(f"\nSubdirectorios con .npz:")
        for d in sorted(subdirs):
            n = len(list(d.glob("*.npz")))
            print(f"  {d.relative_to(output_dir)}: {n} archivos")
