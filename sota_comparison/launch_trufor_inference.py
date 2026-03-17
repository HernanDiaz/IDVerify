"""
launch_trufor_inference.py — Lanza 'docker run trufor_docverify' en background y sale inmediatamente.
La inferencia continúa en el contenedor aunque este script termine.

Uso (desde PyCharm run config 'sota_trufor_inference'):
    python sota_comparison/launch_trufor_inference.py

Cuando termine (monitoriza con sota_check_inference), ejecuta: sota_01_parse_output
"""
import subprocess
from pathlib import Path

_DIR = Path(__file__).resolve().parent

images_abs  = (_DIR / "holdout_images").resolve()
output_abs  = (_DIR / "trufor_output").resolve()
weights_abs = (_DIR / "trufor_weights").resolve()

output_abs.mkdir(parents=True, exist_ok=True)


def win2docker(p: Path) -> str:
    """Convierte ruta Windows (C:\\foo\\bar) a formato Docker mount (/c/foo/bar)."""
    s = str(p).replace("\\", "/")
    if len(s) >= 2 and s[1] == ":":
        s = f"/{s[0].lower()}/{s[3:]}"
    return s


# Nota: la imagen usa 'trufor_test.py' (versión anterior del script TruFor).
# Sintaxis correcta: -gpu <N> (0=primera GPU, -1=CPU), sin -exp, con YACS opts al final.
# --gpus all → necesario para que Docker exponga la GPU al contenedor.
cmd = [
    "docker", "run", "--rm",
    "--name", "trufor_inference",
    "--gpus", "all",
    "-v", f"{win2docker(images_abs)}:/input:ro",
    "-v", f"{win2docker(output_abs)}:/output",
    "-v", f"{win2docker(weights_abs)}:/weights:ro",
    "trufor_docverify",
    "-gpu", "0",            # GPU 0 (usa -1 si no hay GPU disponible)
    "-in",  "/input",
    "-out", "/output",
    "TEST.MODEL_FILE", "/weights/trufor.pth.tar",
]

print("Lanzando TruFor inference en background...")
print(f"  $ {' '.join(cmd)}")
print()
print(f"  Input :  {images_abs}")
print(f"  Output:  {output_abs}")
print(f"  Weights: {weights_abs}")
print()
print("La inferencia tarda ~5-15 min con GPU. Los .npz se guardan en trufor_output/")
print("Monitoriza con run config: sota_check_inference")
print("Cuando termine, ejecuta run config: sota_01_parse_output")
print()

# Popen: lanza y NO espera — el proceso Python termina inmediatamente
proc = subprocess.Popen(cmd)
print(f"[OK] Contenedor 'trufor_inference' iniciado — PID {proc.pid}")
print("Saliendo del script. Docker sigue ejecutando en background.")
