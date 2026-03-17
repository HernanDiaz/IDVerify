"""
launch_trufor_debug.py — Diagnóstico completo del entorno Docker de TruFor.
1. Muestra la ayuda del script dentro del contenedor
2. Lista los archivos en /trufor
3. Muestra las primeras líneas de test.py
4. Prueba invocación correcta con 5 imágenes de muestra

Uso (desde PyCharm run config 'sota_trufor_debug'):
    python sota_comparison/launch_trufor_debug.py
"""
import subprocess
import shutil
from pathlib import Path

_DIR = Path(__file__).resolve().parent
weights_abs = (_DIR / "trufor_weights").resolve()


def run(cmd, timeout=60, check=False):
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    out = (r.stdout + r.stderr).strip()
    print(out[:4000] if out else "(sin salida)")
    print(f"  [exit {r.returncode}]")
    return r


def win2docker(p: Path) -> str:
    s = str(p).replace("\\", "/")
    if len(s) >= 2 and s[1] == ":":
        s = f"/{s[0].lower()}/{s[3:]}"
    return s


# ──────────────────────────────────────────────
print("=" * 60)
print("1. HELP del script dentro del contenedor")
print("=" * 60)
run(["docker", "run", "--rm", "trufor_docverify", "--help"])

# ──────────────────────────────────────────────
print("\n" + "=" * 60)
print("2. Archivos en /trufor (dentro del contenedor)")
print("=" * 60)
run(["docker", "run", "--rm", "--entrypoint", "find",
     "trufor_docverify", "/trufor", "-maxdepth", "1", "-type", "f"])

# ──────────────────────────────────────────────
print("\n" + "=" * 60)
print("3. Primeras 50 líneas de /trufor/test.py")
print("=" * 60)
run(["docker", "run", "--rm", "--entrypoint", "head",
     "trufor_docverify", "-n", "50", "/trufor/test.py"])

# ──────────────────────────────────────────────
print("\n" + "=" * 60)
print("4. Prueba de inferencia (5 imágenes, CPU, sin -exp)")
print("=" * 60)

sample_dir = _DIR / "trufor_sample_input"
output_dir = _DIR / "trufor_sample_output"

if output_dir.exists():
    shutil.rmtree(output_dir)
output_dir.mkdir(parents=True)

if sample_dir.exists():
    shutil.rmtree(sample_dir)
sample_dir.mkdir(parents=True)

all_images = sorted((_DIR / "holdout_images").rglob("*.jpg"))[:5]
for img in all_images:
    shutil.copy(img, sample_dir / img.name)
print(f"Imágenes de muestra: {[f.name for f in sorted(sample_dir.iterdir())]}")

# Intentar sin -exp (por si el script interno no lo soporta)
cmd_no_exp = [
    "docker", "run", "--rm",
    "--name", "trufor_debug_noexp",
    "-v", f"{win2docker(sample_dir.resolve())}:/input:ro",
    "-v", f"{win2docker(output_dir.resolve())}:/output",
    "-v", f"{win2docker(weights_abs)}:/weights:ro",
    "trufor_docverify",
    "-gpu", "-1",          # CPU
    "-in", "/input",
    "-out", "/output",
    "TEST.MODEL_FILE", "/weights/trufor.pth.tar",
]
subprocess.run(["docker", "rm", "-f", "trufor_debug_noexp"], capture_output=True)
print(f"\n$ {' '.join(cmd_no_exp)}")
r_noexp = run(cmd_no_exp, timeout=120)
npz = list(output_dir.rglob("*.npz"))
print(f"  .npz generados: {len(npz)}")

if not npz:
    # Intentar con -exp también
    print("\n--- Reintentando CON -exp trufor_ph3 ---")
    shutil.rmtree(output_dir)
    output_dir.mkdir()
    cmd_exp = [
        "docker", "run", "--rm",
        "-v", f"{win2docker(sample_dir.resolve())}:/input:ro",
        "-v", f"{win2docker(output_dir.resolve())}:/output",
        "-v", f"{win2docker(weights_abs)}:/weights:ro",
        "trufor_docverify",
        "-gpu", "-1",
        "-in", "/input",
        "-out", "/output",
        "-exp", "trufor_ph3",
        "TEST.MODEL_FILE", "/weights/trufor.pth.tar",
    ]
    print(f"\n$ {' '.join(cmd_exp)}")
    r_exp = run(cmd_exp, timeout=120)
    npz2 = list(output_dir.rglob("*.npz"))
    print(f"  .npz generados: {len(npz2)}")

print("\n=== RESUMEN ===")
npz_final = list(output_dir.rglob("*.npz"))
if npz_final:
    print(f"[OK] {len(npz_final)} archivos generados. La inferencia funciona.")
    print("     Lanza la inferencia completa con: sota_trufor_inference")
else:
    print("[ERROR] No se generaron .npz. Revisa el output de arriba.")
