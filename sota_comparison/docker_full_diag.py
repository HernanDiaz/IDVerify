"""Full Docker diagnostics — check daemon, GPU, containers."""
import subprocess
import json
from pathlib import Path

_DIR = Path(__file__).resolve().parent


def run(cmd, timeout=20):
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        out = (r.stdout + r.stderr).strip()
        return out, r.returncode
    except subprocess.TimeoutExpired:
        return "(TIMEOUT)", -999
    except FileNotFoundError:
        return "(command not found)", -1


print("=== 1. Docker daemon status ===")
out, code = run(["docker", "version", "--format", "{{.Server.Version}}"])
print(f"Docker server version: {out}  [exit {code}]")

print("\n=== 2. GPU disponible en Docker ===")
out, code = run(["docker", "run", "--rm", "--gpus", "all", "nvidia/cuda:11.3.0-base-ubuntu20.04",
                 "nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"])
print(f"nvidia-smi output: {out}  [exit {code}]")

print("\n=== 3. docker ps -a ===")
out, _ = run(["docker", "ps", "-a"])
print(out)

print("\n=== 4. Logs trufor_debug_noexp ===")
out, code = run(["docker", "logs", "--tail", "10", "trufor_debug_noexp"])
print(f"{out}  [exit {code}]")

print("\n=== 5. .npz generados en trufor_sample_output ===")
sample_out = _DIR / "trufor_sample_output"
npz = list(sample_out.rglob("*.npz")) if sample_out.exists() else []
print(f"{len(npz)} archivos: {[f.name for f in npz]}")

print("\n=== 6. .npz generados en trufor_output ===")
trufor_out = _DIR / "trufor_output"
npz2 = list(trufor_out.rglob("*.npz")) if trufor_out.exists() else []
print(f"{len(npz2)} archivos")

print("\n=== 7. Intentar lanzar trufor_inference en CPU (sin --gpus) ===")
images_abs = (_DIR / "holdout_images").resolve()
output_abs = trufor_out
weights_abs = (_DIR / "trufor_weights").resolve()

def win2docker(p):
    s = str(p).replace("\\", "/")
    if len(s) >= 2 and s[1] == ":":
        s = f"/{s[0].lower()}/{s[3:]}"
    return s

# Kill any existing container first
subprocess.run(["docker", "rm", "-f", "trufor_inference"], capture_output=True)

cmd = [
    "docker", "run", "--rm",
    "--name", "trufor_inference",
    # NO --gpus
    "-v", f"{win2docker(images_abs)}:/input:ro",
    "-v", f"{win2docker(output_abs)}:/output",
    "-v", f"{win2docker(weights_abs)}:/weights:ro",
    "trufor_docverify",
    "-gpu", "-1",       # CPU
    "-in", "/input",
    "-out", "/output",
    "TEST.MODEL_FILE", "/weights/trufor.pth.tar",
]
print(f"$ {' '.join(cmd)}")
proc = subprocess.Popen(cmd)
print(f"[OK] CPU inference lanzada — PID {proc.pid}")
print("     Monitoriza con sota_check_inference")
