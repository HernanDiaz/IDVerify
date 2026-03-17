"""Inspect the trufor_docverify Docker image without running a container."""
import subprocess
import json
from pathlib import Path

_DIR = Path(__file__).resolve().parent

# Use docker inspect to get image metadata (no container startup needed)
print("=== docker inspect trufor_docverify (entrypoint / cmd / workdir) ===")
r = subprocess.run(
    ["docker", "inspect", "trufor_docverify"],
    capture_output=True, text=True, timeout=15
)
if r.returncode == 0:
    data = json.loads(r.stdout)[0]
    cfg = data.get("Config", {})
    print(f"  Entrypoint : {cfg.get('Entrypoint')}")
    print(f"  Cmd        : {cfg.get('Cmd')}")
    print(f"  WorkingDir : {cfg.get('WorkingDir')}")
    print(f"  Image size : {data.get('Size', 0) // 1_000_000} MB")
else:
    print("STDERR:", r.stderr)

# Check debug container logs
print("\n=== docker logs trufor_debug_noexp (last 20 lines) ===")
r2 = subprocess.run(["docker", "logs", "--tail", "20", "trufor_debug_noexp"],
                    capture_output=True, text=True, timeout=10)
if r2.stdout or r2.stderr:
    print(r2.stdout)
    if r2.stderr:
        print("STDERR:", r2.stderr)
else:
    print("(sin logs o contenedor no existe)")

# Count .npz files
for folder in ["trufor_sample_output", "trufor_output"]:
    d = _DIR / folder
    npz = list(d.rglob("*.npz")) if d.exists() else []
    print(f"\n{folder}: {len(npz)} .npz archivos")
    for f in sorted(npz):
        print(f"  {f.name}")
