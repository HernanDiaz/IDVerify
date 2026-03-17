"""Check logs of the running trufor_debug_noexp container."""
import subprocess
from pathlib import Path

_DIR = Path(__file__).resolve().parent
output_dir = _DIR / "trufor_sample_output"

print("=== docker logs trufor_debug_noexp ===")
r = subprocess.run(["docker", "logs", "--tail", "30", "trufor_debug_noexp"],
                   capture_output=True, text=True)
print(r.stdout)
if r.stderr:
    print("STDERR:", r.stderr)
print(f"Exit: {r.returncode}")

npz = list(output_dir.rglob("*.npz")) if output_dir.exists() else []
print(f"\n.npz generados hasta ahora: {len(npz)}")
for f in sorted(npz):
    print(f"  {f.name}")
