"""Check if TruFor process (test.py) is running and look for new .npz files."""
import subprocess
from pathlib import Path

_DIR = Path(__file__).resolve().parent

# Check all python processes
r = subprocess.run(["wmic", "process", "where", "name='python.exe'",
                    "get", "ProcessId,CommandLine,WorkingSetSize"],
                   capture_output=True, text=True, timeout=10)
print("=== Python processes ===")
print(r.stdout[:3000])

# Count .npz files
output_dir = _DIR / "trufor_output"
npz = list(output_dir.rglob("*.npz")) if output_dir.exists() else []
print(f"\n=== .npz count: {len(npz)} / 498 ===")
for f in sorted(npz, key=lambda x: x.stat().st_mtime, reverse=True)[:5]:
    print(f"  {f.relative_to(output_dir)}")
