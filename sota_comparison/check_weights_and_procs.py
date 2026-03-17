"""Check weights file size and kill duplicate TruFor processes."""
import subprocess
from pathlib import Path

_DIR = Path(__file__).resolve().parent

# Check weights size
w = _DIR / "trufor_weights" / "trufor.pth.tar"
print(f"trufor.pth.tar size: {w.stat().st_size / 1e9:.2f} GB" if w.exists() else "NOT FOUND")

# List all python processes with full command line
r = subprocess.run(
    ["wmic", "process", "where", "name='python.exe'",
     "get", "ProcessId,WorkingSetSize,CommandLine"],
    capture_output=True, text=True, timeout=10
)
print("\n=== Python processes ===")
lines = [l for l in r.stdout.splitlines() if l.strip()]
for line in lines:
    print(line[:200])
