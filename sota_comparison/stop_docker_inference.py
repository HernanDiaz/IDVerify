"""Para el contenedor Docker trufor_inference (CPU, demasiado lento)."""
import subprocess
r = subprocess.run(["docker", "stop", "trufor_inference"], capture_output=True, text=True)
print(r.stdout, r.stderr)
print(f"Exit: {r.returncode}")
r2 = subprocess.run(["docker", "ps", "-a", "--filter", "name=trufor"], capture_output=True, text=True)
print(r2.stdout)
