"""Comprueba si el build de Docker ha terminado."""
import subprocess, sys
r = subprocess.run(["docker", "images", "-q", "trufor_docverify"],
                   capture_output=True, text=True)
if r.stdout.strip():
    print("[OK] Imagen 'trufor_docverify' LISTA")
    sys.exit(0)
else:
    print("[WAIT] Imagen todavia no disponible — build en progreso...")
    # Mostrar capas en curso
    r2 = subprocess.run(["docker", "ps", "-a"], capture_output=True, text=True)
    print(r2.stdout)
    sys.exit(1)
