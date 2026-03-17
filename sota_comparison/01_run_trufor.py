"""
01_run_trufor.py — Construye imagen Docker y ejecuta inferencia de TruFor
sobre el holdout de FantasyID.

Estado tras 00_export_holdout.py:
  - sota_comparison/trufor_repo/          ← ya clonado
  - sota_comparison/trufor_weights/trufor.pth.tar ← ya descargado

Este script:
  1. Construye la imagen Docker (1ª vez ~15 min, luego usa caché)
  2. Ejecuta test.py en Docker sobre holdout_images/
  3. Parsea los .npz de salida y genera trufor_scores.csv

Prerequisitos:
  - Docker Desktop instalado y corriendo (con GPU si es posible)
  - Haber ejecutado 00_export_holdout.py

Uso:
    cd E:/PycharmProjects/DocVerify
    python sota_comparison/01_run_trufor.py

    # Solo parsear output si Docker ya corrió:
    python sota_comparison/01_run_trufor.py --skip_docker

    # Dry run (imprime comandos sin ejecutar):
    python sota_comparison/01_run_trufor.py --dry_run
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_THIS_DIR = Path(__file__).resolve().parent
_ROOT     = _THIS_DIR.parent

TRUFOR_REPO_DIR = _THIS_DIR / "trufor_repo"
WEIGHTS_FILE    = _THIS_DIR / "trufor_weights" / "trufor.pth.tar"
IMAGES_DIR      = _THIS_DIR / "holdout_images"
OUTPUT_DIR      = _THIS_DIR / "trufor_output"
SCORES_CSV      = _THIS_DIR / "trufor_scores.csv"
DOCKERFILE      = _THIS_DIR / "Dockerfile"

DOCKER_IMAGE_NAME = "trufor_docverify"


# ============================================================
# UTILIDADES
# ============================================================

def _run(cmd: list, check: bool = True, dry_run: bool = False) -> int:
    cmd_str = " ".join(str(c) for c in cmd)
    print(f"\n  $ {cmd_str}")
    if dry_run:
        print("  [DRY RUN]")
        return 0
    result = subprocess.run(cmd, check=check)
    return result.returncode


def _check_docker() -> bool:
    try:
        subprocess.run(["docker", "info"], capture_output=True, check=True)
        print("[OK] Docker disponible")
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("[ERROR] Docker no detectado o no está corriendo.")
        print("  Instala Docker Desktop: https://www.docker.com/products/docker-desktop/")
        return False


def _image_exists() -> bool:
    r = subprocess.run(["docker", "images", "-q", DOCKER_IMAGE_NAME],
                       capture_output=True, text=True)
    return bool(r.stdout.strip())


# ============================================================
# PASO 1: Construir imagen Docker
# ============================================================

def build_docker(dry_run: bool = False):
    if _image_exists():
        print(f"[OK] Imagen Docker '{DOCKER_IMAGE_NAME}' ya existe — usando caché")
        return

    print(f"\n[1/3] Construyendo imagen Docker '{DOCKER_IMAGE_NAME}'...")
    print("  (Primera vez: ~15-20 min por descarga e instalación de mmcv-full)")
    print("  Puedes monitorizar el progreso en Docker Desktop")

    _run([
        "docker", "build",
        "-t", DOCKER_IMAGE_NAME,
        "-f", str(DOCKERFILE),
        str(_THIS_DIR)
    ], dry_run=dry_run)

    print(f"[OK] Imagen '{DOCKER_IMAGE_NAME}' construida")


# ============================================================
# PASO 2: Ejecutar inferencia TruFor
# ============================================================

def run_inference(dry_run: bool = False, use_gpu: bool = True):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Verificar prerequisitos
    if not WEIGHTS_FILE.exists():
        print(f"[ERROR] No se encontraron pesos en {WEIGHTS_FILE}")
        sys.exit(1)

    images = list(IMAGES_DIR.rglob("*.jpg")) + list(IMAGES_DIR.rglob("*.png"))
    if not images:
        print(f"[ERROR] No se encontraron imágenes en {IMAGES_DIR}")
        print("  Ejecuta primero: python sota_comparison/00_export_holdout.py")
        sys.exit(1)

    print(f"\n[2/3] Ejecutando TruFor sobre {len(images)} imágenes...")
    print("  (GPU: ~5-10 min | CPU: ~60-90 min)")

    # Rutas absolutas para volúmenes Docker
    images_abs  = IMAGES_DIR.resolve()
    output_abs  = OUTPUT_DIR.resolve()
    weights_abs = WEIGHTS_FILE.resolve()

    # Convertir rutas Windows → formato Docker en WSL2
    # C:\foo\bar → /c/foo/bar   (Docker Desktop con WSL2)
    def _win_to_docker(p: Path) -> str:
        s = str(p).replace("\\", "/")
        if len(s) >= 2 and s[1] == ":":
            drive = s[0].lower()
            s = f"/{drive}/{s[3:]}"
        return s

    images_docker  = _win_to_docker(images_abs)
    output_docker  = _win_to_docker(output_abs)
    weights_docker = _win_to_docker(weights_abs.parent)

    cmd = [
        "docker", "run", "--rm",
    ]

    if use_gpu:
        cmd += ["--gpus", "all"]

    cmd += [
        "-v", f"{images_docker}:/input:ro",
        "-v", f"{output_docker}:/output",
        "-v", f"{weights_docker}:/weights:ro",
        DOCKER_IMAGE_NAME,
        # Argumentos para test.py:
        "-in",  "/input",
        "-out", "/output",
        "-exp", "trufor_ph3",
        "TEST.MODEL_FILE", f"/weights/{WEIGHTS_FILE.name}",
    ]

    _run(cmd, dry_run=dry_run)
    print(f"[OK] Resultados en: {OUTPUT_DIR}")


# ============================================================
# PASO 3: Parsear output .npz → CSV
# ============================================================

def parse_output() -> None:
    """
    Parsea los archivos .npz generados por TruFor.

    Estructura de cada .npz:
      - 'score'  : float [0,1] — score de detección (1=forjado)
      - 'map'    : array 2D float32 — mapa de localización de anomalías
      - 'conf'   : array 2D float32 — mapa de confianza (opcional)
      - 'imgsize': tuple (H, W) — tamaño original de la imagen

    Genera trufor_scores.csv con columnas:
      stem, trufor_score, map_npy_path
    """
    if not OUTPUT_DIR.exists() or not any(OUTPUT_DIR.rglob("*.npz")):
        print(f"\n[ERROR] No se encontraron archivos .npz en {OUTPUT_DIR}")
        print("  ¿Se ejecutó correctamente el paso Docker?")
        print(f"  Contenido de {OUTPUT_DIR}:")
        if OUTPUT_DIR.exists():
            for p in OUTPUT_DIR.rglob("*"):
                print(f"    {p.relative_to(OUTPUT_DIR)}")
        sys.exit(1)

    npz_files = list(OUTPUT_DIR.rglob("*.npz"))
    print(f"\n[3/3] Parseando {len(npz_files)} archivos .npz...")

    rows = []
    n_missing_score = 0

    for npz_path in sorted(npz_files):
        try:
            data = np.load(npz_path, allow_pickle=True)
        except Exception as e:
            print(f"  [WARN] No se pudo cargar {npz_path.name}: {e}")
            continue

        # El stem del .npz es el nombre del archivo imagen original
        # TruFor guarda: output/<subpath>/<image_name>.jpg.npz
        # → stem = image_name (sin .jpg.npz)
        npz_name = npz_path.name  # e.g. "train_attack_digital_1_huawei_img.jpg.npz"
        # Eliminar .npz y posiblemente .jpg
        stem = npz_name
        for suffix in [".npz", ".jpg", ".png", ".jpeg"]:
            if stem.endswith(suffix):
                stem = stem[:-len(suffix)]

        # Score de detección
        if "score" in data:
            score = float(data["score"])
        else:
            # Fallback: media del mapa si no hay score
            n_missing_score += 1
            if "map" in data:
                score = float(data["map"].max())
            else:
                score = float("nan")

        # Guardar el mapa de anomalía como .npy separado para Dice
        map_npy_path = "N/A"
        if "map" in data:
            map_save = OUTPUT_DIR / f"{stem}_map.npy"
            np.save(map_save, data["map"].astype(np.float32))
            map_npy_path = str(map_save)

        rows.append({
            "stem":         stem,
            "trufor_score": score,
            "map_npy_path": map_npy_path,
        })

    df = pd.DataFrame(rows)
    df.to_csv(SCORES_CSV, index=False)

    if n_missing_score > 0:
        print(f"  [WARN] {n_missing_score} imágenes sin 'score' en .npz — usado max(map)")

    print(f"[OK] {len(df)} scores guardados en: {SCORES_CSV}")
    if len(df) > 0:
        valid = df["trufor_score"].dropna()
        print(f"  Score range: [{valid.min():.4f}, {valid.max():.4f}] "
              f"| media={valid.mean():.4f}")
    print("\n[SIGUIENTE PASO] Ejecuta: sota_02_eval_comparison (run config en PyCharm)")


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Ejecutar TruFor (Docker) sobre el holdout de FantasyID"
    )
    parser.add_argument("--skip_docker", action="store_true",
                        help="Omitir build+run Docker (solo parsear output existente)")
    parser.add_argument("--skip_build", action="store_true",
                        help="Omitir docker build (usar imagen ya construida)")
    parser.add_argument("--no_gpu", action="store_true",
                        help="Ejecutar en CPU (más lento pero sin GPU requerida)")
    parser.add_argument("--dry_run", action="store_true",
                        help="Imprimir comandos sin ejecutar")
    args = parser.parse_args()

    print("=" * 60)
    print(" DocVerify — TruFor Inferencia")
    print("=" * 60)

    if not args.skip_docker:
        if not _check_docker():
            sys.exit(1)

        if not args.skip_build:
            build_docker(dry_run=args.dry_run)

        run_inference(dry_run=args.dry_run, use_gpu=not args.no_gpu)

    parse_output()


if __name__ == "__main__":
    main()
