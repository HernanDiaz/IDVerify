"""
run_trufor_finetuned_inference.py — Inferencia TruFor fine-tuneado sobre el holdout.

Usa best.pth.tar (fine-tuning sobre FantasyID, 30 épocas) para evaluar
las 498 imágenes del holdout. Genera:
  sota_comparison/trufor_finetuned_output/   ← .npz por imagen
  sota_comparison/trufor_finetuned_scores.csv ← stem, score, map_path

Uso:
    cd E:/PycharmProjects/DocVerify
    python sota_comparison/run_trufor_finetuned_inference.py
"""

import subprocess
import sys
import os
import numpy as np
import pandas as pd
from pathlib import Path

_DIR    = Path(__file__).resolve().parent
_TRUFOR = _DIR / "trufor_repo" / "TruFor_train_test"
_WEIGHTS = _TRUFOR / "weights" / "trufor_fantasyid" / "best.pth.tar"
_IMAGES  = _DIR / "holdout_images"
_OUTPUT  = _DIR / "trufor_finetuned_output"
_SCORES  = _DIR / "trufor_finetuned_scores.csv"
_LOGFILE = _DIR / "trufor_finetuned_inference.log"

_OUTPUT.mkdir(parents=True, exist_ok=True)


# ============================================================
# PASO 1: Inferencia
# ============================================================

def run_inference():
    if not _WEIGHTS.exists():
        print(f"[ERROR] No se encontró el modelo fine-tuneado: {_WEIGHTS}")
        sys.exit(1)

    images = list(_IMAGES.rglob("*.jpg")) + list(_IMAGES.rglob("*.png"))
    print(f"[INFO] {len(images)} imágenes en holdout")

    env = os.environ.copy()
    env["PYTHONPATH"] = str(_TRUFOR) + os.pathsep + env.get("PYTHONPATH", "")

    cmd = [
        sys.executable,
        str(_TRUFOR / "test.py"),
        "-g", "0",
        "-in",  str(_IMAGES),
        "-out", str(_OUTPUT),
        "-exp", "trufor_ph3",
        "TEST.MODEL_FILE", str(_WEIGHTS),
        "VALID.MAX_SIZE", "512",
    ]

    print(f"[INFO] Lanzando inferencia con modelo fine-tuneado...")
    print(f"  Modelo  : {_WEIGHTS}")
    print(f"  Entrada : {_IMAGES}  ({len(images)} imgs)")
    print(f"  Salida  : {_OUTPUT}")
    print(f"  Log     : {_LOGFILE}")
    print()

    logf = open(_LOGFILE, "wb")
    logf.write(f"CMD: {' '.join(cmd)}\n\n".encode("utf-8"))
    logf.flush()

    proc = subprocess.Popen(
        cmd,
        cwd=str(_TRUFOR),
        env=env,
        stdout=logf,
        stderr=logf,
    )
    logf.close()

    print(f"[OK] PID {proc.pid} iniciado — esperando finalización...")
    proc.wait()

    rc = proc.returncode
    if rc != 0:
        print(f"[ERROR] El proceso terminó con código {rc}. Revisa {_LOGFILE}")
        sys.exit(rc)

    print("[OK] Inferencia completada.")


# ============================================================
# PASO 2: Parsear .npz → CSV
# ============================================================

def parse_output():
    npz_files = list(_OUTPUT.rglob("*.npz"))
    if not npz_files:
        print(f"[ERROR] No se encontraron .npz en {_OUTPUT}")
        sys.exit(1)

    print(f"\n[INFO] Parseando {len(npz_files)} archivos .npz...")
    rows = []
    n_missing_score = 0

    for npz_path in sorted(npz_files):
        try:
            data = np.load(npz_path, allow_pickle=True)
        except Exception as e:
            print(f"  [WARN] No se pudo cargar {npz_path.name}: {e}")
            continue

        # Derivar stem (sin .jpg.npz / .png.npz)
        stem = npz_path.name
        for suffix in [".npz", ".jpg", ".png", ".jpeg"]:
            if stem.endswith(suffix):
                stem = stem[:-len(suffix)]

        # Score de detección
        if "score" in data:
            score = float(data["score"])
        else:
            n_missing_score += 1
            score = float(data["map"].max()) if "map" in data else float("nan")

        # Guardar mapa de anomalía como .npy para Dice
        map_npy_path = "N/A"
        if "map" in data:
            map_save = _OUTPUT / f"{stem}_map.npy"
            np.save(map_save, data["map"].astype(np.float32))
            map_npy_path = str(map_save)

        rows.append({
            "stem":         stem,
            "trufor_score": score,
            "map_npy_path": map_npy_path,
        })

    df = pd.DataFrame(rows)
    df.to_csv(_SCORES, index=False)

    if n_missing_score > 0:
        print(f"  [WARN] {n_missing_score} imágenes sin 'score' — usado max(map)")

    valid = df["trufor_score"].dropna()
    print(f"[OK] {len(df)} scores guardados en {_SCORES}")
    print(f"  Score range : [{valid.min():.4f}, {valid.max():.4f}]")
    print(f"  Media       : {valid.mean():.4f}")
    print(f"\n[SIGUIENTE PASO] Ejecuta: python sota_comparison/eval_finetuned_comparison.py")


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print(" TruFor Fine-tuneado — Inferencia sobre Holdout")
    print("=" * 60)
    run_inference()
    parse_output()
