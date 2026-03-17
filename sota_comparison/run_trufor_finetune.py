"""
run_trufor_finetune.py — Lanza el fine-tuning de TruFor sobre FantasyID.

Prerequisitos (en orden):
  1. El zero-shot ha completado y los resultados están evaluados.
  2. sota_trufor_prepare_fid ha generado los list-files y las máscaras:
       TruFor_train_test/dataset/data/FID_train_list.txt
       TruFor_train_test/dataset/data/FID_valid_list.txt
       sota_comparison/fantasyid_masks/

El fine-tuning:
  - Parte de trufor.pth.tar (Epoch 81, CVPR 2023)
  - Congela NP++, backbone y loc_head
  - Entrena conf_head y det_head durante 30 épocas
  - Guarda best.pth.tar según avg_det_bacc en validación FantasyID
  - Output en: TruFor_train_test/weights/trufor_fantasyid/

Para evaluar después del fine-tuning:
  - Usar run_trufor_logged.py cambiando el modelo a weights/trufor_fantasyid/best.pth.tar
  - O lanzar sota_trufor_eval_finetuned
"""

import subprocess
import sys
import os
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
_TRUFOR     = _SCRIPT_DIR / "trufor_repo" / "TruFor_train_test"
_WEIGHTS    = _SCRIPT_DIR / "trufor_weights" / "trufor.pth.tar"
_TRAIN_LIST = _TRUFOR / "dataset" / "data" / "FID_train_list.txt"
_VALID_LIST = _TRUFOR / "dataset" / "data" / "FID_valid_list.txt"
_LOGFILE    = _SCRIPT_DIR / "trufor_finetune.log"


def check_prerequisites():
    """Verifica que el dataset está preparado antes de lanzar el entrenamiento."""
    ok = True

    if not _WEIGHTS.exists():
        print(f"[ERROR] Weights no encontrados: {_WEIGHTS}")
        ok = False

    if not _TRAIN_LIST.exists():
        print(f"[ERROR] Train list no encontrado: {_TRAIN_LIST}")
        print("        Ejecuta primero: sota_trufor_prepare_fid")
        ok = False
    else:
        n_train = sum(1 for _ in open(_TRAIN_LIST))
        print(f"[OK] Train list: {n_train} imágenes")

    if not _VALID_LIST.exists():
        print(f"[ERROR] Valid list no encontrado: {_VALID_LIST}")
        print("        Ejecuta primero: sota_trufor_prepare_fid")
        ok = False
    else:
        n_valid = sum(1 for _ in open(_VALID_LIST))
        print(f"[OK] Valid list: {n_valid} imágenes")

    return ok


def install_train_deps():
    """Instala dependencias de entrenamiento si no están disponibles."""
    deps = ["albumentations", "tensorboardX", "opencv-python"]
    for dep in deps:
        pkg = dep.split("-")[0]
        try:
            __import__(pkg if pkg != "opencv" else "cv2")
        except ImportError:
            print(f"[DEP] Instalando {dep}...")
            subprocess.run([sys.executable, "-m", "pip", "install", dep, "-q"],
                           check=True)
            print(f"[OK] {dep} instalado")


if __name__ == "__main__":
    print("=" * 60)
    print(" TruFor Fine-tuning — FantasyID")
    print("=" * 60)

    # 1. Verificar prerequisitos
    if not check_prerequisites():
        print("\n[ERROR] Prerequisitos no cumplidos. Abortando.")
        sys.exit(1)

    # 2. Instalar dependencias
    install_train_deps()

    # 3. Preparar entorno
    env = os.environ.copy()
    env["PYTHONPATH"] = str(_TRUFOR) + os.pathsep + env.get("PYTHONPATH", "")

    cmd = [
        sys.executable,
        str(_TRUFOR / "train.py"),
        "-exp", "trufor_fantasyid",
        "-g", "0",
    ]

    print(f"\n[OK] Lanzando fine-tuning...")
    print(f"  Config : trufor_fantasyid.yaml")
    print(f"  Cmd    : {' '.join(cmd)}")
    print(f"  CWD    : {_TRUFOR}")
    print(f"  Log    : {_LOGFILE}")
    print()

    # Usar fichero binario en disco — el hijo escribe directamente aunque el padre termine
    logf = open(_LOGFILE, "wb")
    logf.write(f"CMD: {' '.join(cmd)}\n\n".encode("utf-8"))
    logf.flush()

    proc = subprocess.Popen(
        cmd,
        cwd=str(_TRUFOR),
        env=env,
        stdout=logf,   # fichero binario, NO PIPE
        stderr=logf,   # stderr al mismo fichero
    )
    logf.close()       # padre cierra su handle; hijo conserva el suyo

    print(f"[OK] PID {proc.pid} iniciado en background")
    print(f"     Output en: {_TRUFOR}/weights/trufor_fantasyid/")
    print(f"     Log      : {_LOGFILE}")
    print(f"     Monitoriza con: sota_check_finetune")
