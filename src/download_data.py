"""Descarga y prepara los datasets del TP de detección de grietas.

Uso:
    # Detección (Roboflow) requiere una API key en la variable de entorno:
    #   Windows PowerShell:  $env:ROBOFLOW_API_KEY = "tu_key"
    #   Linux/Mac:           export ROBOFLOW_API_KEY=tu_key
    python src/download_data.py            # baja ambos datasets
    python src/download_data.py --det      # solo detección (Roboflow)
    python src/download_data.py --seg      # solo segmentación (DeepCrack/Kaggle)
    python src/download_data.py --force    # rehace aunque ya exista

Es idempotente: si el dataset ya está preparado, no lo vuelve a descargar.

- dataset_9000_yolo : detección. Se baja de Roboflow (YOLOv11, v1) y se convierten
  los labels de polígono a bounding box (min/max). Clase única 'crack'.
- dataset_1000      : segmentación (DeepCrack). Se baja de Kaggle vía kagglehub.
"""
import argparse
import os
import shutil
import sys
from collections import Counter
from pathlib import Path

# --- Rutas -------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASETS = PROJECT_ROOT / "datasets"
DET_DIR = DATASETS / "dataset_9000_yolo"      # detección (destino final)
SEG_DIR = DATASETS / "dataset_1000"           # segmentación (destino final)

# --- Config Roboflow ---------------------------------------------------------
RF_WORKSPACE = "alan-vignolo"
RF_PROJECT = "crack-detection-katdf-actsg"
RF_VERSION = 1
RF_FORMAT = "yolov11"
CLASS_NAME = "crack"

SPLITS = ["train", "valid", "test"]
IMG_EXTS = {".jpg", ".jpeg", ".png"}


# --- Conversión polígono -> bbox (idéntica a la validada en el EDA) ----------
def _poly_to_box(coords):
    xs, ys = coords[0::2], coords[1::2]
    x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
    return ((x0 + x1) / 2, (y0 + y1) / 2, x1 - x0, y1 - y0)


def _clamp01(v):
    return min(1.0, max(0.0, v))


def _convert_line(line):
    """Línea YOLO (bbox o polígono) -> 'cls xc yc w h', o None si es inválida."""
    p = line.split()
    if len(p) < 5:
        return None
    cls = int(float(p[0]))
    vals = [float(x) for x in p[1:]]
    if len(p) == 5:
        xc, yc, w, h = vals
    else:
        if len(vals) % 2 != 0:
            return None
        xc, yc, w, h = _poly_to_box(vals)
    xc, yc, w, h = (_clamp01(xc), _clamp01(yc), _clamp01(w), _clamp01(h))
    if w <= 0 or h <= 0:
        return None
    return f"{cls} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}"


def _convert_labels_inplace(root: Path):
    """Reescribe todos los .txt de root/{split}/labels a formato bbox de 5 campos."""
    stats = Counter()
    for split in SPLITS:
        lbl_dir = root / split / "labels"
        if not lbl_dir.is_dir():
            continue
        for txt in lbl_dir.glob("*.txt"):
            out = []
            for line in txt.read_text().strip().splitlines():
                if not line.strip():
                    continue
                nf = len(line.split())
                conv = _convert_line(line)
                if conv is None:
                    stats["descartadas"] += 1
                    continue
                if nf > 5:
                    stats["desde_poligono"] += 1
                out.append(conv)
            txt.write_text("\n".join(out) + ("\n" if out else ""))
            stats["boxes"] += len(out)
    return stats


def _write_data_yaml(root: Path):
    yaml_txt = f"""# Dataset de detección de grietas — TP Visión por Computadora II (CEIA, Grupo 5)
# Generado por src/download_data.py (Roboflow YOLOv11 v{RF_VERSION}, labels polígono -> bbox).
path: {root.as_posix()}
train: train/images
val: valid/images
test: test/images

nc: 1
names: ['{CLASS_NAME}']

# Fuente: https://universe.roboflow.com/{RF_WORKSPACE}/{RF_PROJECT}
# Licencia: CC BY 4.0
"""
    (root / "data.yaml").write_text(yaml_txt, encoding="utf-8")


def _is_ready(root: Path) -> bool:
    """True si el dataset ya está preparado (splits + data.yaml presentes)."""
    if not (root / "data.yaml").is_file():
        return False
    return all((root / s / "images").is_dir() and (root / s / "labels").is_dir()
               for s in SPLITS)


# --- Detección: Roboflow -----------------------------------------------------
def prepare_detection(force=False):
    if _is_ready(DET_DIR) and not force:
        print(f"[det] Ya preparado en {DET_DIR} — se omite (usá --force para rehacer).")
        return

    api_key = os.environ.get("ROBOFLOW_API_KEY")
    if not api_key:
        sys.exit(
            "[det] ERROR: falta la variable de entorno ROBOFLOW_API_KEY.\n"
            "      PowerShell:  $env:ROBOFLOW_API_KEY = \"tu_key\"\n"
            "      Linux/Mac:   export ROBOFLOW_API_KEY=tu_key\n"
            "      (Obtené tu key en Roboflow > Settings > API Keys)"
        )

    try:
        from roboflow import Roboflow
    except ImportError:
        sys.exit("[det] ERROR: falta 'roboflow'. Instalá con:  uv add roboflow")

    if DET_DIR.exists():
        shutil.rmtree(DET_DIR)

    print(f"[det] Descargando de Roboflow ({RF_PROJECT} v{RF_VERSION}, {RF_FORMAT})...")
    rf = Roboflow(api_key=api_key)
    project = rf.workspace(RF_WORKSPACE).project(RF_PROJECT)
    dataset = project.version(RF_VERSION).download(RF_FORMAT, location=str(DET_DIR))
    print(f"[det] Descargado en {dataset.location}")

    print("[det] Convirtiendo labels de polígono a bounding box...")
    stats = _convert_labels_inplace(DET_DIR)
    _write_data_yaml(DET_DIR)
    print(f"[det] Listo. boxes={stats['boxes']}  "
          f"desde_poligono={stats['desde_poligono']}  descartadas={stats['descartadas']}")


# --- Segmentación: Kaggle (DeepCrack) ---------------------------------------
def prepare_segmentation(force=False):
    # DeepCrack usa carpetas train_img/train_lab/test_img/test_lab
    ready = all((SEG_DIR / d).is_dir()
                for d in ["train_img", "train_lab", "test_img", "test_lab"])
    if ready and not force:
        print(f"[seg] Ya preparado en {SEG_DIR} — se omite (usá --force para rehacer).")
        return

    try:
        import kagglehub
    except ImportError:
        sys.exit(
            "[seg] ERROR: falta 'kagglehub'. Instalá con:  uv add kagglehub\n"
            "      (Requiere credenciales de Kaggle: ~/.kaggle/kaggle.json "
            "o variables KAGGLE_USERNAME / KAGGLE_KEY)"
        )

    print("[seg] Descargando DeepCrack de Kaggle...")
    path = kagglehub.dataset_download("rukiyeaydn/deepcrack-dataset")
    print(f"[seg] Descargado en caché de kagglehub: {path}")
    print(f"[seg] NOTA: revisá la estructura y copiá/enlazá las carpetas a {SEG_DIR}\n"
          "      (train_img/, train_lab/, test_img/, test_lab/). "
          "La estructura interna del zip de Kaggle puede variar.")


def main():
    ap = argparse.ArgumentParser(description="Descarga los datasets del TP.")
    ap.add_argument("--det", action="store_true", help="solo detección (Roboflow)")
    ap.add_argument("--seg", action="store_true", help="solo segmentación (Kaggle)")
    ap.add_argument("--force", action="store_true", help="rehacer aunque exista")
    args = ap.parse_args()

    do_det = args.det or not args.seg
    do_seg = args.seg or not args.det

    DATASETS.mkdir(exist_ok=True)
    if do_det:
        prepare_detection(force=args.force)
    if do_seg:
        prepare_segmentation(force=args.force)
    print("\nListo.")


if __name__ == "__main__":
    main()
