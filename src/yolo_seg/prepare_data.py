"""Convierte las máscaras binarias de DeepCrack (dataset_1000) al formato
YOLO-Seg: un .txt por imagen con una línea por instancia 'cls x1 y1 x2 y2 ...'
(polígono, coordenadas normalizadas 0-1).

Cada componente conexa de la máscara se trata como una instancia de grieta.

Módulo de utilidades importado por los notebooks (no es un script de un solo uso):
- `01_check_labels.ipynb`: inspecciona las labels convertidas.
- `02_train.ipynb`: llama a `build_dataset()` para generar los datasets YOLO-Seg
  (versión full-polygon y versión con train dilatado), y usa `mask_to_polygons()`
  en la §6 para medir el techo de IoU de la conversión máscara→polígono.

Ejecutado directamente (`python prepare_data.py`) genera el dataset por defecto.
"""
from pathlib import Path

import albumentations as A
import cv2
import numpy as np

import random
import shutil

MIN_AREA_PX = 15 # descarta componentes muy pequeñas (ruido)
APROX_EPS_FRAC = 0.002 # tolerancia de approxPolyDP, como fracción del perímetro (solo si simplify=True)
CLASS_ID = 0 # clase de grieta

PROJECT_ROOT = Path(__file__).resolve().parents[2]  # sube desde src/yolo_seg/ hasta la raíz del repo
SRC_ROOT = PROJECT_ROOT / "datasets" / "dataset_1000"           # DeepCrack original
OUT_ROOT = PROJECT_ROOT / "datasets" / "dataset_1000_yoloseg"   # dataset convertido, listo para Ultralytics

VAL_FRACTION = 0.15  # % de train_img que se separa como validación
SPLIT_SEED = 42      # fija el split para que sea reproducible entre corridas y entre compañeros


def mask_to_polygons(mask: np.ndarray, min_area: int = MIN_AREA_PX, dilate_px: int = 0,
                     simplify: bool = False) -> list[np.ndarray]:
    """Devuelve una lista de poligonos (Nx2, en pixeles) por instancia de grieta en la máscara binaria.

    Args:
        mask: máscara binaria (0/255 o 0/1) de grietas con shape (H, W).
        min_area: área mínima en píxeles para considerar una componente como grieta.
        dilate_px: si > 0, engrosa la máscara antes de extraer contornos (kernel (2*dilate_px+1)).
            Compensa la sensibilidad del IoU semántico a formas finas: el modelo aprende contornos
            un poco más generosos, que calzan mejor con el grosor real de la grieta al evaluar.
            Usar solo en train -- val/test deben evaluarse contra el ground truth real.
        simplify: si True, aplica approxPolyDP (eps=APROX_EPS_FRAC) para reducir vértices.
            DESACTIVADO por defecto: en grietas finas la simplificación recorta el grosor real
            del trazo y baja el techo de IoU semántico de ~0.93 a ~0.78 (medido sobre test_lab).
            Solo tiene sentido si se necesita achicar el tamaño de las labels.

    Nota: se extraen TODOS los contornos de cada componente (antes solo el mayor), para no perder
    ramas/lazos internos de la grieta -- otra fuente de área perdida en la conversión a polígono."""

    bin_mask = (mask > 127).astype(np.uint8)  # binariza la máscara
    if dilate_px > 0:
        kernel = np.ones((dilate_px * 2 + 1, dilate_px * 2 + 1), np.uint8)
        bin_mask = cv2.dilate(bin_mask, kernel)
    n_labels, labels = cv2.connectedComponents(bin_mask) # obtiene componentes conexas (píxeles adyacentes forman una región)

    polygons = []
    for label in range(1, n_labels): # etiqueta 0 es el fondo
        component = (labels == label).astype(np.uint8) # obtiene la máscara de la componente
        if component.sum() < min_area:
            continue # descarta componentes muy pequeños
        # CHAIN_APPROX_NONE: conserva todos los puntos del contorno (no submuestrea el borde)
        contours, _ = cv2.findContours(component, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        for contour in contours: # todos los contornos, no solo el mayor
            if len(contour) < 3:
                continue # descarta si el contorno tiene menos de 3 puntos
            if simplify:
                perimeter = cv2.arcLength(contour, closed=True)
                approx = cv2.approxPolyDP(contour, epsilon=APROX_EPS_FRAC * perimeter, closed=True)
                if len(approx) >= 3:
                    contour = approx  # solo si la simplificación no colapsó la forma
            polygons.append(contour.reshape(-1, 2)) # agrega el polígono (Nx2) a la lista
    return polygons


def polygons_to_yolo_lines(polygons: list[np.ndarray], img_w: int, img_h: int, class_id: int = CLASS_ID) -> list[str]:
    """Convierte una lista de poligonos (Nx2, en pixeles) a líneas de texto en formato YOLO-Seg.
    
    Args:
        polygons: lista de poligonos (Nx2, en pixeles).
        img_w: ancho de la imagen.
        img_h: alto de la imagen.
        class_id: id de clase para las instancias."""
    
    lines = []
    for poly in polygons:
        norm = poly.astype(np.float64) # copia en float para dividir sin perder precisión
        norm[:, 0] /= img_w # normaliza coordenadas x
        norm[:, 1] /= img_h # normaliza coordenadas y
        norm = np.clip(norm, 0.0, 1.0) # asegura que las coordenadas estén en [0, 1]
        coords = " ".join(f"{v:.6f}" for v in norm.flatten())  # aplana a x1 y1 x2 y2 ... como string
        lines.append(f"{class_id} {coords}") # agrega la línea con clase y coordenadas
    return lines

def mask_file_to_yolo_txt(mask_path: Path, img_w: int, img_h: int, dilate_px: int = 0) -> list[str]:
    """Lee un archivo de máscara y devuelve las líneas YOLO-Seg correspondientes."""
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE) # lee la máscara en escala de grises
    if mask is None:
        raise FileNotFoundError(f"No se pudo leer la máscara: {mask_path}")
    polygons = mask_to_polygons(mask, dilate_px=dilate_px) # obtiene los poligonos de la máscara

    return polygons_to_yolo_lines(polygons, img_w, img_h) # convierte a líneas YOLO-Seg

def _stem_pairs(img_dir: Path, lab_dir: Path) -> list[tuple[Path, Path]]:
    """Empareja cada imagen con su máscara por nombre de archivo (mismo stem)."""
    pairs = []
    for img_path in sorted(img_dir.iterdir()):
        lab_path = lab_dir / f"{img_path.stem}.png"  # asume que las máscaras son .png
        if not lab_path.is_file():
            raise FileNotFoundError(f"Falta máscara para {img_path.name}: {lab_path}")
        pairs.append((img_path, lab_path))
    return pairs

def _write_split(pairs: list[tuple[Path, Path]], split: str, out_root: Path, dilate_px: int = 0):
    """Copia imágenes y escribe labels YOLO-Seg para un split (train/val/test).

    dilate_px solo debe usarse para 'train' -- val/test deben quedar con el ground truth real."""
    img_out = out_root / split / "images"
    lbl_out = out_root / split / "labels"
    img_out.mkdir(parents=True, exist_ok=True) # crea carpetas si no existen
    lbl_out.mkdir(parents=True, exist_ok=True)

    for img_path, lab_path in pairs:
        img = cv2.imread(str(img_path)) # lee la imagen en tamaño real para normalizar
        if img is None:
            raise FileNotFoundError(f"No se pudo leer la imagen: {img_path}")
        h, w = img.shape[:2]

        lines = mask_file_to_yolo_txt(lab_path, w, h, dilate_px=dilate_px) # obtiene líneas YOLO-Seg de la máscara
        (lbl_out / f"{img_path.stem}.txt").write_text("\n".join(lines), encoding="utf-8") # escribe el archivo de labels
        shutil.copy2(img_path, img_out / img_path.name) # copia la imagen al directorio de salida

def _write_data_yaml(out_root: Path):
    yaml_txt = f"""# Dataset de segmentación de grietas (YOLO-Seg) — TP Visión por Computadora II (CEIA, Grupo 5)
    # Generado por src/yolo_seg/prepare_data.py a partir de máscaras DeepCrack (componentes conexas -> polígonos).
    path: {out_root.as_posix()}
    train: train/images
    val: val/images
    test: test/images

    nc: 1
    names: ['crack']

    # Fuente original: https://www.kaggle.com/datasets/rukiyeaydn/deepcrack-dataset
    # Split train/val generado con seed={SPLIT_SEED}, val_fraction={VAL_FRACTION} (test_img/test_lab del dataset original queda intacto)
    """
    (out_root / "data.yaml").write_text(yaml_txt, encoding="utf-8")

def build_dataset(force: bool = False, dilate_train_px: int = 0, out_root: Path = None):
    """Arma {out_root}/{train,val,test}/{images,labels}.

    dilate_train_px: engrosa la máscara de TRAIN antes de generar polígonos (ver mask_to_polygons).
        val y test siempre usan la máscara original, sin dilatar.
    out_root: carpeta de salida; por defecto OUT_ROOT (datasets/dataset_1000_yoloseg)."""
    out_root = out_root or OUT_ROOT
    if out_root.exists():
        if not force:
            print(f"Ya existe {out_root} — usá force=True para rehacer.")
            return
        shutil.rmtree(out_root)

    train_pairs = _stem_pairs(SRC_ROOT / "train_img", SRC_ROOT / "train_lab")
    test_pairs = _stem_pairs(SRC_ROOT / "test_img", SRC_ROOT / "test_lab")

    rng = random.Random(SPLIT_SEED)  # generador propio, no toca el estado global de random
    shuffled = train_pairs[:]         # copia para no mutar train_pairs
    rng.shuffle(shuffled)
    n_val = round(len(shuffled) * VAL_FRACTION)
    val_pairs = shuffled[:n_val]
    train_only_pairs = shuffled[n_val:]

    _write_split(train_only_pairs, "train", out_root, dilate_px=dilate_train_px)
    _write_split(val_pairs, "val", out_root)   # sin dilatar: ground truth real
    _write_split(test_pairs, "test", out_root)  # sin dilatar: ground truth real
    _write_data_yaml(out_root)

    print(f"train={len(train_only_pairs)} val={len(val_pairs)} test={len(test_pairs)}")
    print(f"Listo en {out_root} (dilate_train_px={dilate_train_px})")


# --- Augmentation offline (multiplica train con variantes sintéticas) ------

_OFFLINE_AUG_TRANSFORM = A.Compose(
    [
        A.Rotate(limit=20, border_mode=cv2.BORDER_CONSTANT, fill=0, p=0.7),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.5),
    ],
    keypoint_params=A.KeypointParams(format="xy", remove_invisible=False),
)


def _augment_once(img: np.ndarray, polygons: list[np.ndarray]) -> tuple[np.ndarray, list[np.ndarray]]:
    """Aplica una transformación aleatoria a la imagen y sus polígonos, preservando la forma.

    Los polígonos se aplanan a keypoints (remove_invisible=False) para no perder vértices
    si una parte de la grieta queda fuera del frame tras rotar -- eso dejaría un polígono
    con menos de 3 puntos, inválido para YOLO-Seg.
    """
    all_points = [(float(x), float(y)) for poly in polygons for x, y in poly]
    poly_lengths = [len(poly) for poly in polygons]

    result = _OFFLINE_AUG_TRANSFORM(image=img, keypoints=all_points)
    aug_img = result["image"]
    aug_points = result["keypoints"]

    h, w = aug_img.shape[:2]
    aug_polygons = []
    idx = 0
    for length in poly_lengths:
        pts = np.array(aug_points[idx:idx + length], dtype=np.float64)
        idx += length
        if len(pts) < 3:
            continue  # la transformación dejó muy pocos puntos válidos; se descarta esa instancia
        pts[:, 0] = np.clip(pts[:, 0], 0, w - 1)
        pts[:, 1] = np.clip(pts[:, 1], 0, h - 1)
        aug_polygons.append(pts)
    return aug_img, aug_polygons


def augment_train_offline(out_root: Path, n_variants: int = 5, dilate_px: int = 0, seed: int = SPLIT_SEED):
    """Multiplica train/{images,labels} de out_root con n_variants copias aumentadas por imagen.

    Debe correrse DESPUÉS de build_dataset (usa las imágenes/máscaras originales de
    SRC_ROOT, no las ya escritas en out_root, para no perder precisión por dilatar dos veces).
    Las variantes se guardan con sufijo '_augN' en el nombre, junto a las imágenes originales
    -- val y test no se tocan.
    """
    train_pairs = _stem_pairs(SRC_ROOT / "train_img", SRC_ROOT / "train_lab")

    # mismo split que build_dataset, para generar variantes solo de las imágenes que quedaron en train
    rng_split = random.Random(seed)
    shuffled = train_pairs[:]
    rng_split.shuffle(shuffled)
    n_val = round(len(shuffled) * VAL_FRACTION)
    train_only_pairs = shuffled[n_val:]

    img_out = out_root / "train" / "images"
    lbl_out = out_root / "train" / "labels"

    # nota: albumentations usa su propio RNG interno, no determinístico vía random/numpy.seed
    # acá -- no hace falta reproducibilidad estricta de las variantes, solo que sean válidas y variadas.
    total_written = 0
    for img_path, lab_path in train_only_pairs:
        img = cv2.imread(str(img_path))
        mask = cv2.imread(str(lab_path), cv2.IMREAD_GRAYSCALE)
        if img is None or mask is None:
            raise FileNotFoundError(f"No se pudo leer {img_path} o {lab_path}")
        h, w = img.shape[:2]
        polygons = mask_to_polygons(mask, dilate_px=dilate_px)
        if not polygons:
            continue  # sin instancias, no hay nada que aumentar

        for i in range(n_variants):
            aug_img, aug_polygons = _augment_once(img, polygons)
            if not aug_polygons:
                continue  # la transformación destruyó todas las instancias (raro, pero posible)
            aug_h, aug_w = aug_img.shape[:2]
            lines = polygons_to_yolo_lines(aug_polygons, aug_w, aug_h)

            stem = f"{img_path.stem}_aug{i}"
            cv2.imwrite(str(img_out / f"{stem}{img_path.suffix}"), aug_img)
            (lbl_out / f"{stem}.txt").write_text("\n".join(lines), encoding="utf-8")
            total_written += 1

    print(f"Augmentation offline: +{total_written} imágenes sintéticas en {img_out}")


if __name__ == "__main__":
    build_dataset()
