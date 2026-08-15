# Detección de grietas en superficies de infraestructura civil

Trabajo Práctico Final — **Visión por Computadora II** (CEIA, FIUBA) — Grupo 5.

Prototipo de detección de grietas en imágenes de superficies mediante deep learning
(YOLOv11), orientado a automatizar la inspección de infraestructura civil.

**Integrantes:** Agustín Biancardi · Gabriel Quiroga · Alan Vignolo

## Instalación

Requiere [uv](https://docs.astral.sh/uv/) y Python 3.12.

```bash
uv sync            # crea el .venv e instala dependencias desde uv.lock
```

## Datasets

Los datasets **no se versionan en el repo** (~112 MB, ~1k archivos). Se descargan con
`src/download_data.py`, que es idempotente (no re-descarga si ya están preparados).

### Segmentación — `dataset_1000` (DeepCrack)

Dataset opcional para experimentos de segmentación (máscaras binarias). Fuente:
[Kaggle — DeepCrack](https://www.kaggle.com/datasets/rukiyeaydn/deepcrack-dataset).
Requiere credenciales de Kaggle (`~/.kaggle/kaggle.json`).

```bash
uv run python src/download_data.py --seg
```

## Estructura

```
tp_computer_vision_ii/
├── src/
│   ├── data_analysis.ipynb   # EDA de los datasets
│   └── download_data.py      # descarga + preparación de datasets
├── datasets/                 # (ignorado por git — ver arriba)
├── doc/                      # plantilla IEEE del paper
├── pyproject.toml
└── uv.lock
```

## Análisis exploratorio

El notebook [`src/data_analysis.ipynb`](src/data_analysis.ipynb) analiza volumen,
densidad de grietas por imagen, distribución de tamaños y niveles de severidad. Usar el
kernel **"Python (CEIA - Vision II TP)"**.

## Framework Unificado de Segmentación (PIDNet & YOLO-Seg)

El proyecto cuenta con un framework unificado ubicado en `src/unificado/` para el entrenamiento y la evaluación de modelos de segmentación semántica (PIDNet) y de instancias (YOLOv11), utilizando un estándar común sin perder el control granular de cada arquitectura.

El entrenamiento y la evaluación se manejan mediante archivos de configuración `YAML`.

### 1. Pesos Preentrenados

- **PIDNet**: Descarga los pesos `PIDNet_S_Cityscapes_test.pt` desde el [repositorio original](https://drive.google.com/drive/folders/0BySIOtxxULinfjlGdGFiT3NQVUdLVDBxWnhhTjB4VXNBRkFOa281WHlkektYY2VBcWVZb1k?resourcekey=0-w0JIXUekD-FCW-Rm1Z-HfQ) y colócalos en `src/pidnet/pretrained_models/`. 
- **YOLOv11**: Los pesos base (`yolo11n-seg.pt`) se descargarán automáticamente la primera vez que inicies un entrenamiento.

### 2. Entrenamiento (`train.py`)

El entrypoint leerá el archivo de configuración y decidirá qué arquitectura usar. Cada modelo utiliza sus funciones de pérdida designadas (PIDNet usa su propia `BoundaryAwareCrossEntropy`, YOLO usa las internas de Ultralytics).

**Entrenar PIDNet:**
```bash
uv run python src/unificado/train.py --config configs/pidnet.yaml
```

**Entrenar YOLO-Seg:**
```bash
uv run python src/unificado/train.py --config configs/yolo_seg.yaml
```

### 3. Evaluación Unificada (`eval.py`)

La evaluación es estrictamente agnóstica al modelo. Calcula el **Mean IoU** y **Median IoU** semántico siempre contra las máscaras originales (`.png`) para evitar los techos de precisión asociados a las conversiones.

**Evaluar PIDNet:**
```bash
uv run python src/unificado/eval.py \
    --config configs/pidnet.yaml \
    --weights runs/pidnet/best_pidnet_S.pth
```

**Evaluar YOLO-Seg:**
```bash
uv run python src/unificado/eval.py \
    --config configs/yolo_seg.yaml \
    --weights runs/yolo_seg/yolo11n_seg_nano/weights/best.pt \
    --conf 0.10
```

### 4. Inferencia en Video (`infer_video.py`)

Para aplicar la segmentación a un archivo de video y visualizar los resultados con las grietas resaltadas, se utiliza el script de inferencia unificada. Al finalizar, reportará la cantidad total de frames procesados, el tiempo transcurrido y la velocidad media en FPS.

**Ejemplo (PIDNet):**
```bash
uv run python src/unificado/infer_video.py \
    --config configs/pidnet.yaml \
    --weights runs/pidnet/best_pidnet_S.pth \
    --input datasets/videos/video_cracks.mp4 \
    --output datasets/videos/video_cracks_pidnet.mp4
```

**Ejemplo (YOLO-Seg):**
```bash
uv run python src/unificado/infer_video.py \
    --config configs/yolo_seg.yaml \
    --weights runs/yolo_seg/yolo11n_seg_nano/weights/best.pt \
    --input datasets/videos/video_cracks.mp4 \
    --output datasets/videos/video_cracks_yolo.mp4 \
    --conf 0.25
```
