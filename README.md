# Detección y Segmentación de Grietas en Superficies de Infraestructura Civil

Trabajo Práctico Final — **Visión por Computadora II** (CEIA, FIUBA) — Grupo 5.

Prototipo de detección y segmentación de grietas en imágenes y secuencias de video mediante Deep Learning, orientado a automatizar la inspección estructural de infraestructura civil (pavimento, puentes y concreto).

**Integrantes:** Agustín Biancardi · Gabriel Quiroga · Alan Vignolo

---

## 1. Instalación y Prerrequisitos Generales

El proyecto gestiona su entorno y dependencias mediante [uv](https://docs.astral.sh/uv/) sobre **Python 3.12**.

```bash
# Clonar el repositorio
git clone https://github.com/AlanVignolo/tp_computer_vision_ii.git
cd tp_computer_vision_ii

# Sincronizar el entorno virtual e instalar dependencias
uv sync
```

> [!TIP]
> Para el entrenamiento e inferencia en video se recomienda el uso de una GPU compatible con CUDA (NVIDIA). Los notebooks también están preparados para ejecutarse directamente en **Google Colab** (GPU T4/V100/A100).

---

## 2. Datasets

Los datasets no se versionan en el repositorio debido a su tamaño (~112 MB, ~1k imágenes). Se descargan y preparan automáticamente con el script `src/download_data.py` (idempotente):

### Segmentación — `dataset_1000` (DeepCrack)
Dataset de segmentación semántica con máscaras binarias anotadas a nivel de píxel. Fuente: [Kaggle — DeepCrack Dataset](https://www.kaggle.com/datasets/rukiyeaydn/deepcrack-dataset).

Requiere credenciales de Kaggle configuradas en `~/.kaggle/kaggle.json`.

```bash
uv run python src/download_data.py --seg
```

El dataset quedará organizado en `datasets/dataset_1000/`:
- `train_img/` y `train_lab/` (Imágenes y máscaras de entrenamiento)
- `test_img/` y `test_lab/` (Imágenes y máscaras de prueba)

---

## 3. Estructura del Proyecto

```
tp_computer_vision_ii/
├── src/
│   ├── data_analysis.ipynb           # Análisis exploratorio de datos (EDA)
│   ├── download_data.py              # Script de descarga y preparación de datasets
│   ├── pidnet/                       # Módulo PIDNet (Segmentación en Tiempo Real)
│   │   ├── pidnet.ipynb              # Entrenamiento, validación e inferencia de imágenes
│   │   ├── infer_video_pidnet.ipynb  # Inferencia y segmentación de video optimizada
│   │   ├── models/                   # Arquitectura PIDNet, PagFM, DAPPM/PAPPM, Bag
│   │   ├── pretrained_models/        # Backbones preentrenados de Cityscapes (.pt)
│   │   └── best_trained/             # Checkpoints de los mejores modelos entrenados (.pth)
│   ├── yolo_seg/                     # Módulo YOLO-Seg (Segmentación de Instancias)
│   │   ├── 01_check_labels.ipynb
│   │   ├── 02_train.ipynb
│   │   ├── 03_eval.ipynb
│   │   └── prepare_data.py
│   └── unet/                         # Módulo U-Net / U-Net++
│       └── models/                   # Notebooks de arquitecturas U-Net++ con backbones
├── datasets/                         # Datasets e imágenes/videos (ignorado en git)
│   ├── dataset_1000/
│   └── videos/
├── configs/                          # Archivos de configuración YAML
├── doc/                              # Documentación y plantilla IEEE del paper
├── pyproject.toml
└── uv.lock
```

---

## 4. Análisis Exploratorio de Datos (EDA)

El notebook [`src/data_analysis.ipynb`](src/data_analysis.ipynb) analiza el volumen de datos, la densidad de grietas por imagen, la distribución de tamaños y los niveles de severidad de defectos.

---

## 5. PIDNet: Segmentación Semántica en Tiempo Real

**PIDNet** (*Proportional-Integral-Derivative Network*, CVPR 2023) es una arquitectura de tres ramas diseñada para segmentación semántica en tiempo real. Utiliza el concepto de un controlador PID para mitigar el sobrepaso (*overshoot*) en los límites de los objetos mediante una rama derivativa (**D**) dedicada a preservar bordes de alta frecuencia, ideal para la elevada relación perímetro/área de las grietas.

### 5.1 Prerrequisitos de PIDNet
1. **Pesos Pre-entrenados**:
   Descarga los pesos base de Cityscapes `PIDNet_S_Cityscapes_test.pt` (o variantes `M`/`L`) desde el [Google Drive oficial de PIDNet](https://drive.google.com/drive/folders/0BySIOtxxULinfld0LTcxYndTbFpWNjVpWm9nREU1T3hJUW5IS2otOUJDMmtnZERuODFPVU0?resourcekey=0-nauDQNE1efkunvcg89ZlDA) y colócalos en `src/pidnet/pretrained_models/`.
2. **Generación de Bordes on-the-fly**:
   El dataset genera automáticamente mapas de borde binarios con el operador de Canny y dilatación morfológica para supervisar la rama derivativa.

### 5.2 Entrenamiento y Evaluación ([`src/pidnet/pidnet.ipynb`](src/pidnet/pidnet.ipynb))
- **Aumentación de datos especializada**: Flips horizontales/verticales, rotaciones leves y distorsión fotométrica.
- **Funciones de pérdida compuestas**:
  $$\mathcal{L}_{total} = 0.4 \ell_0 + 20.0 \ell_1 + 1.0 \ell_2 + 1.0 \ell_3$$
  - $\ell_0, \ell_2$: Cross-Entropy semántica.
  - $\ell_1$: Weighted BCE (`pos_weight=15.0`) para compensar el desbalance de píxeles de frontera.
  - $\ell_3$: Boundary-Aware Cross-Entropy.
- **Métricas de evaluación**: Cálculo desglosado de **Crack IoU**, **mIoU**, **Precision**, **Recall** y **F1-Score / Dice**.
- **Aceleración**: Soporte nativo de *Automatic Mixed Precision* (AMP FP16).
- **Inspección visual**: Generación automática de figuras de validación $2 \times 4$ (Original, GT, Canny Edges y Superposición).

### 5.3 Inferencia y Segmentación de Video ([`src/pidnet/infer_video_pidnet.ipynb`](src/pidnet/infer_video_pidnet.ipynb))
Pipeline optimizado para el procesamiento y análisis de secuencias de video:
- **Configuración centralizada**: Rutas de checkpoint, video de entrada y video de salida editables al inicio del notebook.
- **Preservación de Aspect Ratio**: Redimensionado adaptativo a múltiplos de 32 (ej. 16:9 $\rightarrow 800 \times 448$) sin distorsión geométrica.
- **Procesamiento por Lotes (Batching)**: Aceleración por GPU mediante `BATCH_SIZE` configurable, `torch.inference_mode()` y AMP FP16.
- **Superposición Vectorizada y Telemetría HUD**: Overlay translúcido con contornos vectoriales (`cv2.drawContours`) y telemetría en tiempo real (FPS instantáneo, detección y porcentaje de severidad/área).
- **Timeline de Daño y Keyframes**: Gráfico continuo de porcentaje de área afectada a lo largo de la línea temporal del video y reproductor HTML5 integrado.

---

## 6. YOLO-Seg: Segmentación de Instancias (YOLOv11)

> *Sección en desarrollo / Placeholder*

Módulo para la detección y segmentación de grietas como instancias individuales mediante **YOLOv11-Seg** (Ultralytics).

- **Inspección de etiquetas**: [`src/yolo_seg/01_check_labels.ipynb`](src/yolo_seg/01_check_labels.ipynb)
- **Entrenamiento**: [`src/yolo_seg/02_train.ipynb`](src/yolo_seg/02_train.ipynb)
- **Evaluación**: [`src/yolo_seg/03_eval.ipynb`](src/yolo_seg/03_eval.ipynb)

---

## 7. U-Net / U-Net++: Segmentación Basada en Encoders Pre-entrenados

> *Sección en desarrollo / Placeholder*

Módulo para la comparación de arquitecturas clásicas de segmentación basadas en **U-Net** y **U-Net++** con múltiples backbones (ResNet34, EfficientNet-B0) y mecanismos de atención espacial y de canal (SCSE).

- **Notebooks de experimentación**: [`src/unet/models/`](src/unet/models/)
