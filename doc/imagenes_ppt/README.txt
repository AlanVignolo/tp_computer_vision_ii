IMÁGENES PARA EL POWERPOINT — TP Segmentación de grietas
=========================================================

Cada archivo está nombrado con el número de diapositiva donde va.
El prefijo NN_ indica la diapo destino.

DIAPO 4  — Dataset
  04_dataset_muestras.png
    Grid de 6 muestras de DeepCrack (imagen arriba, máscara abajo).

DIAPO 6  — Metodología
  06_metodologia_pipeline.png
    Diagrama del pipeline: DeepCrack -> split común -> 3 modelos -> métricas.

DIAPO 10 — YOLO / Entrenamiento
  10_yolo_entrenamiento_nano.png    (curvas de entrenamiento YOLO11n)
  10_yolo_entrenamiento_small.png   (curvas de entrenamiento YOLO11s)
    Poner una o las dos; muestran loss y métricas por época.

DIAPO 11 — YOLO / Resultados
  11_yolo_resultados_muestras.png
    Muestras cualitativas (peor / mediana / mejor IoU): imagen, GT, predicción.

DIAPO 12 — YOLO / Mejoras
  12_yolo_mejoras_techo_poligonos.png
    Techo de IoU por conversión máscara->polígono (simplificada vs completa).
  12_yolo_mejoras_hires_curvas.png
    Curvas del modelo Hi-Res (imgsz=768), la mejora por resolución.

DIAPO 13 — PIDNet / Entrenamiento
  13_pidnet_entrenamiento_curvas.png
    Curvas de loss e IoU de PIDNet-S.

DIAPO 14 — PIDNet / Resultados
  14_pidnet_resultados_muestras.png
    Muestras cualitativas de PIDNet: imagen, GT, predicción.

--------------------------------------------------------------
PENDIENTE (las tiene que exportar Agustín):
  - DIAPO 7/8  U-Net / Entrenamiento y Resultados: faltan las figuras
    de curvas y muestras cualitativas de U-Net++.

Diapos sin imagen (texto/tabla): 1, 2, 3 (ya tiene), 5, 9, 15, 16, 17, 18, 19.
