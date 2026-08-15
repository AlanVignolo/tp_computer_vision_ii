import os
from pathlib import Path
from ultralytics import YOLO

from src.unificado.data import prepare_data

def train_yolo(cfg):
    print("Iniciando entrenamiento de YOLO-Seg...")

    dataset_path = Path(cfg['dataset_path']).resolve()
    runs_dir = Path(cfg.get('runs_dir', 'runs/yolo_seg')).resolve()
    
    # El YAML que YOLO espera
    # Asumimos que data.yaml se guarda donde decida prepare_data.py
    data_yaml = dataset_path.parent / f"{dataset_path.name}_yoloseg_fullpoly" / "data.yaml"
    
    if not data_yaml.exists():
        print("data.yaml no encontrado. Convirtiendo máscaras a formato YOLO (Polígonos)...")
        # Forzar generación
        prepare_data.build_dataset(force=True, out_root=data_yaml.parent)

    print(f"Utilizando dataset YOLO: {data_yaml}")
    
    yolo_cfg = {
        'imgsz': cfg['imgsz'],
        'batch': cfg['batch_size'],
        'epochs': cfg['epochs'],
        'patience': cfg.get('patience', 30),
        'scale': cfg.get('scale', 0.5),
        'fliplr': cfg.get('fliplr', 0.5),
        'degrees': cfg.get('degrees', 0.0),
        'flipud': cfg.get('flipud', 0.0),
        'hsv_h': cfg.get('hsv_h', 0.0),
        'hsv_s': cfg.get('hsv_s', 0.0),
        'hsv_v': cfg.get('hsv_v', 0.0),
        'mixup': cfg.get('mixup', 0.0),
        'copy_paste': cfg.get('copy_paste', 0.0),
        'seed': cfg.get('seed', 42)
    }

    model = YOLO(cfg['model_name'])
    
    # YOLO maneja sus losses internamente
    model.train(
        data=str(data_yaml),
        project=str(runs_dir),
        name=cfg['run_name'],
        **yolo_cfg
    )

    best_model_path = runs_dir / cfg['run_name'] / "weights" / "best.pt"
    print(f"Entrenamiento finalizado. Mejor modelo: {best_model_path}")
    return str(best_model_path)
