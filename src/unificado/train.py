import argparse
import yaml
from pathlib import Path
import sys
import os

# Asegurar import de modulos hermanos
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.unificado.trainers.train_pidnet import train_pidnet
from src.unificado.trainers.train_yolo import train_yolo

def load_yaml(file_path):
    with open(file_path, 'r') as f:
        return yaml.safe_load(f)

def main():
    parser = argparse.ArgumentParser(description="Script unificado de entrenamiento")
    parser.add_argument("--config", type=str, required=True, help="Ruta al archivo yaml de configuración")
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    model_type = cfg.get('model_type')

    if model_type == 'pidnet':
        train_pidnet(cfg)
    elif model_type == 'yolo_seg':
        train_yolo(cfg)
    else:
        raise ValueError(f"model_type '{model_type}' no soportado.")

if __name__ == "__main__":
    main()
