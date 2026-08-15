import argparse
import yaml
import os
import torch
import cv2
import numpy as np
from pathlib import Path
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.unificado.utils.evaluator import evaluate_semantic_iou

def load_yaml(file_path):
    with open(file_path, 'r') as f:
        return yaml.safe_load(f)

def main():
    parser = argparse.ArgumentParser(description="Script unificado de evaluación")
    parser.add_argument("--config", type=str, required=True, help="Ruta al archivo yaml de configuración")
    parser.add_argument("--weights", type=str, required=True, help="Ruta al modelo preentrenado (.pt o .pth)")
    parser.add_argument("--conf", type=float, default=0.10, help="Umbral de confianza (Solo para YOLO)")
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    model_type = cfg.get('model_type')
    
    dataset_path = Path(cfg['dataset_path']).resolve()
    test_images_dir = dataset_path / "test_img"
    test_masks_dir = dataset_path / "test_lab"
    
    if not test_images_dir.exists() or not test_masks_dir.exists():
        raise FileNotFoundError(f"No se encontraron las carpetas de test en {dataset_path}")

    print(f"Evaluando {model_type} con pesos {args.weights}...")

    if model_type == 'pidnet':
        from src.unificado.models.pidnet_model import build_custom_pidnet
        import torch.nn.functional as F

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = build_custom_pidnet(pretrained_path=args.weights, num_classes=cfg['num_classes'], variant=cfg['variant'])
        model.to(device)
        model.eval()

        mean_val = np.array((0.485, 0.456, 0.406), dtype=np.float32)
        std_val = np.array((0.229, 0.224, 0.225), dtype=np.float32)

        def predict_pidnet(img_path, conf=None):
            image = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
            orig_h, orig_w = image.shape[:2]
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = cv2.resize(image, (512, 512), interpolation=cv2.INTER_LINEAR)
            
            image = (image.astype(np.float32) / 255.0 - mean_val) / std_val
            image = torch.from_numpy(image).permute(2, 0, 1).float().unsqueeze(0).to(device)
            
            with torch.no_grad():
                outputs = model(image)
                pred_final = outputs[1]
                pred_final = F.interpolate(pred_final, size=(orig_h, orig_w), mode='bilinear', align_corners=True)
                pred_mask = pred_final.argmax(dim=1).squeeze(0).cpu().numpy()
                
            return pred_mask

        predict_fn = predict_pidnet

    elif model_type == 'yolo_seg':
        from ultralytics import YOLO
        model = YOLO(args.weights)
        imgsz = cfg.get('imgsz', 512)

        def predict_yolo(img_path, conf=0.10):
            result = model.predict(str(img_path), conf=conf, imgsz=imgsz, retina_masks=True, verbose=False)[0]
            h, w = result.orig_shape
            semantic = np.zeros((h, w), dtype=bool)
            if result.masks is not None:
                for m in result.masks.data.cpu().numpy():
                    m = np.squeeze(m)
                    if m.shape != (h, w):
                        m = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)
                    semantic |= m > 0.5
            return semantic

        predict_fn = lambda img, conf=args.conf: predict_yolo(img, conf)

    else:
        raise ValueError(f"model_type '{model_type}' no soportado.")

    results = evaluate_semantic_iou(predict_fn, test_images_dir, test_masks_dir, conf=args.conf)
    
    print(f"\nResultados de Evaluación ({model_type}):")
    print(f"Mean IoU Semántico: {results['mean']:.4f}")
    print(f"Median IoU Semántico: {results['median']:.4f}")

if __name__ == "__main__":
    main()
