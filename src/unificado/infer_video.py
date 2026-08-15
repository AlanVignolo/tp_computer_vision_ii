import argparse
import os
import sys
import time
import yaml
from pathlib import Path
import cv2
import numpy as np
import torch
import torch.nn.functional as F

# Compatible con ejecución como script y como notebook (Colab/Jupyter)
try:
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
except NameError:
    sys.path.append(os.path.abspath('.'))

def load_yaml(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def get_pidnet_target_size(orig_h, orig_w, target_h=448):
    """
    Calcula (target_h, target_w) preservando la relación de aspecto original
    tal que ambas dimensiones sean múltiplos de 32 (requerimiento de PIDNet).
    """
    scale = target_h / float(orig_h)
    target_w = int(round(orig_w * scale))
    
    h_32 = max(32, int(round(target_h / 32.0)) * 32)
    w_32 = max(32, int(round(target_w / 32.0)) * 32)
    return h_32, w_32

def overlay_mask(frame_bgr, mask, color=(0, 0, 255), alpha=0.45):
    """
    Superpone la máscara binaria (True = grieta) sobre el frame original.
    color: tupla BGR (rojo por defecto: (0, 0, 255)).
    """
    if not np.any(mask):
        return frame_bgr
    overlay = frame_bgr.copy()
    overlay[mask] = color
    return cv2.addWeighted(overlay, alpha, frame_bgr, 1.0 - alpha, 0)

def main():
    parser = argparse.ArgumentParser(description="Inferencia unificada de video para segmentación de grietas (PIDNet & YOLO-Seg)")
    parser.add_argument("--config", type=str, required=True, help="Ruta al archivo YAML de configuración")
    parser.add_argument("--weights", type=str, required=True, help="Ruta a los pesos del modelo (.pt o .pth)")
    parser.add_argument("--input", type=str, required=True, help="Ruta al video de entrada")
    parser.add_argument("--output", type=str, required=True, help="Ruta al video de salida (.mp4)")
    parser.add_argument("--conf", type=float, default=0.10, help="Umbral de confianza para YOLO-Seg / Softmax threshold (default: 0.10)")
    parser.add_argument("--target-height", type=int, default=448, help="Altura objetivo para PIDNet preservando aspecto (múltiplo de 32, default: 448)")
    parser.add_argument("--imgsz", type=int, default=640, help="Tamaño de imagen para YOLO-Seg (default: 640)")
    parser.add_argument("--alpha", type=float, default=0.45, help="Opacidad del overlay de segmentación (0 a 1, default: 0.45)")
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    model_type = cfg.get('model_type')

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Dispositivo de cómputo: {device}")
    print(f"Cargando modelo '{model_type}' con pesos: {args.weights}")

    if model_type == 'pidnet':
        from src.unificado.models.pidnet_model import build_custom_pidnet

        num_classes = cfg.get('num_classes', 2)
        variant = cfg.get('variant', 'S')
        model = build_custom_pidnet(pretrained_path=args.weights, num_classes=num_classes, variant=variant)
        model.to(device)
        model.eval()

        mean_val = torch.tensor((0.485, 0.456, 0.406), device=device, dtype=torch.float32).view(3, 1, 1)
        std_val = torch.tensor((0.229, 0.224, 0.225), device=device, dtype=torch.float32).view(3, 1, 1)

        def predict_pidnet(frame_bgr):
            orig_h, orig_w = frame_bgr.shape[:2]
            target_h, target_w = get_pidnet_target_size(orig_h, orig_w, target_h=args.target_height)

            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            frame_resized = cv2.resize(frame_rgb, (target_w, target_h), interpolation=cv2.INTER_LINEAR)

            tensor = torch.from_numpy(frame_resized).permute(2, 0, 1).float().to(device) / 255.0
            tensor = (tensor - mean_val) / std_val
            input_tensor = tensor.unsqueeze(0)

            t0 = time.perf_counter()
            with torch.no_grad():
                outputs = model(input_tensor)
                pred_final = outputs[1]
                pred_final = F.interpolate(pred_final, size=(orig_h, orig_w), mode='bilinear', align_corners=True)
                if num_classes == 2:
                    probs = F.softmax(pred_final, dim=1)[:, 1]
                    pred_mask = (probs > 0.5).squeeze(0).cpu().numpy()
                else:
                    pred_mask = pred_final.argmax(dim=1).squeeze(0).cpu().numpy().astype(bool)
            infer_time = time.perf_counter() - t0

            return pred_mask, infer_time

        predict_fn = predict_pidnet

    elif model_type == 'yolo_seg':
        from ultralytics import YOLO
        model = YOLO(args.weights)
        imgsz = args.imgsz or cfg.get('imgsz', 640)

        def predict_yolo(frame_bgr):
            h, w = frame_bgr.shape[:2]
            t0 = time.perf_counter()
            result = model.predict(frame_bgr, conf=args.conf, imgsz=imgsz, retina_masks=True, verbose=False)[0]
            infer_time = time.perf_counter() - t0

            semantic = np.zeros((h, w), dtype=bool)
            if result.masks is not None:
                for m in result.masks.data.cpu().numpy():
                    m = np.squeeze(m)
                    if m.shape != (h, w):
                        m = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)
                    semantic |= (m > 0.5)
            return semantic, infer_time

        predict_fn = predict_yolo

    else:
        raise ValueError(f"model_type '{model_type}' no reconocido en {args.config}")

    print(f"Abriendo video de entrada: {args.input}")
    cap = cv2.VideoCapture(args.input)
    if not cap.isOpened():
        raise IOError(f"No se pudo abrir el video: {args.input}")

    fps_in = cap.get(cv2.CAP_PROP_FPS)
    if fps_in <= 0 or np.isnan(fps_in):
        fps_in = 25.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    out_dir = os.path.dirname(os.path.abspath(args.output))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(args.output, fourcc, fps_in, (w, h))

    print(f"Procesando video... Resolución: {w}x{h}, FPS entrada: {fps_in:.2f}, Total frames: {total_frames}")

    frame_count = 0
    total_infer_time = 0.0
    t_start = time.perf_counter()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        pred_mask, infer_time = predict_fn(frame)
        total_infer_time += infer_time

        frame_out = overlay_mask(frame, pred_mask, color=(0, 0, 255), alpha=args.alpha)
        writer.write(frame_out)
        frame_count += 1

        if frame_count % 50 == 0 or frame_count == total_frames:
            elapsed = time.perf_counter() - t_start
            current_fps = frame_count / elapsed if elapsed > 0 else 0
            print(f"Progreso: {frame_count}/{total_frames} frames ({frame_count/max(1, total_frames)*100:.1f}%) - {current_fps:.1f} FPS")

    t_end = time.perf_counter()
    cap.release()
    writer.release()

    total_time = t_end - t_start
    avg_pipeline_fps = frame_count / total_time if total_time > 0 else 0
    avg_infer_fps = frame_count / total_infer_time if total_infer_time > 0 else 0

    print("\n=======================================================")
    print("           RESUMEN DE PROCESAMIENTO DE VIDEO           ")
    print("=======================================================")
    print(f" Modelo utilizado           : {model_type}")
    print(f" Total imágenes (frames)    : {frame_count}")
    print(f" Tiempo total pipeline      : {total_time:.2f} segundos")
    print(f" Tiempo neto de inferencia  : {total_infer_time:.2f} segundos")
    print(f" FPS pipeline completo      : {avg_pipeline_fps:.2f} FPS")
    print(f" FPS solo inferencia (red)  : {avg_infer_fps:.2f} FPS")
    print(f" Video resultante guardado  : {args.output}")
    print("=======================================================\n")

if __name__ == "__main__":
    main()
