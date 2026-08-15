import os
import cv2
import numpy as np

def semantic_iou(pred_mask, gt_mask):
    """IoU a nivel píxel entre dos máscaras binarias. NaN si ambas vacías."""
    pred_mask = np.squeeze(pred_mask)
    gt_mask = np.squeeze(gt_mask)
    inter = np.logical_and(pred_mask, gt_mask).sum()
    union = np.logical_or(pred_mask, gt_mask).sum()
    return float("nan") if union == 0 else inter / union

def evaluate_semantic_iou(predict_fn, test_images_dir, test_masks_dir, conf=None):
    """
    Evaluador universal que calcula el mIoU semántico.
    
    predict_fn: callback function(image_path, conf) -> np.array de shape (H, W) binario.
    test_images_dir: directorio con las imágenes de prueba (.jpg)
    test_masks_dir: directorio con las máscaras ground truth (.png)
    conf: umbral de confianza (opcional) que necesite predict_fn.
    
    Retorna: diccionario con 'mean', 'median', e iou 'per_image'
    """
    per_image = []
    
    test_files = sorted([f for f in os.listdir(test_images_dir) if f.endswith('.jpg')])
    
    for img_name in test_files:
        img_path = os.path.join(test_images_dir, img_name)
        mask_name = os.path.splitext(img_name)[0] + '.png'
        mask_path = os.path.join(test_masks_dir, mask_name)
        
        gt = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if gt is None:
            continue
            
        gt_mask = np.squeeze(gt) > 127
        
        # Llamada agnóstica al modelo
        if conf is not None:
            pred = predict_fn(img_path, conf=conf)
        else:
            pred = predict_fn(img_path)
            
        per_image.append((img_path, semantic_iou(pred, gt_mask)))
        
    ious = np.array([i for _, i in per_image])
    valid = ~np.isnan(ious)
    
    return {
        "mean": float(ious[valid].mean()) if valid.any() else 0.0,
        "median": float(np.median(ious[valid])) if valid.any() else 0.0,
        "per_image": per_image
    }
