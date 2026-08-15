import os
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

from src.unificado.data.datasets import DeepCrackDataset, JointTransform
from src.unificado.utils.pidnet_losses import get_pidnet_losses
from src.unificado.models.pidnet_model import build_custom_pidnet

def compute_miou(preds, targets, num_classes=2):
    pred_labels = preds.argmax(dim=1)
    pred_flat = pred_labels.view(-1)
    target_flat = targets.view(-1)

    valid = target_flat != 255
    pred_flat = pred_flat[valid]
    target_flat = target_flat[valid]

    indices = num_classes * target_flat + pred_flat
    conf_matrix = torch.bincount(indices, minlength=num_classes ** 2)
    conf_matrix = conf_matrix.reshape(num_classes, num_classes)
    return conf_matrix

def miou_from_conf_matrix(conf_matrix):
    tp = conf_matrix.diag()
    fp = conf_matrix.sum(dim=0) - tp
    fn = conf_matrix.sum(dim=1) - tp
    denom = tp + fp + fn

    valid_classes = denom > 0
    iou_per_class = torch.zeros_like(tp, dtype=torch.float32)
    iou_per_class[valid_classes] = tp[valid_classes].float() / denom[valid_classes].float()

    miou = iou_per_class[valid_classes].mean().item()
    return miou, iou_per_class.tolist()

def run_epoch_pidnet(model, loader, device, criterion_semantic, criterion_boundary, criterion_boundary_aware, optimizer=None):
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    total_loss = 0.0
    conf_matrix = torch.zeros(2, 2, dtype=torch.long, device=device)

    ctx = torch.enable_grad() if is_train else torch.no_grad()
    with ctx:
        for images, masks, boundaries in loader:
            images = images.to(device)
            masks = masks.to(device)
            boundaries = boundaries.to(device).unsqueeze(1)

            if is_train:
                optimizer.zero_grad()

            outputs = model(images)
            pred_p, pred_final, pred_d = outputs[0], outputs[1], outputs[2]

            h, w = masks.shape[1], masks.shape[2]
            pred_p = F.interpolate(pred_p, size=(h, w), mode='bilinear', align_corners=True)
            pred_final = F.interpolate(pred_final, size=(h, w), mode='bilinear', align_corners=True)
            pred_d = F.interpolate(pred_d, size=(h, w), mode='bilinear', align_corners=True)

            loss_0 = criterion_semantic(pred_p, masks)
            loss_1 = criterion_boundary(pred_d, boundaries)
            loss_2 = criterion_semantic(pred_final, masks)
            loss_3 = criterion_boundary_aware(pred_final, torch.sigmoid(pred_d), masks)

            loss_total = (0.4 * loss_0) + (20.0 * loss_1) + (1.0 * loss_2) + (1.0 * loss_3)

            if is_train:
                loss_total.backward()
                optimizer.step()

            total_loss += loss_total.item()
            conf_matrix += compute_miou(pred_final.detach(), masks, 2)

    avg_loss = total_loss / len(loader)
    miou, iou_classes = miou_from_conf_matrix(conf_matrix)
    return avg_loss, miou, iou_classes

def train_pidnet(cfg):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[{device}] Iniciando entrenamiento de PIDNet...")

    # DataLoaders
    train_transform = JointTransform(
        scale_limit=cfg.get('scale_limit', [0.5, 2.0]), 
        crop_size=cfg.get('crop_size', [512, 512]), 
        hflip_p=cfg.get('hflip_p', 0.5)
    )
    
    train_dataset = DeepCrackDataset(root_dir=cfg['dataset_path'], split='train', transform=train_transform)
    train_loader = DataLoader(train_dataset, batch_size=cfg['batch_size'], shuffle=True, num_workers=cfg['num_workers'], drop_last=True)
    
    val_dataset = DeepCrackDataset(root_dir=cfg['dataset_path'], split='test', transform=None)
    val_loader = DataLoader(val_dataset, batch_size=cfg['batch_size'], shuffle=False, num_workers=cfg['num_workers'])

    # Modelo
    model = build_custom_pidnet(pretrained_path=cfg['pretrained_path'], num_classes=cfg['num_classes'], variant=cfg['variant'])
    model.to(device)

    # Losses expecíficas de PIDNet
    crit_sem, crit_bound, crit_bound_aw = get_pidnet_losses()

    # Optimizador y Scheduler
    optimizer = optim.SGD(model.parameters(), lr=cfg['learning_rate'], momentum=cfg['momentum'], weight_decay=cfg['weight_decay'], nesterov=True)
    
    max_epochs = cfg['epochs']
    def poly_lr_scheduler(epoch):
        return (1 - epoch / max_epochs) ** 0.9
    scheduler = LambdaLR(optimizer, lr_lambda=poly_lr_scheduler)

    best_val_miou = 0.0
    patience = cfg.get('patience', 10)
    patience_counter = 0

    os.makedirs('runs/pidnet', exist_ok=True)
    save_path = f"runs/pidnet/best_pidnet_{cfg['variant']}.pth"

    for epoch in range(max_epochs):
        train_loss, train_miou, _ = run_epoch_pidnet(
            model, train_loader, device, crit_sem, crit_bound, crit_bound_aw, optimizer=optimizer
        )

        val_loss, val_miou, val_iou_classes = run_epoch_pidnet(
            model, val_loader, device, crit_sem, crit_bound, crit_bound_aw, optimizer=None
        )

        print(f"Epoch {epoch+1}/{max_epochs} | Train Loss: {train_loss:.4f} mIoU: {train_miou:.4f} | Val Loss: {val_loss:.4f} mIoU: {val_miou:.4f}")

        if val_miou > best_val_miou:
            best_val_miou = val_miou
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'val_miou': val_miou,
            }, save_path)
            print(f"  -> Mejor modelo guardado en {save_path}")
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"Early stopping activado en la época {epoch+1}")
            break

        scheduler.step()

    print("Entrenamiento finalizado.")
    return save_path
