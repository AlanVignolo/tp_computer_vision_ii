import torch
import torch.nn as nn
import torch.nn.functional as F

class BoundaryAwareCrossEntropy(nn.Module):
    def __init__(self, gamma=1.0):
        super(BoundaryAwareCrossEntropy, self).__init__()
        self.gamma = gamma

    def forward(self, semantic_preds, boundary_preds, targets):
        """
        semantic_preds: Probabilidades Softmax (s_head2) [B, C, H, W]
        boundary_preds: Activaciones Sigmoide (d_head) [B, 1, H, W]
        targets: Etiquetas manuales [B, H, W]
        """
        # Calcular entropía cruzada estándar sin reducción
        ce_loss = F.cross_entropy(semantic_preds, targets, reduction='none')

        # Escalar la pérdida espacialmente con las predicciones de frontera
        # (1 + gamma * p_boundary)
        boundary_weight = 1.0 + (self.gamma * boundary_preds.squeeze(1))

        # Aplicar el castigo ponderado
        weighted_ce = ce_loss * boundary_weight

        return weighted_ce.mean()

def get_pidnet_losses():
    """
    Retorna las 3 funciones de pérdida usadas en PIDNet:
    l_0 y l_2: semantic (CrossEntropy)
    l_1: boundary (BCE)
    l_3: boundary aware (Custom)
    """
    criterion_semantic = nn.CrossEntropyLoss(ignore_index=255)
    criterion_boundary = nn.BCEWithLogitsLoss()
    criterion_boundary_aware = BoundaryAwareCrossEntropy()
    return criterion_semantic, criterion_boundary, criterion_boundary_aware
