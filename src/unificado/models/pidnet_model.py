import torch
import sys
import os

from src.unificado.models.pidnet import PIDNet

def build_custom_pidnet(pretrained_path='PIDNet_M_Cityscapes_test.pt', num_classes=2, variant='S'):
    """
    Construye una PIDNet evitando usar el sistema yacs/cfg del repositorio original y carga
    los pesos pre-entrenados de Cityscapes y los adapta a num_classes.
    """
    if variant == 'S':
        model = PIDNet(m=2, n=3, num_classes=num_classes, planes=32, ppm_planes=96, head_planes=128, augment=True)
    elif variant == 'M':
        model = PIDNet(m=2, n=3, num_classes=num_classes, planes=64, ppm_planes=96, head_planes=128, augment=True)
    elif variant == 'L':
        model = PIDNet(m=3, n=4, num_classes=num_classes, planes=64, ppm_planes=112, head_planes=256, augment=True)
    else:
        raise ValueError("Variante no soportada")

    if not os.path.exists(pretrained_path):
        print(f"ADVERTENCIA: Archivo de pesos {pretrained_path} no encontrado. Instanciando modelo desde cero.")
        return model

    print(f"Cargando pesos desde {pretrained_path}...")
    pretrained_dict = torch.load(pretrained_path, map_location='cpu')

    # Manejo de diccionarios si el modelo fue guardado con DataParallel
    if 'state_dict' in pretrained_dict:
        pretrained_dict = pretrained_dict['state_dict']
    pretrained_dict = {k.replace('module.', '').replace('model.', ''): v for k, v in pretrained_dict.items()}

    # Eliminar capas que predecían N clases para reinicializarlas para num_classes
    layers_to_reset = [
        'final_layer.conv2.weight', 'final_layer.conv2.bias',
        'seghead_p.conv2.weight', 'seghead_p.conv2.bias'
    ]
    
    for layer in layers_to_reset:
        if layer in pretrained_dict:
            del pretrained_dict[layer]

    missing_keys, unexpected_keys = model.load_state_dict(pretrained_dict, strict=False)
    
    print("\n--- Diagnóstico de Carga ---")
    print(f"Capas reinicializadas aleatoriamente para {num_classes} clases: {len(missing_keys)}")
    return model
