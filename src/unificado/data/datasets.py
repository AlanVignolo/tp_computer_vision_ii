import os
import cv2
import torch
import numpy as np
import random
from torch.utils.data import Dataset

class JointTransform:
    """
    Aplica las mismas transformaciones geométricas a las imágenes y las máscaras.
    Los bordes deben generarse (con Canny) después de aplicar las transformaciones sobre la máscara.
    """
    def __init__(
        self, 
        scale_limit=(0.5, 2.0),
        crop_size=(512, 512),
        hflip_p=0.5,
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225)
    ):
        self.scale_limit = scale_limit
        self.crop_h, self.crop_w = crop_size
        self.hflip_p = hflip_p
        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)

    def __call__(self, image, mask):
        # image: HxWx3 uint8 RGB ; mask: HxW uint8 (0/1)
        
        # Random Scale
        scale = random.uniform(*self.scale_limit)
        h, w = image.shape[:2]
        new_h, new_w = max(1, int(h * scale)), max(1, int(w * scale))
        image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)

        # Padding (Si queda más chica que el crop)
        pad_h = max(self.crop_h - new_h, 0)
        pad_w = max(self.crop_w - new_w, 0)
        if pad_h > 0 or pad_w > 0:
            image = cv2.copyMakeBorder(image, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=0)
            mask = cv2.copyMakeBorder(mask, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=0)

        # Random crop
        h, w = image.shape[:2]
        top = random.randint(0, h - self.crop_h)
        left = random.randint(0, w - self.crop_w)
        image = image[top:top + self.crop_h, left:left + self.crop_w]
        mask = mask[top:top + self.crop_h, left:left + self.crop_w]

        # Horizontal flip
        if random.random() < self.hflip_p:
            image = cv2.flip(image, 1)
            mask = cv2.flip(mask, 1)

        return image, mask

    def normalize_and_to_tensor(self, image):
        image = image.astype(np.float32) / 255.0
        image = (image - self.mean) / self.std
        return torch.from_numpy(image).permute(2, 0, 1).float()


class DeepCrackDataset(Dataset):
    def __init__(self, root_dir, split='train', transform=None, target_size=(512, 512)):
        """
        split: 'train' o 'test'
        """
        self.root_dir = root_dir
        self.img_dir = os.path.join(self.root_dir, f"{split}_img")
        self.lab_dir = os.path.join(self.root_dir, f"{split}_lab")
        self.images = sorted(os.listdir(self.img_dir))
        self.transform = transform      # Usar JointTransform
        self.target_size = target_size
        self.mean = np.array((0.485, 0.456, 0.406), dtype=np.float32)
        self.std = np.array((0.229, 0.224, 0.225), dtype=np.float32)

    def __len__(self):
        return len(self.images)

    def _generate_boundary_on_the_fly(self, mask):
        # Aplicar Canny para detectar los bordes
        edges = cv2.Canny(mask, 10, 100)

        # Dilatación morfológica para expandir fronteras y conpensar ruido
        kernel = np.ones((3, 3), np.uint8)
        dilated_edges = cv2.dilate(edges, kernel, iterations=1)

        # Normalizar 0-1
        return (dilated_edges > 0).astype(np.float32)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        img_path = os.path.join(self.img_dir, img_name)

        lab_name = os.path.splitext(img_name)[0] + '.png'
        lab_path = os.path.join(self.lab_dir, lab_name)

        # Cargar imágenes y máscara
        image = cv2.imread(img_path, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(lab_path, cv2.IMREAD_GRAYSCALE)

        # Binarizar máscara
        mask = (mask > 127).astype(np.uint8)

        if self.transform:
            image, mask = self.transform(image, mask)
        else:
            # Redimensionar a un tamaño fijo
            image = cv2.resize(image, self.target_size, interpolation=cv2.INTER_LINEAR)
            mask = cv2.resize(mask, self.target_size, interpolation=cv2.INTER_NEAREST)
        
        # Generar ground truth para la rama derivativa (PIDNet specific)
        boundary = self._generate_boundary_on_the_fly((mask * 255).astype(np.uint8))

        if self.transform:
            image = self.transform.normalize_and_to_tensor(image)
        else:
            image = (image.astype(np.float32) / 255.0 - self.mean) / self.std
            image = torch.from_numpy(image).permute(2, 0, 1).float()

        mask = torch.from_numpy(mask.astype(np.int64))
        boundary = torch.from_numpy(boundary)

        return image, mask, boundary
