"""
Dataset dan DataLoader untuk PalmNet-Lite training.

Augmentation policy:
  ✅ Rotation ±15°
  ✅ RandomResizedCrop scale 0.9-1.0
  ✅ ColorJitter brightness/contrast/saturation/hue
  ✅ GaussianBlur (low prob)
  ✅ GaussianNoise (low prob, Albumentations 2.x compatible)
  ❌ NO horizontal flip (kiri ≠ kanan)
  ❌ NO vertical flip
"""
from __future__ import annotations

import random
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from palm_recognition.paths import resolve_ml_path

try:
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
    _ALBUMENTATIONS = True
except ImportError:
    import torchvision.transforms as T
    _ALBUMENTATIONS = False

_NORM_MEAN = [0.5, 0.5, 0.5]
_NORM_STD  = [0.5, 0.5, 0.5]
_ROI_SIZE  = 112


def seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def get_train_transform(
    rotate_deg: float = 15,
    crop_scale: tuple = (0.90, 1.00),
    crop_ratio: tuple = (0.85, 1.15),
    brightness: float = 0.25,
    contrast: float = 0.25,
    saturation: float = 0.15,
    hue: float = 0.05,
    blur_prob: float = 0.2,
    noise_prob: float = 0.1,
):
    """Training augmentation transform (Albumentations 2.x compatible)."""
    if _ALBUMENTATIONS:
        # Albumentations 2.x compatible parameters
        return A.Compose([
            A.Rotate(limit=rotate_deg, p=0.8, border_mode=0),
            A.RandomResizedCrop(
                size=(_ROI_SIZE, _ROI_SIZE),
                scale=crop_scale,
                ratio=crop_ratio,
                p=1.0,
            ),
            A.ColorJitter(
                brightness=brightness,
                contrast=contrast,
                saturation=saturation,
                hue=hue,
                p=0.5,
            ),
            A.GaussianBlur(blur_limit=(3, 5), p=blur_prob),
            A.GaussNoise(p=noise_prob),
            A.Normalize(mean=_NORM_MEAN, std=_NORM_STD),
            ToTensorV2(),
        ])
    else:
        return T.Compose([
            T.RandomRotation(degrees=rotate_deg),
            T.RandomResizedCrop(size=_ROI_SIZE, scale=crop_scale, ratio=crop_ratio),
            T.ColorJitter(brightness=brightness, contrast=contrast,
                          saturation=saturation, hue=hue),
            T.GaussianBlur(kernel_size=(3, 5), sigma=(0.1, 2.0)),
            T.ToTensor(),
            T.Normalize(mean=_NORM_MEAN, std=_NORM_STD),
        ])


def get_val_transform():
    """Validation/test transform — deterministic, no augmentation."""
    if _ALBUMENTATIONS:
        return A.Compose([
            A.Resize(_ROI_SIZE, _ROI_SIZE),
            A.Normalize(mean=_NORM_MEAN, std=_NORM_STD),
            ToTensorV2(),
        ])
    else:
        return T.Compose([
            T.Resize((_ROI_SIZE, _ROI_SIZE)),
            T.ToTensor(),
            T.Normalize(mean=_NORM_MEAN, std=_NORM_STD),
        ])


class PalmDataset(Dataset):
    """Dataset untuk palm ROI yang sudah di-preprocess."""

    def __init__(
        self,
        csv_path: str | Path,
        transform: Optional[Callable] = None,
    ):
        self.csv_path = resolve_ml_path(csv_path)
        if not self.csv_path.exists():
            raise FileNotFoundError(f"CSV file not found: {self.csv_path}")

        self.data = pd.read_csv(self.csv_path)

        required_cols = {"path", "label"}
        missing = required_cols - set(self.data.columns)
        if missing:
            raise ValueError(f"CSV {self.csv_path} missing kolom: {missing}")

        self.transform = transform

        if not pd.api.types.is_integer_dtype(self.data["label"]):
            unique_labels = sorted(self.data["label"].unique())
            self.label_map: dict = {lbl: idx for idx, lbl in enumerate(unique_labels)}
            self.data = self.data.copy()
            self.data["label"] = self.data["label"].map(self.label_map)
        else:
            self.label_map = {}

        self.num_classes: int = int(self.data["label"].nunique())

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int):
        row = self.data.iloc[idx]
        img_path = resolve_ml_path(row["path"])

        try:
            img = Image.open(img_path).convert("RGB")
        except Exception as e:
            raise RuntimeError(f"Gagal load gambar {img_path}: {e}")

        img_np = np.array(img)

        if self.transform is not None:
            if _ALBUMENTATIONS and hasattr(self.transform, "__call__"):
                try:
                    transformed = self.transform(image=img_np)
                    tensor = transformed["image"]
                except (TypeError, KeyError):
                    tensor = self.transform(img)
            else:
                tensor = self.transform(img)
        else:
            tensor = torch.from_numpy(img_np).permute(2, 0, 1).float() / 255.0

        label = int(row["label"])
        return tensor, label


def build_dataloaders(
    train_csv: str | Path,
    val_csv: str | Path,
    batch_size: int = 64,
    num_workers: int = 0,
    pin_memory: bool = False,
    aug_config: Optional[dict] = None,
    seed: int = 42,
) -> tuple:
    """Build reproducible train + val DataLoader."""
    aug = aug_config or {}
    valid_aug_keys = {
        "rotate_deg", "crop_scale", "crop_ratio",
        "brightness", "contrast", "saturation", "hue",
        "blur_prob", "noise_prob"
    }

    # Map legacy keys if present
    mapped_aug = {}
    for k, v in aug.items():
        if k == "gaussian_blur_prob":
            mapped_aug["blur_prob"] = v
        elif k == "gaussian_noise_prob":
            mapped_aug["noise_prob"] = v
        elif k in valid_aug_keys:
            mapped_aug[k] = v

    train_ds = PalmDataset(train_csv, transform=get_train_transform(**mapped_aug))
    val_ds   = PalmDataset(val_csv,   transform=get_val_transform())

    g = torch.Generator()
    g.manual_seed(seed)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
        generator=g,
        worker_init_fn=seed_worker if num_workers > 0 else None,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )

    return train_loader, val_loader, train_ds.num_classes
