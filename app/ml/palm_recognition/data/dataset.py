"""
Dataset dan DataLoader untuk PalmNet-Lite training.

Reuses augmentation policy dari existing pipeline, disesuaikan
untuk PalmNet-Lite yang dilatih from scratch.

Format CSV: path, label, palm_id, session
  - train.csv: Palm 1-400, session 1 (label = 0..N-1 integer)
  - val.csv:   Palm 1-400, session 2 (same label map)
  - test.csv:  Palm 401-600, session 1+2 (label = palm_XXXXX string)

Augmentation policy:
  ✅ Rotation ±15°
  ✅ RandomResizedCrop scale 0.9-1.0
  ✅ ColorJitter brightness/contrast/saturation/hue
  ✅ GaussianBlur (low prob)
  ✅ GaussianNoise (low prob)
  ❌ NO horizontal flip (kiri ≠ kanan)
  ❌ NO vertical flip
  ❌ NO aggressive augmentation pada val/test
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

try:
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
    _ALBUMENTATIONS = True
except ImportError:
    import torchvision.transforms as T
    _ALBUMENTATIONS = False

# Normalization constants (sama dengan inference contract)
_NORM_MEAN = [0.5, 0.5, 0.5]
_NORM_STD  = [0.5, 0.5, 0.5]
_ROI_SIZE  = 112


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
    """Training augmentation transform.

    Stochastic augmentation hanya untuk training.
    NO horizontal/vertical flip.
    """
    if _ALBUMENTATIONS:
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
            A.GaussNoise(var_limit=(10.0, 30.0), p=noise_prob),
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
    """Dataset untuk palm ROI yang sudah di-preprocess.

    CSV harus memiliki kolom: path, label
    Kolom tambahan (palm_id, session) dipertahankan untuk evaluation protocol.

    Args:
        csv_path:   path ke CSV split file
        transform:  albumentations atau torchvision transform
        base_path:  jika path di CSV relatif, ini base path-nya
    """

    def __init__(
        self,
        csv_path: str | Path,
        transform: Optional[Callable] = None,
        base_path: Optional[str | Path] = None,
    ):
        self.csv_path = Path(csv_path)
        self.data = pd.read_csv(self.csv_path)

        required_cols = {"path", "label"}
        missing = required_cols - set(self.data.columns)
        if missing:
            raise ValueError(f"CSV {csv_path} missing kolom: {missing}")

        self.transform = transform
        self.base_path = Path(base_path) if base_path else None

        # Encode label ke integer jika string
        if not pd.api.types.is_integer_dtype(self.data["label"]):
            unique_labels = sorted(self.data["label"].unique())
            self.label_map: dict = {lbl: idx for idx, lbl in enumerate(unique_labels)}
            self.data = self.data.copy()
            self.data["label"] = self.data["label"].map(self.label_map)
        else:
            self.label_map = {}

        self.num_classes: int = int(self.data["label"].nunique())
        # Simpan session info jika ada (untuk evaluation protocol)
        self.has_session = "session" in self.data.columns
        self.has_palm_id = "palm_id" in self.data.columns

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int):
        row = self.data.iloc[idx]
        img_path = Path(row["path"])
        if self.base_path is not None and not img_path.is_absolute():
            img_path = self.base_path / img_path

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

    def get_metadata(self) -> pd.DataFrame:
        """Return metadata (path, label, palm_id, session) untuk evaluation."""
        return self.data.copy()


def build_dataloaders(
    train_csv: str | Path,
    val_csv: str | Path,
    batch_size: int = 64,
    num_workers: int = 0,
    pin_memory: bool = False,
    aug_config: Optional[dict] = None,
) -> tuple:
    """Build train + val DataLoader.

    Returns:
        (train_loader, val_loader, num_classes)
    """
    aug = aug_config or {}
    train_ds = PalmDataset(train_csv, transform=get_train_transform(**{
        k: aug[k] for k in [
            "rotate_deg", "crop_scale", "crop_ratio",
            "brightness", "contrast", "saturation", "hue",
            "blur_prob", "noise_prob"
        ] if k in aug
    }))
    val_ds = PalmDataset(val_csv, transform=get_val_transform())

    g = torch.Generator()
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,      # BatchNorm stability
        generator=g,
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
