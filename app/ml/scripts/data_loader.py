"""
Data loader untuk training MobileFaceNet di Tongji palm ROI.

Format input (CSV):
    path,label
    ml/data/tongji/processed/ind_001/left/img001.jpg,0
    ml/data/tongji/processed/ind_001/left/img002.jpg,0
    ml/data/tongji/processed/ind_001/right/img001.jpg,1
    ...

Augmentation policy:
- ✅ Rotation ±10° (palm bisa miring saat capture)
- ✅ Random crop 0.9-1.0 scale (mengatasi sedikit translasi ROI)
- ✅ Color jitter (brightness, contrast, saturation — mengatasi pencahayaan)
- ❌ NO horizontal flip — telapak kiri ≠ telapak kanan, dan keduanya
   adalah kelas berbeda di sistem kita
- ❌ NO vertical flip — telapak terbalik tidak realistis untuk use case
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Callable

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader

try:
    import albumentations as A
    from albumentations.pytorch import ToTensorV2

    _ALBUMENTATIONS_AVAILABLE = True
except ImportError:
    _ALBUMENTATIONS_AVAILABLE = False
    # Fallback: pakai torchvision transforms
    import torchvision.transforms as T

from config import (
    ROI_SIZE,
    NORM_MEAN,
    NORM_STD,
    AUG_ROTATE_DEG,
    AUG_CROP_SCALE,
    AUG_CROP_RATIO,
    AUG_BRIGHTNESS,
    AUG_CONTRAST,
    AUG_SATURATION,
    AUG_HUE,
)


def get_train_transform():
    """
    Augmentation untuk training. Pakai albumentations kalau tersedia
    (lebih cepat), fallback ke torchvision.
    """
    if _ALBUMENTATIONS_AVAILABLE:
        return A.Compose([
            A.Rotate(limit=AUG_ROTATE_DEG, p=0.8, border_mode=0),
            A.RandomResizedCrop(
                size=(ROI_SIZE, ROI_SIZE),
                scale=AUG_CROP_SCALE,
                ratio=AUG_CROP_RATIO,
                p=1.0,
            ),
            A.ColorJitter(
                brightness=AUG_BRIGHTNESS,
                contrast=AUG_CONTRAST,
                saturation=AUG_SATURATION,
                hue=AUG_HUE,
                p=0.5,
            ),
            A.GaussianBlur(blur_limit=(3, 5), p=0.2),
            A.GaussNoise(var_limit=(10.0, 30.0), p=0.1),
            A.Normalize(mean=NORM_MEAN, std=NORM_STD),
            ToTensorV2(),
        ])
    else:
        return T.Compose([
            T.RandomRotation(degrees=AUG_ROTATE_DEG),
            T.RandomResizedCrop(
                size=ROI_SIZE,
                scale=AUG_CROP_SCALE,
                ratio=AUG_CROP_RATIO,
            ),
            T.ColorJitter(
                brightness=AUG_BRIGHTNESS,
                contrast=AUG_CONTRAST,
                saturation=AUG_SATURATION,
                hue=AUG_HUE,
            ),
            T.GaussianBlur(kernel_size=(3, 5), sigma=(0.1, 2.0)),
            T.ToTensor(),
            T.Normalize(mean=NORM_MEAN, std=NORM_STD),
        ])


def get_val_transform():
    """Transform untuk evaluation — deterministic, tanpa augmentasi."""
    if _ALBUMENTATIONS_AVAILABLE:
        return A.Compose([
            A.Resize(ROI_SIZE, ROI_SIZE),
            A.Normalize(mean=NORM_MEAN, std=NORM_STD),
            ToTensorV2(),
        ])
    else:
        return T.Compose([
            T.Resize((ROI_SIZE, ROI_SIZE)),
            T.ToTensor(),
            T.Normalize(mean=NORM_MEAN, std=NORM_STD),
        ])


class PalmDataset(Dataset):
    """
    PyTorch Dataset untuk palm ROI yang sudah di-preprocess.

    CSV harus punya kolom 'path' dan 'label'. Path bisa absolute atau relatif
    terhadap CSV file location.

    Args:
        csv_path: path ke CSV split file (train.csv / val.csv)
        transform: albumentations atau torchvision transform
        base_path: kalau path di CSV relatif, ini base path-nya
    """

    def __init__(
        self,
        csv_path: str | Path,
        transform: Optional[Callable] = None,
        base_path: Optional[str | Path] = None,
    ):
        self.csv_path = Path(csv_path)
        self.data = pd.read_csv(self.csv_path)

        # Validasi kolom
        required_cols = {"path", "label"}
        if not required_cols.issubset(self.data.columns):
            missing = required_cols - set(self.data.columns)
            raise ValueError(f"CSV {csv_path} missing kolom: {missing}")

        self.transform = transform
        self.base_path = Path(base_path) if base_path else None

        # Cek apakah label int — kalau string, encode jadi int
        if not pd.api.types.is_integer_dtype(self.data["label"]):
            unique_labels = sorted(self.data["label"].unique())
            self.label_map = {lbl: idx for idx, lbl in enumerate(unique_labels)}
            self.data["label"] = self.data["label"].map(self.label_map)
            print(f"[PalmDataset] Encoded {len(unique_labels)} string labels to int")

        self.num_classes = self.data["label"].nunique()

    def __len__(self):
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
            if _ALBUMENTATIONS_AVAILABLE and isinstance(self.transform, A.Compose):
                transformed = self.transform(image=img_np)
                tensor = transformed["image"]
            else:
                # torchvision expects PIL Image
                tensor = self.transform(img)
        else:
            tensor = torch.from_numpy(img_np).permute(2, 0, 1).float() / 255.0

        label = int(row["label"])
        return tensor, label


def build_dataloaders(
    train_csv: str | Path,
    val_csv: str | Path,
    batch_size: int = 64,
    num_workers: int = 4,
    pin_memory: bool = True,
):
    """
    Convenience builder untuk train + val DataLoader.

    Returns:
        (train_loader, val_loader, num_classes)
    """
    train_ds = PalmDataset(train_csv, transform=get_train_transform())
    val_ds = PalmDataset(val_csv, transform=get_val_transform())

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,  # untuk batch norm stability
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


if __name__ == "__main__":
    # Smoke test
    import sys
    if len(sys.argv) > 1:
        csv_path = sys.argv[1]
        ds = PalmDataset(csv_path, transform=get_train_transform())
        print(f"Dataset size: {len(ds)}")
        print(f"Num classes: {ds.num_classes}")
        if len(ds) > 0:
            img, lbl = ds[0]
            print(f"Sample 0 tensor shape: {tuple(img.shape)}, label: {lbl}")
    else:
        print("Usage: python data_loader.py <train_csv>")
