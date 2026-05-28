"""
Generate train / val / test split CSV dari dataset Tongji yang sudah
di-preprocess (ROI 112×112).

Struktur input yang diharapkan:
    ml/data/tongji/processed/
        individual_001/
            left/
                img001.jpg, img002.jpg, ...
            right/
                img001.jpg, img002.jpg, ...
        individual_002/
            ...

Tongji punya 300 individu × 2 tangan × 10 gambar per sesi × 2 sesi.

Protocol:
- 200 individu pertama → training (Phase 1 + Phase 2)
  - Setiap individu × 2 tangan = 400 kelas
  - Sesi 1 → train (8 gambar per tangan)
  - Sesi 2 → val (2 gambar per tangan)
- 100 individu sisanya → test set HOLD-OUT (cross-session evaluation, open-set)

Output CSV:
    ml/data/splits/train.csv     ← 200 individu × 2 tangan × 8 = 3200 rows
    ml/data/splits/val.csv       ← 200 individu × 2 tangan × 2 = 800 rows
    ml/data/splits/test.csv      ← 100 individu × semua gambar (untuk eval)

NOTE: Skema "sesi 1 / sesi 2" yang asli di Tongji bisa berupa naming
convention atau folder terpisah. Script ini default ke split berdasarkan
URUTAN FILE (alfabet) — kalau dataset Anda punya naming khusus, sesuaikan
fungsi `_split_session()`.
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path

import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import (
    PROCESSED_DIR,
    SPLITS_DIR,
    TRAIN_INDIVIDUALS,
    TEST_INDIVIDUALS,
    TRAIN_IMGS_PER_HAND,
    RANDOM_SEED,
)


def _list_images(folder: Path) -> list[Path]:
    """List gambar valid di folder, sorted alfabetis."""
    if not folder.exists():
        return []
    images = []
    for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp"):
        images.extend(folder.glob(ext))
    return sorted(images)


def _split_session(images: list[Path], train_count: int) -> tuple[list[Path], list[Path]]:
    """
    Default split: N gambar pertama → train (sesi 1), sisanya → val (sesi 2).
    Override fungsi ini kalau dataset Anda punya naming convention untuk sesi.
    """
    return images[:train_count], images[train_count:]


def create_splits(
    processed_dir: Path = PROCESSED_DIR,
    output_dir: Path = SPLITS_DIR,
    train_individuals: int = TRAIN_INDIVIDUALS,
    test_individuals: int = TEST_INDIVIDUALS,
    train_imgs_per_hand: int = TRAIN_IMGS_PER_HAND,
    random_seed: int = RANDOM_SEED,
):
    """
    Generate train.csv, val.csv, test.csv.

    Returns:
        dict dengan statistik split.
    """
    random.seed(random_seed)

    processed_dir = Path(processed_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not processed_dir.exists():
        raise FileNotFoundError(f"Processed dir tidak ada: {processed_dir}")

    # List individu, deterministic order
    individuals = sorted([d for d in processed_dir.iterdir() if d.is_dir()])
    if len(individuals) == 0:
        raise RuntimeError(
            f"Tidak ada folder individu di {processed_dir}. "
            f"Pastikan dataset sudah di-preprocess dengan extract_roi.py."
        )

    print(f"Total individu ditemukan: {len(individuals)}")

    # Validasi jumlah individu cukup
    needed = train_individuals + test_individuals
    if len(individuals) < needed:
        print(
            f"⚠️  Hanya {len(individuals)} individu tersedia, butuh minimal {needed}. "
            f"Adjust auto: train={int(len(individuals)*2/3)}, test={len(individuals) - int(len(individuals)*2/3)}"
        )
        train_individuals = int(len(individuals) * 2 / 3)
        test_individuals = len(individuals) - train_individuals

    train_inds = individuals[:train_individuals]
    test_inds = individuals[train_individuals : train_individuals + test_individuals]

    print(f"Training: {len(train_inds)} individu")
    print(f"Test: {len(test_inds)} individu")

    train_rows = []
    val_rows = []
    test_rows = []
    label_map = {}  # mapping: (ind_name, hand) → class_idx untuk training set

    # ====== TRAIN + VAL ======
    label_idx = 0
    for ind_dir in train_inds:
        ind_name = ind_dir.name

        # Tangan kiri dan kanan = 2 kelas berbeda
        for hand in ("left", "right"):
            hand_dir = ind_dir / hand
            if not hand_dir.exists():
                # Fallback: kalau dataset tidak punya subfolder left/right,
                # treat semua gambar sebagai 1 kelas
                if hand == "left":
                    hand_imgs = _list_images(ind_dir)
                    if not hand_imgs:
                        continue
                else:
                    continue  # right tidak ada, skip
            else:
                hand_imgs = _list_images(hand_dir)
                if not hand_imgs:
                    continue

            session1, session2 = _split_session(hand_imgs, train_imgs_per_hand)
            class_key = f"{ind_name}__{hand}"
            label_map[class_key] = label_idx

            for img_path in session1:
                train_rows.append({
                    "path": str(img_path),
                    "label": label_idx,
                    "individual": ind_name,
                    "hand": hand,
                })
            for img_path in session2:
                val_rows.append({
                    "path": str(img_path),
                    "label": label_idx,
                    "individual": ind_name,
                    "hand": hand,
                })

            label_idx += 1

    # ====== TEST (open-set, label = individual name) ======
    for ind_dir in test_inds:
        ind_name = ind_dir.name

        for hand in ("left", "right"):
            hand_dir = ind_dir / hand
            if hand_dir.exists():
                hand_imgs = _list_images(hand_dir)
            elif hand == "left":
                hand_imgs = _list_images(ind_dir)
            else:
                continue

            for img_path in hand_imgs:
                test_rows.append({
                    "path": str(img_path),
                    "label": f"{ind_name}__{hand}",
                    "individual": ind_name,
                    "hand": hand,
                })

    # Save
    train_df = pd.DataFrame(train_rows)
    val_df = pd.DataFrame(val_rows)
    test_df = pd.DataFrame(test_rows)

    train_df.to_csv(output_dir / "train.csv", index=False)
    val_df.to_csv(output_dir / "val.csv", index=False)
    test_df.to_csv(output_dir / "test.csv", index=False)

    # Save label map juga untuk reproducibility
    label_map_df = pd.DataFrame([
        {"class_key": k, "label": v} for k, v in label_map.items()
    ])
    label_map_df.to_csv(output_dir / "label_map.csv", index=False)

    # Summary
    summary = {
        "train_rows": len(train_df),
        "val_rows": len(val_df),
        "test_rows": len(test_df),
        "num_classes_train": label_idx,
        "test_unique_individuals": len(test_inds),
    }

    print("\n=== Split Summary ===")
    for k, v in summary.items():
        print(f"  {k}: {v}")
    print(f"\nFiles ditulis ke: {output_dir}")

    return summary


def main():
    parser = argparse.ArgumentParser(description="Generate train/val/test splits")
    parser.add_argument("--processed-dir", type=str, default=str(PROCESSED_DIR))
    parser.add_argument("--output-dir", type=str, default=str(SPLITS_DIR))
    parser.add_argument("--train-individuals", type=int, default=TRAIN_INDIVIDUALS)
    parser.add_argument("--test-individuals", type=int, default=TEST_INDIVIDUALS)
    parser.add_argument("--train-imgs-per-hand", type=int, default=TRAIN_IMGS_PER_HAND)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    args = parser.parse_args()

    create_splits(
        processed_dir=Path(args.processed_dir),
        output_dir=Path(args.output_dir),
        train_individuals=args.train_individuals,
        test_individuals=args.test_individuals,
        train_imgs_per_hand=args.train_imgs_per_hand,
        random_seed=args.seed,
    )


if __name__ == "__main__":
    main()
