
"""
Generate train / val / test split CSV dari dataset Tongji (flat structure).
Format: session1/00001.bmp, dst.

Protocol:
- Total 600 telapak (palms).
- Palm 1-400 (400 kelas) -> Training & Validation.
  - Session 1 -> Train (10 gambar per telapak)
  - Session 2 -> Val (10 gambar per telapak)
- Palm 401-600 (200 telapak) -> Test set HOLD-OUT (Open-set evaluation).
"""
import argparse
from pathlib import Path
import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import (
    PROCESSED_DIR,
    SPLITS_DIR,
    RANDOM_SEED,
)

def create_splits():
    processed_dir = Path(PROCESSED_DIR)
    output_dir = Path(SPLITS_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not processed_dir.exists():
        print(f"❌ Folder processed tidak ada: {processed_dir}")
        return

    train_rows = []
    val_rows = []
    test_rows = []

    print(f"Mulai membuat split dari {processed_dir}...")

    # Loop untuk 600 telapak
    for palm_id in range(1, 601):
        # 1-400 untuk Train/Val
        # 401-600 untuk Test
        is_train_palm = (palm_id <= 400)
        
        # Range file untuk telapak ini (misal palm 1: 1-10, palm 2: 11-20)
        start_idx = (palm_id - 1) * 10 + 1
        end_idx = palm_id * 10
        
        for img_idx in range(start_idx, end_idx + 1):
            filename = f"{img_idx:05d}.bmp" # Sesuaikan extension jika perlu (.bmp atau .jpg)
            
            # Cek di Sesi 1
            s1_path = processed_dir / "session1" / filename
            if s1_path.exists():
                if is_train_palm:
                    train_rows.append({
                        "path": str(s1_path),
                        "label": palm_id - 1, # Label 0-399
                        "palm_id": palm_id,
                        "session": 1
                    })
                else:
                    test_rows.append({
                        "path": str(s1_path),
                        "label": f"palm_{palm_id:05d}",
                        "palm_id": palm_id,
                        "session": 1
                    })

            # Cek di Sesi 2
            s2_path = processed_dir / "session2" / filename
            if s2_path.exists():
                if is_train_palm:
                    # Sesi 2 dari telapak training dijadikan data Validasi (Cross-Session)
                    val_rows.append({
                        "path": str(s2_path),
                        "label": palm_id - 1,
                        "palm_id": palm_id,
                        "session": 2
                    })
                else:
                    test_rows.append({
                        "path": str(s2_path),
                        "label": f"palm_{palm_id:05d}",
                        "palm_id": palm_id,
                        "session": 2
                    })

    # Simpan ke CSV
    train_df = pd.DataFrame(train_rows)
    val_df = pd.DataFrame(val_rows)
    test_df = pd.DataFrame(test_rows)

    train_df.to_csv(output_dir / "train.csv", index=False)
    val_df.to_csv(output_dir / "val.csv", index=False)
    test_df.to_csv(output_dir / "test.csv", index=False)

    print("\n=== Split Summary ===")
    print(f"  Train (S1, Palms 1-400): {len(train_df)} gambar")
    print(f"  Val   (S2, Palms 1-400): {len(val_df)} gambar")
    print(f"  Test  (S1&S2, Palms 401-600): {len(test_df)} gambar")
    print(f"\nCSV disimpan di: {output_dir}")

if __name__ == "__main__":
    create_splits()
