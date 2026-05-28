"""
Preprocess seluruh dataset: raw → ROI 112×112.

Pipeline:
  raw image → MediaPipe landmark → ROI extraction → save 112×112 JPEG

Run sekali di awal sebelum training (estimasi 5-15 menit untuk 12k gambar
Tongji, tergantung CPU).

Output:
  ml/data/tongji/processed/individual_XXX/{left,right}/img_001.jpg
"""
from __future__ import annotations

import argparse
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import RAW_DIR, PROCESSED_DIR, HAND_LANDMARKER_PATH
from extract_roi import batch_extract_dataset
from extract_landmarks import LandmarkExtractor


def main():
    parser = argparse.ArgumentParser(description="Preprocess raw dataset to ROI 112×112")
    parser.add_argument("--raw-dir", type=str, default=str(RAW_DIR))
    parser.add_argument("--output-dir", type=str, default=str(PROCESSED_DIR))
    parser.add_argument("--landmark-model", type=str, default=str(HAND_LANDMARKER_PATH))
    args = parser.parse_args()

    raw_dir = Path(args.raw_dir)
    out_dir = Path(args.output_dir)

    if not raw_dir.exists():
        print(f"❌ Raw dir tidak ada: {raw_dir}")
        print(f"   Download dataset Tongji dulu dan letakkan di {raw_dir}")
        return 1

    print(f"Raw dir:        {raw_dir}")
    print(f"Output dir:     {out_dir}")
    print(f"Landmark model: {args.landmark_model}")
    print()

    extractor = LandmarkExtractor(model_path=args.landmark_model)
    stats = batch_extract_dataset(
        raw_dir=str(raw_dir),
        output_dir=str(out_dir),
        landmark_extractor=extractor,
        verbose=True,
    )
    extractor.close()

    print(f"\n✅ Selesai. ROI tersimpan di: {out_dir}")
    print(f"   Next: python scripts/create_splits.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
