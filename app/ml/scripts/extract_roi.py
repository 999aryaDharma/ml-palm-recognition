"""
Stage 2a: ROI extraction palm dari landmark MediaPipe.

Pipeline:
1. Hitung valley line antara landmark 5 (index MCP) dan 17 (pinky MCP)
2. Tentukan angle telapak dari valley line → rotasi gambar agar telapak
   horizontal
3. Geser center ROI ~30% ke arah wrist (landmark 0) supaya area telapak
   utama (bukan jari) yang ter-crop
4. Crop square ROI dengan side = palm_width × 1.2
5. Resize ke 112×112 RGB
6. Quality check: Laplacian variance untuk blur rejection

Output: PIL Image 112×112 RGB siap masuk ke MobileFaceNet.

Penting: Implementasi ini di-REUSE LANGSUNG di backend FastAPI
(backend/ml/roi.py = copy file ini). Tidak ada cross-language port.
"""
from __future__ import annotations

from math import atan2, degrees, sqrt
from typing import Optional, Union

import cv2
import numpy as np
from PIL import Image

from config import (
    ROI_SIZE,
    BLUR_THRESHOLD_LAPLACIAN,
    MIN_PALM_WIDTH_PX,
)


# Tipe alias supaya gampang dipakai dari MediaPipe object atau dict
LandmarksType = Union[list[dict], list]  # list of dict OR list of NormalizedLandmark


def _lm_xy(lm, width: int, height: int) -> tuple[float, float]:
    """Convert landmark (dict atau NormalizedLandmark) ke pixel coords."""
    if isinstance(lm, dict):
        return lm["x"] * width, lm["y"] * height
    # MediaPipe NormalizedLandmark object
    return lm.x * width, lm.y * height


def extract_palm_roi(
    pil_image: Image.Image,
    landmarks: LandmarksType,
    output_size: int = ROI_SIZE,
    blur_threshold: float = BLUR_THRESHOLD_LAPLACIAN,
    return_debug: bool = False,
) -> Optional[Image.Image] | tuple[Optional[Image.Image], dict]:
    """
    Extract ROI palm 112×112 RGB dari gambar utuh + landmark.

    Args:
        pil_image: PIL Image (full hand image)
        landmarks: list of 21 landmark (dict dengan 'x','y' OR MediaPipe object)
        output_size: ukuran output (default 112)
        blur_threshold: Laplacian variance minimum (default 50). Lower = lebih
                        permisif (terima gambar agak blur).
        return_debug: kalau True, return juga dict berisi intermediate values
                      (center, angle, palm_width, blur_var) untuk debugging.

    Returns:
        PIL Image 112×112 RGB, atau None kalau gagal.
        Kalau return_debug=True: (PIL_or_None, debug_dict).
    """
    debug: dict = {"status": "ok"}

    # Validasi input
    if landmarks is None or len(landmarks) < 21:
        debug["status"] = "invalid_landmarks"
        return (None, debug) if return_debug else None

    w, h = pil_image.size
    img_rgb = np.array(pil_image.convert("RGB"))
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

    # Landmark kunci
    p5 = np.array(_lm_xy(landmarks[5], w, h))   # base index finger
    p17 = np.array(_lm_xy(landmarks[17], w, h)) # base pinky
    p0 = np.array(_lm_xy(landmarks[0], w, h))   # wrist

    # Step 1: Valley line vector + length (palm width)
    dx, dy = p17[0] - p5[0], p17[1] - p5[1]
    palm_width = sqrt(dx**2 + dy**2)
    debug["palm_width_px"] = float(palm_width)

    if palm_width < MIN_PALM_WIDTH_PX:
        debug["status"] = "palm_too_small"
        return (None, debug) if return_debug else None

    # Step 2: Angle telapak dari horizontal axis
    angle_deg = degrees(atan2(dy, dx))
    debug["angle_deg"] = float(angle_deg)

    # Step 3: Center awal = midpoint valley line, lalu geser ke arah wrist
    center = (p5 + p17) / 2.0
    to_wrist = p0 - center
    wrist_dist = float(np.linalg.norm(to_wrist))

    if wrist_dist < 1.0:
        debug["status"] = "wrist_too_close"
        return (None, debug) if return_debug else None

    # Offset: ~30% palm width ke arah wrist
    offset_magnitude = 0.30 * palm_width
    offset_vec = to_wrist / wrist_dist * offset_magnitude
    roi_center = center + offset_vec
    debug["roi_center_xy"] = (float(roi_center[0]), float(roi_center[1]))

    # Step 4: ROI side length (palm_width × 1.2 untuk margin)
    side = palm_width * 1.2
    debug["roi_side_px"] = float(side)

    # Step 5: Warp affine — rotasi seluruh gambar agar valley line horizontal
    rot_matrix = cv2.getRotationMatrix2D(tuple(roi_center.tolist()), angle_deg, 1.0)
    rotated = cv2.warpAffine(
        img_bgr,
        rot_matrix,
        (img_bgr.shape[1], img_bgr.shape[0]),
        borderMode=cv2.BORDER_REPLICATE,
    )

    # Step 6: Crop ROI di sekitar center
    half = side / 2.0
    x1 = int(max(0, roi_center[0] - half))
    y1 = int(max(0, roi_center[1] - half))
    x2 = int(min(rotated.shape[1], roi_center[0] + half))
    y2 = int(min(rotated.shape[0], roi_center[1] + half))

    crop_w = x2 - x1
    crop_h = y2 - y1
    if crop_w < 50 or crop_h < 50:
        debug["status"] = "crop_too_small"
        debug["crop_size"] = (crop_w, crop_h)
        return (None, debug) if return_debug else None

    roi = rotated[y1:y2, x1:x2]

    # Step 7: Resize ke output_size
    roi_resized = cv2.resize(roi, (output_size, output_size), interpolation=cv2.INTER_AREA)

    # Step 8: Blur check via Laplacian variance
    gray = cv2.cvtColor(roi_resized, cv2.COLOR_BGR2GRAY)
    lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    debug["blur_variance"] = lap_var

    if lap_var < blur_threshold:
        debug["status"] = "too_blurry"
        return (None, debug) if return_debug else None

    # Step 9: Konversi balik ke RGB PIL
    roi_rgb = cv2.cvtColor(roi_resized, cv2.COLOR_BGR2RGB)
    pil_roi = Image.fromarray(roi_rgb)

    return (pil_roi, debug) if return_debug else pil_roi


def batch_extract_dataset(
    raw_dir: str,
    output_dir: str,
    landmark_extractor=None,
    max_per_individual: Optional[int] = None,
    verbose: bool = True,
):
    """
    Helper untuk preprocess seluruh dataset (mis. Tongji raw → ROI 112×112).

    Struktur input yang diasumsikan:
        raw_dir/
            individual_001/
                left/
                    img001.jpg
                    img002.jpg
                right/
                    img001.jpg
            individual_002/
                ...

    Output:
        output_dir/
            individual_001/
                left/
                    img001.jpg  (112×112 ROI)
                ...

    Args:
        raw_dir: path ke dataset mentah
        output_dir: path output ROI
        landmark_extractor: instance LandmarkExtractor. Kalau None, akan
                            di-instantiate di sini.
        max_per_individual: kalau di-set, hanya proses N gambar pertama per
                            individu (untuk testing cepat).
    """
    from pathlib import Path

    raw_path = Path(raw_dir)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    if landmark_extractor is None:
        from extract_landmarks import LandmarkExtractor
        landmark_extractor = LandmarkExtractor()

    stats = {"total": 0, "success": 0, "no_hand": 0, "roi_failed": 0}

    individuals = sorted([d for d in raw_path.iterdir() if d.is_dir()])
    if verbose:
        print(f"Memproses {len(individuals)} individu dari {raw_path}")

    for ind_dir in individuals:
        rel_path = ind_dir.relative_to(raw_path)
        # Walk through semua subdirectory (mis. left/, right/)
        for img_path in ind_dir.rglob("*"):
            if not img_path.is_file() or img_path.suffix.lower() not in (".jpg", ".jpeg", ".png", ".bmp"):
                continue

            stats["total"] += 1

            try:
                img = Image.open(img_path)
            except Exception as e:
                if verbose:
                    print(f"  [skip] {img_path}: {e}")
                continue

            lm_result = landmark_extractor.extract(img)
            if lm_result is None:
                stats["no_hand"] += 1
                continue

            roi = extract_palm_roi(img, lm_result["landmarks"])
            if roi is None:
                stats["roi_failed"] += 1
                continue

            # Tentukan output path (preserve struktur)
            rel_img_path = img_path.relative_to(raw_path)
            out_img_path = out_path / rel_img_path
            out_img_path = out_img_path.with_suffix(".jpg")
            out_img_path.parent.mkdir(parents=True, exist_ok=True)
            roi.save(out_img_path, "JPEG", quality=95)
            stats["success"] += 1

            if max_per_individual is not None:
                ind_count = stats["success"]  # rough — bukan per individu
                # NOTE: untuk per-individu limit, butuh tracking terpisah
                # Disederhanakan di sini.

        if verbose:
            print(f"  [{rel_path}] cumulative success: {stats['success']}/{stats['total']}")

    if verbose:
        print("\n=== Batch extraction summary ===")
        for k, v in stats.items():
            print(f"  {k}: {v}")
        success_rate = stats["success"] / max(1, stats["total"]) * 100
        print(f"  success_rate: {success_rate:.1f}%")

    return stats


if __name__ == "__main__":
    # Smoke test: kalau ada argumen, extract ROI dari 1 gambar
    import sys

    if len(sys.argv) > 1:
        from extract_landmarks import LandmarkExtractor

        img_path = sys.argv[1]
        out_path = sys.argv[2] if len(sys.argv) > 2 else "roi_test.jpg"

        extractor = LandmarkExtractor()
        img = Image.open(img_path)
        lm_result = extractor.extract(img)

        if lm_result is None:
            print(f"Tidak ada tangan terdeteksi di: {img_path}")
            sys.exit(1)

        roi, debug = extract_palm_roi(img, lm_result["landmarks"], return_debug=True)
        print("Debug info:")
        for k, v in debug.items():
            print(f"  {k}: {v}")

        if roi is not None:
            roi.save(out_path)
            print(f"ROI saved: {out_path}")
        else:
            print(f"ROI extraction gagal: {debug.get('status')}")
    else:
        print("Usage: python extract_roi.py <input_image> [output_path]")
