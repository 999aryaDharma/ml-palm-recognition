"""
Palm ROI Extraction Module
Extracts Region of Interest (palm area) from detected hand
"""

from PIL import Image
from typing import Optional
import numpy as np
import cv2
from math import atan2, degrees, sqrt


def _lm_xy(lm, width: int, height: int) -> tuple[float, float]:
    """Convert landmark dict to pixel coords."""
    return lm["x"] * width, lm["y"] * height


def extract_palm_roi(image: Image.Image, landmarks: list[dict], output_size: int = 112) -> Optional[Image.Image]:
    """
    Extract palm region of interest from detected hand.
    
    Args:
        image: Original PIL Image
        landmarks: List of 21 landmark dictionaries
        output_size: Target size (square)
        
    Returns:
        Cropped PIL Image of palm region, or None if extraction fails
    """
    if not landmarks or len(landmarks) < 21:
        return None

    w, h = image.size
    img_rgb = np.array(image.convert("RGB"))
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

    p5 = np.array(_lm_xy(landmarks[5], w, h))
    p17 = np.array(_lm_xy(landmarks[17], w, h))
    p0 = np.array(_lm_xy(landmarks[0], w, h))

    dx, dy = p17[0] - p5[0], p17[1] - p5[1]
    palm_width = sqrt(dx**2 + dy**2)
    if palm_width < 50:
        return None

    angle_deg = degrees(atan2(dy, dx))
    center = (p5 + p17) / 2.0
    to_wrist = p0 - center
    wrist_dist = float(np.linalg.norm(to_wrist))

    if wrist_dist < 1.0:
        return None

    offset_vec = to_wrist / wrist_dist * (0.30 * palm_width)
    roi_center = center + offset_vec
    side = palm_width * 1.2

    rot_matrix = cv2.getRotationMatrix2D(tuple(roi_center.tolist()), angle_deg, 1.0)
    rotated = cv2.warpAffine(
        img_bgr,
        rot_matrix,
        (img_bgr.shape[1], img_bgr.shape[0]),
        borderMode=cv2.BORDER_REPLICATE,
    )

    half = side / 2.0
    x1 = int(max(0, roi_center[0] - half))
    y1 = int(max(0, roi_center[1] - half))
    x2 = int(min(rotated.shape[1], roi_center[0] + half))
    y2 = int(min(rotated.shape[0], roi_center[1] + half))

    crop_w = x2 - x1
    crop_h = y2 - y1
    if crop_w < 50 or crop_h < 50:
        return None

    roi = rotated[y1:y2, x1:x2]
    roi_resized = cv2.resize(roi, (output_size, output_size), interpolation=cv2.INTER_AREA)

    roi_rgb = cv2.cvtColor(roi_resized, cv2.COLOR_BGR2RGB)
    return Image.fromarray(roi_rgb)
