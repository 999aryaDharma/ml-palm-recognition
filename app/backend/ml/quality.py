"""
Image Quality Assessment Module
Validates palm image quality for biometric processing
"""

from PIL import Image
import cv2
import numpy as np
from typing import Tuple


def assess_image_quality(image: Image.Image, detection_result: dict = None) -> Tuple[str, float]:
    """
    Assess quality of palm image for biometric processing using Laplacian variance.
    
    Args:
        image: PIL Image object
        detection_result: Hand detection result (optional)
        
    Returns:
        Tuple of (quality_code, quality_score)
        quality_code: 'good', 'acceptable', 'poor'
        quality_score: Normalized score 0-1
    """
    img_rgb = np.array(image.convert("RGB"))
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    
    # Normalize laplacian variance to a 0-1 score (rough heuristic)
    # 50 is our minimum threshold
    score = min(1.0, lap_var / 200.0)
    
    if lap_var < 50.0:
        return "poor", score
    elif lap_var < 100.0:
        return "acceptable", score
    else:
        return "good", score


QUALITY_ERRORS = {
    "no_hand_detected": "No hand detected in image",
    "detection_failed": "Hand detection failed",
    "roi_extraction_failed": "Failed to extract palm ROI",
    "image_too_blurry": "Image is too blurry",
    "poor_lighting": "Insufficient lighting",
    "hand_too_small": "Hand is too small in frame",
    "multiple_hands": "Multiple hands detected",
}
