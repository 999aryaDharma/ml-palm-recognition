"""
Image Quality Assessment Module
Validates palm image quality for biometric processing
"""

from PIL import Image
from typing import Tuple


def assess_image_quality(image: Image.Image, detection_result: dict) -> Tuple[str, float]:
    """
    Assess quality of palm image for biometric processing.
    
    Args:
        image: PIL Image object
        detection_result: Hand detection result
        
    Returns:
        Tuple of (quality_code, quality_score)
        quality_code: 'good', 'acceptable', 'poor'
        quality_score: Normalized score 0-1
    """
    # TODO: Implement actual quality assessment
    # Check for: blur, lighting, hand visibility, palm coverage, etc.
    
    # For now, return a default acceptable score
    return "acceptable", 0.80


QUALITY_ERRORS = {
    "no_hand_detected": "No hand detected in image",
    "detection_failed": "Hand detection failed",
    "roi_extraction_failed": "Failed to extract palm ROI",
    "image_too_blurry": "Image is too blurry",
    "poor_lighting": "Insufficient lighting",
    "hand_too_small": "Hand is too small in frame",
    "multiple_hands": "Multiple hands detected",
}
