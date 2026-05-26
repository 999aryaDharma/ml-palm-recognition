"""
Palm ROI Extraction Module
Extracts Region of Interest (palm area) from detected hand
"""

from PIL import Image
from typing import Optional


def extract_palm_roi(image: Image.Image, detection_result: dict) -> Optional[Image.Image]:
    """
    Extract palm region of interest from detected hand.
    
    Args:
        image: Original PIL Image
        detection_result: Detection result with bounding box
        
    Returns:
        Cropped PIL Image of palm region, or None if extraction fails
    """
    # TODO: Implement actual ROI extraction with landmark-based refinement
    # For now, return None to indicate ROI extraction not implemented
    return None
