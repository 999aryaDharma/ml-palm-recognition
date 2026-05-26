"""
Enrollment Service Module
Handles biometric template enrollment
"""

from PIL import Image
import numpy as np
import io
import time


class EnrollmentService:
    """Service for enrolling biometric templates."""
    
    def __init__(self, app_state):
        """
        Initialize enrollment service.
        
        Args:
            app_state: FastAPI app state with ML models
        """
        self.detector = app_state.detector
        self.recognizer = app_state.recognizer
        self.settings = app_state.settings
    
    def process_template(self, image: Image.Image) -> tuple[np.ndarray, float, str]:
        """
        Process image for enrollment (detect hand, extract ROI, extract embedding).
        
        Args:
            image: PIL Image of palm
            
        Returns:
            Tuple of (embedding, quality_score, quality_status)
            
        Raises:
            ValueError: If processing fails
        """
        start_time = time.time()
        
        # 1. Detect hand
        if self.detector is None:
            raise ValueError("detection_failed")
        
        detection_result = self.detector.detect(image)
        if detection_result is None:
            raise ValueError("no_hand_detected")
        
        # 2. Extract palm ROI
        from ml.roi import extract_palm_roi
        palm_roi = extract_palm_roi(image, detection_result)
        if palm_roi is None:
            raise ValueError("roi_extraction_failed")
        
        # 3. Extract embedding
        if self.recognizer is None:
            raise ValueError("detection_failed")
        
        embedding = self.recognizer.extract_embedding(palm_roi)
        if embedding is None:
            raise ValueError("image_too_blurry")
        
        # 4. Assess quality
        from ml.quality import assess_image_quality
        quality_status, quality_score = assess_image_quality(image, detection_result)
        
        latency_ms = int((time.time() - start_time) * 1000)
        
        return embedding, quality_score, quality_status
