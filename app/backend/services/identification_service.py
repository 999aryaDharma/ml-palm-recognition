"""
Identification Service Module
Handles biometric identification matching against templates
"""

from PIL import Image
import numpy as np
import time
from sqlalchemy.orm import Session


class IdentificationService:
    """Service for identifying users by palm biometric."""
    
    def __init__(self, app_state, db: Session = None):
        """
        Initialize identification service.
        
        Args:
            app_state: FastAPI app state with ML models
            db: SQLAlchemy session (optional)
        """
        self.detector = app_state.detector
        self.recognizer = app_state.recognizer
        self.cache = app_state.cache
        self.settings = app_state.settings
        self.db = db
    
    def identify_palm(self, image: Image.Image) -> tuple[dict, int]:
        """
        Identify user from palm image.
        
        Args:
            image: PIL Image of palm
            
        Returns:
            Tuple of (result_dict, latency_ms)
            result_dict: {status, user_id, user_name, score, quality_status}
        """
        start_time = time.time()
        
        try:
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
            
            query_embedding = self.recognizer.extract_embedding(palm_roi)
            if query_embedding is None:
                raise ValueError("image_too_blurry")
            
            # 4. Match against templates in cache
            if self.cache is None or self.cache.user_count == 0:
                raise ValueError("no_templates_enrolled")
            
            from ml.matcher import cosine_similarity
            
            best_user = None
            best_score = 0.0
            threshold = self.settings.default_threshold
            
            for user in self.cache.get_all():
                user_embeddings = user.get("embeddings", [])
                if not user_embeddings:
                    continue
                
                # Score all templates for this user
                scores = [
                    cosine_similarity(query_embedding, template)
                    for template in user_embeddings
                ]
                
                # Use top-k average
                top_k = min(self.settings.top_k_templates, len(scores))
                top_scores = sorted(scores, reverse=True)[:top_k]
                avg_score = np.mean(top_scores)
                
                if avg_score > best_score:
                    best_score = avg_score
                    best_user = user if avg_score >= threshold else None
            
            latency_ms = int((time.time() - start_time) * 1000)
            
            if best_user is not None:
                return {
                    "status": "identified",
                    "user_id": best_user["user_id"],
                    "user_name": best_user["user_name"],
                    "score": best_score,
                    "quality_status": "good"
                }, latency_ms
            else:
                return {
                    "status": "unknown",
                    "user_id": None,
                    "user_name": None,
                    "score": best_score,
                    "quality_status": "good"
                }, latency_ms
                
        except ValueError as e:
            latency_ms = int((time.time() - start_time) * 1000)
            error_code = str(e)
            return {
                "status": "error",
                "user_id": None,
                "user_name": None,
                "score": 0.0,
                "error_code": error_code,
                "quality_status": error_code
            }, latency_ms
