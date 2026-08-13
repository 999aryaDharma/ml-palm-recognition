"""
Enrollment Service Module
Handles biometric template enrollment using model_id provenance.
Strictly hard-fails with KeyError if an explicit invalid model_id is requested.
"""
from PIL import Image
import numpy as np
import time


class EnrollmentService:
    """Service for enrolling biometric templates."""

    def __init__(self, app_state, model_id: str | None = None):
        self.detector = getattr(app_state, "detector", None)
        self.registry = getattr(app_state, "registry", None)
        self.settings = getattr(app_state, "settings", None)

        default_id = self.settings.default_model_id if self.settings else "mobilefacenet-pretrained"

        if model_id is not None and model_id != "":
            if not self.registry or not self.registry.is_available(model_id):
                raise KeyError(f"model_not_found:{model_id}")
            target_model_id = model_id
        else:
            target_model_id = default_id

        if self.registry and self.registry.is_available(target_model_id):
            self.runtime = self.registry.get(target_model_id)
            self.model_id = self.runtime.model_id
            self.model_version = self.runtime.version
            self.recognizer = self.runtime
        else:
            self.recognizer = getattr(app_state, "recognizer", None)
            self.model_id = target_model_id
            self.model_version = "1.0.0"

    def process_template(self, image: Image.Image) -> tuple[np.ndarray, float, str, str, str]:
        """Process image for enrollment.

        Returns:
            Tuple of (embedding, quality_score, quality_status, model_id, model_version)
        """
        start_time = time.time()

        if self.detector is None:
            raise ValueError("detection_failed")

        detection_result = self.detector.detect(image)
        if detection_result is None:
            raise ValueError("no_hand_detected")

        from ml.roi import extract_palm_roi
        palm_roi = extract_palm_roi(image, detection_result)
        if palm_roi is None:
            raise ValueError("roi_extraction_failed")

        if self.recognizer is None:
            raise ValueError("detection_failed")

        embedding = self.recognizer.extract_embedding(palm_roi)
        if embedding is None:
            raise ValueError("image_too_blurry")

        from ml.quality import assess_image_quality
        quality_status, quality_score = assess_image_quality(image, detection_result)

        return embedding, quality_score, quality_status, self.model_id, self.model_version
