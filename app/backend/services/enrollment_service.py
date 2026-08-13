"""
Enrollment Service Module.

Shared capture preprocessing is performed once, while embeddings remain
model-specific. Single-model enrollment is preserved for backward compatibility.
"""
from PIL import Image
import numpy as np


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
            self.runtime = None
            self.recognizer = getattr(app_state, "recognizer", None)
            self.model_id = target_model_id
            self.model_version = "1.0.0"

    def prepare_sample(self, image: Image.Image) -> tuple[Image.Image, float, str]:
        """Run shared detection, ROI extraction, and quality assessment once."""
        if self.detector is None:
            raise ValueError("detection_failed")

        detection_result = self.detector.detect(image)
        if detection_result is None:
            raise ValueError("no_hand_detected")

        from ml.roi import extract_palm_roi

        landmarks = detection_result.get("landmarks")
        if not landmarks:
            raise ValueError("roi_extraction_failed")

        palm_roi = extract_palm_roi(image, landmarks)
        if palm_roi is None:
            raise ValueError("roi_extraction_failed")

        from ml.quality import assess_image_quality

        quality_status, quality_score = assess_image_quality(image, detection_result)
        return palm_roi, quality_score, quality_status

    def process_template(self, image: Image.Image) -> tuple[np.ndarray, float, str, str, str]:
        """Process one image for one requested model."""
        palm_roi, quality_score, quality_status = self.prepare_sample(image)

        if self.recognizer is None:
            raise ValueError("detection_failed")

        embedding = self.recognizer.extract_embedding(palm_roi)
        if embedding is None:
            raise ValueError("image_too_blurry")

        return embedding, quality_score, quality_status, self.model_id, self.model_version

    def process_template_all(self, image: Image.Image) -> tuple[list[dict], float, str]:
        """Create one embedding per active registry model from one shared ROI.

        No persistence happens here. If any active runtime fails, the whole capture
        is rejected so callers can keep multi-model persistence atomic.
        """
        if not self.registry:
            raise ValueError("backend_not_ready")

        available = self.registry.list_available()
        if not available:
            raise ValueError("backend_not_ready")

        palm_roi, quality_score, quality_status = self.prepare_sample(image)
        results: list[dict] = []

        for model_info in available:
            model_id = model_info["id"]
            runtime = self.registry.get(model_id)
            embedding = runtime.extract_embedding(palm_roi)
            if embedding is None:
                raise ValueError("image_too_blurry")
            results.append(
                {
                    "embedding": embedding,
                    "model_id": runtime.model_id,
                    "model_version": runtime.version,
                }
            )

        return results, quality_score, quality_status
