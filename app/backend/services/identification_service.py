"""
Identification Service
Full biometric pipeline: detection → ROI → embedding → cosine matching.
Uses per-request model selection via ModelRegistry.
Strictly hard-fails with KeyError if an explicit invalid model_id is requested.
"""
import time
import numpy as np
from PIL import Image
from sqlalchemy.orm import Session


class IdentificationService:
    def __init__(
        self,
        app_state,
        db: Session | None = None,
        model_id: str | None = None,
    ):
        self.detector = getattr(app_state, "detector", None)
        self.registry = getattr(app_state, "registry", None)
        self.cache = getattr(app_state, "cache", None)
        self.settings = getattr(app_state, "settings", None)
        self.db = db

        default_id = self.settings.default_model_id if self.settings else "mobilefacenet-pretrained"

        if model_id is not None and model_id != "":
            # Explicit model_id requested — MUST exist in registry or hard fail 404
            if not self.registry or not self.registry.is_available(model_id):
                raise KeyError(f"model_not_found:{model_id}")
            target_model_id = model_id
        else:
            # Fallback ONLY when model_id was not explicitly specified
            target_model_id = default_id

        if self.registry and self.registry.is_available(target_model_id):
            self.runtime = self.registry.get(target_model_id)
            self.model_id = self.runtime.model_id
            self.model_version = self.runtime.version
            self.threshold = float(self.runtime.threshold)
            self.recognizer = self.runtime
        else:
            self.recognizer = getattr(app_state, "recognizer", None)
            self.model_id = target_model_id
            self.model_version = "1.0.0"
            self.threshold = float(getattr(self.settings, "default_threshold", 0.50)) if self.settings else 0.50

    def identify_palm(self, image: Image.Image) -> tuple[dict, int]:
        """Run the full biometric pipeline on a PIL image."""
        start_ms = time.time() * 1000

        def _err(code: str, detection_data: dict | None = None):
            res = {
                "status": "error",
                "user_id": None,
                "user_name": None,
                "score": 0.0,
                "error_code": code,
                "model_id": self.model_id,
                "model_version": self.model_version,
            }
            if detection_data:
                res["bbox"] = detection_data.get("bbox")
                res["landmarks"] = detection_data.get("landmarks")
            return res, int(time.time() * 1000 - start_ms)

        if self.detector is None or self.recognizer is None:
            return _err("backend_not_ready")

        detection = self.detector.detect(image)
        if detection is None:
            return _err("detection_failed")

        from ml.roi import extract_palm_roi
        roi = extract_palm_roi(image, detection["landmarks"])
        if roi is None:
            return _err("roi_extraction_failed", detection)

        embedding = self.recognizer.extract_embedding(roi)
        if embedding is None:
            return _err("image_too_blurry", detection)

        enrolled = self.cache.get_all(self.model_id, self.model_version) if self.cache else []
        if not enrolled:
            return _err("no_templates_enrolled", detection)

        valid_enrolled = [u for u in enrolled if len(u.get("embeddings", [])) >= 1]
        if not valid_enrolled:
            return _err("no_templates_enrolled", detection)

        top_k = getattr(self.settings, "top_k_templates", 3) if self.settings else 3
        best_raw_score = -1.0
        best_user_id = None
        best_name = None

        for user in valid_enrolled:
            embs = user.get("embeddings", [])
            sims = sorted([_cosine(embedding, e) for e in embs], reverse=True)
            user_raw_score = float(np.mean(sims[:min(top_k, len(sims))]))
            if user_raw_score > best_raw_score:
                best_raw_score = user_raw_score
                best_user_id = user["user_id"]
                best_name = user["user_name"]

        latency_ms = int(time.time() * 1000 - start_ms)

        if best_raw_score >= self.threshold:
            return {
                "status": "identified",
                "user_id": best_user_id,
                "user_name": best_name,
                "score": best_raw_score,
                "raw_score": best_raw_score,
                "threshold_used": self.threshold,
                "model_id": self.model_id,
                "model_version": self.model_version,
                "bbox": detection.get("bbox"),
                "landmarks": detection.get("landmarks"),
                "quality_score": 1.0,
            }, latency_ms

        return {
            "status": "unknown",
            "user_id": None,
            "user_name": None,
            "score": best_raw_score,
            "raw_score": best_raw_score,
            "threshold_used": self.threshold,
            "model_id": self.model_id,
            "model_version": self.model_version,
            "bbox": detection.get("bbox"),
            "landmarks": detection.get("landmarks"),
            "quality_score": 1.0,
        }, latency_ms


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-8
    return float(np.dot(a, b) / denom)
