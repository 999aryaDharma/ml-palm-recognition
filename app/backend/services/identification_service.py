"""
Identification Service
Full biometric pipeline: detection → ROI → embedding → cosine matching.
"""
import time
import numpy as np
from PIL import Image
from sqlalchemy.orm import Session


class IdentificationService:
    def __init__(self, app_state, db: Session | None = None):
        self.detector   = getattr(app_state, "detector",   None)
        self.recognizer = getattr(app_state, "recognizer", None)
        self.cache      = getattr(app_state, "cache",      None)
        self.settings   = getattr(app_state, "settings",   None)
        self.db         = db

    def identify_palm(self, image: Image.Image) -> tuple[dict, int]:
        """
        Run the full biometric pipeline on a PIL image.

        Returns:
            (result_dict, latency_ms)

        result_dict keys:
            status      – "identified" | "unknown" | "error"
            user_id     – int or None
            user_name   – str or None
            score       – float (0–1)
            error_code  – str (only when status == "error")
        """
        start_ms = time.time() * 1000

        def _err(code: str):
            return {"status": "error", "user_id": None, "user_name": None,
                    "score": 0.0, "error_code": code}

        # ── Guards ────────────────────────────────────────────────────────────
        if self.detector is None or self.recognizer is None:
            return _err("backend_not_ready"), 0

        # ── Stage 1: Hand detection ───────────────────────────────────────────
        detection = self.detector.detect(image)
        if detection is None:
            return _err("detection_failed"), int(time.time() * 1000 - start_ms)

        # ── Stage 2a: ROI extraction ──────────────────────────────────────────
        from ml.roi import extract_palm_roi
        roi = extract_palm_roi(image, detection["landmarks"])
        if roi is None:
            return _err("roi_extraction_failed"), int(time.time() * 1000 - start_ms)

        # ── Stage 2b: Embedding ───────────────────────────────────────────────
        embedding = self.recognizer.extract_embedding(roi)
        if embedding is None:
            return _err("image_too_blurry"), int(time.time() * 1000 - start_ms)

        # ── Stage 3: Matching ─────────────────────────────────────────────────
        enrolled = self.cache.get_all() if self.cache else []
        if not enrolled:
            return _err("no_templates_enrolled"), int(time.time() * 1000 - start_ms)

        threshold = self.settings.default_threshold if self.settings else 0.70
        top_k     = self.settings.top_k_templates   if self.settings else 3

        best_score   = -1.0
        best_user_id = None
        best_name    = None

        for user in enrolled:
            embs = user.get("embeddings", [])
            if not embs:
                continue
            sims = sorted(
                [_cosine(embedding, e) for e in embs],
                reverse=True,
            )
            user_score = float(np.mean(sims[:min(top_k, len(sims))]))
            if user_score > best_score:
                best_score   = user_score
                best_user_id = user["user_id"]
                best_name    = user["user_name"]

        latency_ms = int(time.time() * 1000 - start_ms)

        if best_score >= threshold:
            return {
                "status":    "identified",
                "user_id":   best_user_id,
                "user_name": best_name,
                "score":     best_score,
            }, latency_ms

        return {
            "status":    "unknown",
            "user_id":   None,
            "user_name": None,
            "score":     max(0.0, best_score),
        }, latency_ms


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-8
    return float(np.dot(a, b) / denom)