"""
Identification Service
Full biometric pipeline: detection → ROI → embedding → cosine matching.
"""
import time
import os
import json
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
        
        # Load threshold from threshold.json if available, fallback to settings
        self.threshold = self._load_threshold()

    def _load_threshold(self) -> float:
        """Load calibrated threshold from threshold.json."""
        if not self.settings or not os.path.exists(self.settings.threshold_path):
            return self.settings.default_threshold if self.settings else 0.70
            
        try:
            with open(self.settings.threshold_path, "r") as f:
                data = json.load(f)
                # EER threshold dari kalibrasi (biasanya ~0.25 - 0.30)
                calibrated = data.get("threshold", 0.25)
                # Untuk demo, kita naikkan sedikit supaya lebih 'aman' 
                # tapi tetap jauh di bawah 0.70 yang lama.
                return max(0.40, calibrated + 0.10) 
        except Exception:
            return self.settings.default_threshold if self.settings else 0.70

    def _map_to_confidence(self, raw_score: float) -> float:
        """
        Map raw cosine similarity to a user-friendly Confidence Score (0-1).
        Target: 
        - raw_score == self.threshold -> ~75% (Borderline success)
        - raw_score >= 0.60 -> 90%+
        - raw_score >= 0.80 -> 98%+
        """
        threshold = self.threshold
        if raw_score >= 1.0: return 0.999
        if raw_score <= 0.0: return 0.0

        if raw_score >= threshold:
            # Map [threshold, 1.0] to [0.75, 1.0]
            # Menggunakan power 0.7 agar score naik lebih cepat di awal (sensasi 90%)
            norm = (raw_score - threshold) / (1.0 - threshold)
            return 0.75 + (norm ** 0.7) * 0.24
        else:
            # Map [0, threshold] to [0, 0.74]
            norm = raw_score / threshold
            return (norm ** 1.5) * 0.74

    def identify_palm(self, image: Image.Image) -> tuple[dict, int]:
        """
        Run the full biometric pipeline on a PIL image.
        """
        start_ms = time.time() * 1000

        def _err(code: str, detection_data: dict | None = None):
            res = {"status": "error", "user_id": None, "user_name": None,
                    "score": 0.0, "error_code": code}
            if detection_data:
                res["bbox"] = detection_data.get("bbox")
                res["landmarks"] = detection_data.get("landmarks")
            return res, int(time.time() * 1000 - start_ms)

        # ── Guards ────────────────────────────────────────────────────────────
        if self.detector is None or self.recognizer is None:
            return _err("backend_not_ready")

        # ── Stage 1: Hand detection ───────────────────────────────────────────
        detection = self.detector.detect(image)
        if detection is None:
            return _err("detection_failed")

        # ── Stage 2a: ROI extraction ──────────────────────────────────────────
        from ml.roi import extract_palm_roi
        roi = extract_palm_roi(image, detection["landmarks"])
        if roi is None:
            return _err("roi_extraction_failed", detection)

        # ── Stage 2b: Embedding ───────────────────────────────────────────────
        embedding = self.recognizer.extract_embedding(roi)
        if embedding is None:
            return _err("image_too_blurry", detection)

        # ── Stage 3: Matching ─────────────────────────────────────────────────
        enrolled = self.cache.get_all() if self.cache else []
        if not enrolled:
            return _err("no_templates_enrolled", detection)

        valid_enrolled = [u for u in enrolled if len(u.get("embeddings", [])) >= 1]
        if not valid_enrolled:
            return _err("no_templates_enrolled", detection)

        # Matching Strategy: Average of Top-K best matches for each user
        # Ini memberikan score yang lebih stabil daripada hanya 1 Max.
        top_k = 3
        best_mapped_score = -1.0
        best_raw_score    = -1.0
        best_user_id      = None
        best_name         = None

        for user in valid_enrolled:
            embs = user.get("embeddings", [])
            # Hitung similarity untuk semua template user ini, urutkan dari yang terbesar
            sims = sorted([_cosine(embedding, e) for e in embs], reverse=True)
            
            # Ambil rata-rata dari top-K (biasanya 3)
            user_raw_score = float(np.mean(sims[:min(top_k, len(sims))]))
            
            if user_raw_score > best_raw_score:
                best_raw_score = user_raw_score
                best_user_id   = user["user_id"]
                best_name      = user["user_name"]

        # Map ke Confidence Score untuk kepuasan User (90% target)
        best_mapped_score = self._map_to_confidence(best_raw_score)
        
        latency_ms = int(time.time() * 1000 - start_ms)

        # Gunakan threshold hasil kalibrasi
        if best_raw_score >= self.threshold:
            return {
                "status":    "identified",
                "user_id":   best_user_id,
                "user_name": best_name,
                "score":     best_mapped_score, # Return MAPPED score to UI
                "raw_score": best_raw_score,    # Keep raw for logs
                "bbox":      detection.get("bbox"),
                "landmarks": detection.get("landmarks"),
                "quality_score": 1.0,
            }, latency_ms

        return {
            "status":    "unknown",
            "user_id":   None,
            "user_name": None,
            "score":     best_mapped_score,
            "raw_score": best_raw_score,
            "bbox":      detection.get("bbox"),
            "landmarks": detection.get("landmarks"),
            "quality_score": 1.0,
        }, latency_ms


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-8
    return float(np.dot(a, b) / denom)
