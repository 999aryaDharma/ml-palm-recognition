"""
Hand Detection Module
Implementation of MediaPipe hand detection for palm biometrics.
"""

from PIL import Image
from typing import Optional
import numpy as np
import time

try:
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision
    _MEDIAPIPE_AVAILABLE = True
except ImportError:
    _MEDIAPIPE_AVAILABLE = False


class HandDetector:
    """Detects hand landmarks in images using MediaPipe."""
    
    def __init__(self, model_path: str):
        """
        Initialize hand detector.
        
        Args:
            model_path: Path to hand_landmarker.task model
        """
        self.model_path = model_path
        self.model = None
        self._last_timestamp_ms = 0
        
        if not _MEDIAPIPE_AVAILABLE:
            print("[HandDetector] WARNING: mediapipe is not installed. Detection will fail.")
            return

        import os
        if not os.path.exists(model_path):
            print(f"[HandDetector] WARNING: Model file not found at {model_path}")
            return

        base_options = mp_python.BaseOptions(model_asset_path=model_path)
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.VIDEO,
            num_hands=1,
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.model = vision.HandLandmarker.create_from_options(options)
        print(f"[HandDetector] Loaded MediaPipe model in VIDEO mode from: {model_path}")
    
    def detect(self, image: Image.Image) -> Optional[dict]:
        """
        Detect hand in image.
        
        Args:
            image: PIL Image object
            
        Returns:
            Detection result with bounding box and landmarks, or None if no hand found
        """
        if self.model is None:
            return None

        rgb_array = np.array(image.convert("RGB"))
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_array)

        # Ensure monotonically increasing timestamps for VIDEO mode
        current_ms = int(time.time() * 1000)
        if current_ms <= self._last_timestamp_ms:
            current_ms = self._last_timestamp_ms + 1
        self._last_timestamp_ms = current_ms

        result = self.model.detect_for_video(mp_image, current_ms)
        if not result.hand_landmarks:
            return None

        raw_landmarks = result.hand_landmarks[0]
        landmarks = [
            {
                "x": float(p.x),
                "y": float(p.y),
                "z": float(p.z),
                "visibility": float(p.visibility) if p.visibility is not None else 1.0,
            }
            for p in raw_landmarks
        ]

        handedness = None
        if result.handedness:
            handedness = result.handedness[0][0].category_name

        # Calculate bounding box
        w, h = image.size
        xs = [lm["x"] * w for lm in landmarks]
        ys = [lm["y"] * h for lm in landmarks]
        
        xmin, xmax = min(xs), max(xs)
        ymin, ymax = min(ys), max(ys)
        
        # Add padding to bbox
        padding_w = (xmax - xmin) * 0.15
        padding_h = (ymax - ymin) * 0.15
        
        bbox = {
            "x": max(0, xmin - padding_w) / w,
            "y": max(0, ymin - padding_h) / h,
            "width": min(w, xmax - xmin + 2 * padding_w) / w,
            "height": min(h, ymax - ymin + 2 * padding_h) / h
        }

        bbox_diag = ((xmax - xmin) ** 2 + (ymax - ymin) ** 2) ** 0.5
        frame_diag = (w**2 + h**2) ** 0.5
        if bbox_diag < 0.15 * frame_diag:
            # Hand too small
            return None

        return {
            "landmarks": landmarks,
            "bbox": bbox,
            "handedness": handedness,
            "image_size": image.size
        }
