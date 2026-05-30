"""
Hand Detection Module
Implementation of MediaPipe hand detection for palm biometrics.
"""

from PIL import Image
from typing import Optional
import numpy as np

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
            running_mode=vision.RunningMode.IMAGE,
            num_hands=1,
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.model = vision.HandLandmarker.create_from_options(options)
        print(f"[HandDetector] Loaded MediaPipe model from: {model_path}")
    
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

        result = self.model.detect(mp_image)
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

        # Basic quality check: hand size in frame
        w, h = image.size
        xs = [lm["x"] * w for lm in landmarks]
        ys = [lm["y"] * h for lm in landmarks]
        bbox_diag = ((max(xs) - min(xs)) ** 2 + (max(ys) - min(ys)) ** 2) ** 0.5
        frame_diag = (w**2 + h**2) ** 0.5
        if bbox_diag < 0.15 * frame_diag:
            # Hand too small
            return None

        return {
            "landmarks": landmarks,
            "handedness": handedness,
            "image_size": image.size
        }
