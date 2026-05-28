"""
Stage 1: Hand landmark extraction menggunakan MediaPipe Hand Landmarker.

Menghasilkan 21 landmark (x, y, z, visibility) per gambar palm.
Gambar yang lulus quality gate akan diteruskan ke ROI extractor.

Download model task file:
    https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker
    File: hand_landmarker.task (~7MB)
    Letakkan di: ml/models/hand_landmarker.task
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image

# MediaPipe imports — lazy untuk hindari issue di environment tanpa MediaPipe
try:
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision

    _MEDIAPIPE_AVAILABLE = True
except ImportError:  # pragma: no cover
    _MEDIAPIPE_AVAILABLE = False

from config import (
    HAND_LANDMARKER_PATH,
    MP_MIN_DETECTION_CONFIDENCE,
    MP_MIN_PRESENCE_CONFIDENCE,
    MP_MIN_TRACKING_CONFIDENCE,
    MP_NUM_HANDS,
    MP_LANDMARK_VISIBILITY_MIN,
    HAND_BBOX_MIN_FRAC,
)


class LandmarkExtractor:
    """
    Wrapper MediaPipe Hand Landmarker untuk single-image inference.

    Mode: IMAGE (single frame per call) — cocok untuk:
    - Offline preprocessing dataset
    - Backend FastAPI yang menerima frame via HTTP

    Untuk live webcam streaming, pakai mode LIVE_STREAM (tidak diimplementasi
    di sini karena backend kita stateless per request).
    """

    def __init__(self, model_path: str | Path = HAND_LANDMARKER_PATH):
        if not _MEDIAPIPE_AVAILABLE:
            raise ImportError(
                "MediaPipe tidak tersedia. Install dengan: pip install mediapipe==0.10.9"
            )

        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(
                f"Hand landmarker task file tidak ditemukan: {model_path}\n"
                f"Download dari: https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker"
            )

        base_options = mp_python.BaseOptions(model_asset_path=str(model_path))
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.IMAGE,
            num_hands=MP_NUM_HANDS,
            min_hand_detection_confidence=MP_MIN_DETECTION_CONFIDENCE,
            min_hand_presence_confidence=MP_MIN_PRESENCE_CONFIDENCE,
            min_tracking_confidence=MP_MIN_TRACKING_CONFIDENCE,
        )
        self.detector = vision.HandLandmarker.create_from_options(options)

    def extract(self, pil_image: Image.Image, run_quality_gate: bool = True) -> Optional[dict]:
        """
        Extract 21 landmark dari gambar.

        Args:
            pil_image: PIL Image (RGB atau bisa di-convert ke RGB)
            run_quality_gate: kalau True, return None untuk gambar yang gagal
                              quality check. Kalau False, return raw landmark
                              (berguna untuk debugging visualisasi).

        Returns:
            dict dengan:
                - 'landmarks': list of 21 dict {'x','y','z','visibility'}
                  (x,y,z normalized [0,1] terhadap ukuran gambar)
                - 'handedness': 'Left' atau 'Right' (dari MediaPipe)
                - 'image_size': (w, h) untuk konversi ke pixel coords
            None kalau tidak ada tangan terdeteksi atau gagal quality gate.
        """
        rgb_array = np.array(pil_image.convert("RGB"))
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_array)

        result = self.detector.detect(mp_image)
        if not result.hand_landmarks:
            return None

        raw_landmarks = result.hand_landmarks[0]  # ambil tangan pertama
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

        out = {
            "landmarks": landmarks,
            "handedness": handedness,
            "image_size": pil_image.size,  # (w, h)
        }

        if run_quality_gate and not self._quality_gate(landmarks, pil_image.size):
            return None

        return out

    @staticmethod
    def _quality_gate(landmarks: list[dict], image_size: tuple[int, int]) -> bool:
        """
        Quality checks:
        1. Semua 21 landmark visible (visibility >= threshold)
        2. Telapak menghadap kamera (heuristic: middle MCP z < middle TIP z
           berarti tangan menghadap ke belakang dari kamera → tolak)
        3. Tangan cukup besar di frame (hand bbox diagonal >= 25% frame diagonal)
        """
        # 1. Visibility check
        if any(lm["visibility"] < MP_LANDMARK_VISIBILITY_MIN for lm in landmarks):
            return False

        # 2. Palm facing camera (z lebih dekat → nilai z lebih kecil di MediaPipe)
        middle_mcp = landmarks[9]
        middle_tip = landmarks[12]
        if middle_mcp["z"] < middle_tip["z"]:
            # Palm facing AWAY (back of hand to camera) → reject
            return False

        # 3. Hand size in frame
        w, h = image_size
        xs = [lm["x"] * w for lm in landmarks]
        ys = [lm["y"] * h for lm in landmarks]
        bbox_diag = ((max(xs) - min(xs)) ** 2 + (max(ys) - min(ys)) ** 2) ** 0.5
        frame_diag = (w**2 + h**2) ** 0.5
        if bbox_diag < HAND_BBOX_MIN_FRAC * frame_diag:
            return False

        return True

    def close(self):
        """Cleanup resource MediaPipe."""
        if hasattr(self, "detector"):
            self.detector.close()


# Konvensi landmark MediaPipe (21 titik):
#   0: WRIST
#   1-4: THUMB (CMC, MCP, IP, TIP)
#   5-8: INDEX (MCP, PIP, DIP, TIP)
#   9-12: MIDDLE (MCP, PIP, DIP, TIP)
#   13-16: RING (MCP, PIP, DIP, TIP)
#   17-20: PINKY (MCP, PIP, DIP, TIP)
LANDMARK_NAMES = [
    "WRIST",
    "THUMB_CMC", "THUMB_MCP", "THUMB_IP", "THUMB_TIP",
    "INDEX_MCP", "INDEX_PIP", "INDEX_DIP", "INDEX_TIP",
    "MIDDLE_MCP", "MIDDLE_PIP", "MIDDLE_DIP", "MIDDLE_TIP",
    "RING_MCP", "RING_PIP", "RING_DIP", "RING_TIP",
    "PINKY_MCP", "PINKY_PIP", "PINKY_DIP", "PINKY_TIP",
]


if __name__ == "__main__":
    # Smoke test (butuh hand_landmarker.task tersedia di MODELS_DIR)
    import sys

    if len(sys.argv) > 1:
        img_path = sys.argv[1]
        extractor = LandmarkExtractor()
        img = Image.open(img_path)
        result = extractor.extract(img)
        if result is None:
            print(f"Tidak ada tangan terdeteksi di: {img_path}")
        else:
            print(f"Handedness: {result['handedness']}")
            print(f"Image size: {result['image_size']}")
            print(f"Landmark[0] (WRIST):   {result['landmarks'][0]}")
            print(f"Landmark[9] (MID_MCP): {result['landmarks'][9]}")
        extractor.close()
    else:
        print("Usage: python extract_landmarks.py <path_to_image>")
