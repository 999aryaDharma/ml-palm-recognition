import pytest
import numpy as np
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ── ML Mocks (applied before app import) ─────────────────────────────────────

def _make_detector_mock():
    mock = MagicMock()
    # Returns a valid detection result — 21 dummy landmarks
    lm = MagicMock()
    lm.x = lm.y = lm.z = 0.5
    lm.visibility = 1.0
    mock.detect.return_value = {"landmarks": [lm] * 21, "handedness": "Right"}
    mock.model = MagicMock()   # non-None → model_loaded = True in health endpoint
    return mock


def _make_recognizer_mock():
    mock = MagicMock()
    emb = np.ones(128, dtype=np.float32)
    emb /= np.linalg.norm(emb)
    mock.extract_embedding.return_value = emb
    mock.model = MagicMock()   # non-None
    return mock


def _make_roi_mock():
    from PIL import Image
    roi = Image.new("RGB", (112, 112), color=(128, 128, 128))
    return roi


@pytest.fixture(scope="session")
def app_with_mocks():
    """Create a FastAPI TestClient with ML services fully mocked."""
    with (
        patch("ml.detection.HandDetector.__init__", return_value=None),
        patch("ml.recognizer.PalmRecognizer.__init__", return_value=None),
        patch("ml.roi.extract_palm_roi", return_value=_make_roi_mock()),
        patch("os.path.exists", return_value=True),
    ):
        from main import app

        with TestClient(app) as client:
            # Inject mocks into app state
            client.app.state.detector   = _make_detector_mock()
            client.app.state.recognizer = _make_recognizer_mock()

            from ml.cache import EmbeddingCache
            client.app.state.cache = EmbeddingCache()

            yield client


@pytest.fixture
def client(app_with_mocks):
    return app_with_mocks


# ── Isolated DB for each test ─────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def clean_db():
    """Wipe all rows before each test for isolation."""
    from db.database import SessionLocal, Base, engine
    from db.models import User, Template, DemoLog
    db = SessionLocal()
    db.query(DemoLog).delete()
    db.query(Template).delete()
    db.query(User).delete()
    db.commit()
    db.close()
    yield