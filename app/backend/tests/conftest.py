import pytest
import numpy as np
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient

import sys
import os
backend_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
parent_dir = os.path.dirname(backend_dir)
# Remove parent_dir from sys.path if present to prevent it from shadowing backend packages (like ml)
sys.path = [p for p in sys.path if p and os.path.abspath(p) != os.path.abspath(parent_dir)]
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)

# Clean cached 'ml' from sys.modules if it was loaded from the parent directory
if "ml" in sys.modules:
    del sys.modules["ml"]


# ── ML Mocks (applied before app import) ─────────────────────────────────────

def _make_detector_mock():
    mock = MagicMock()
    # Returns a valid detection result — 21 dummy landmarks
    landmarks = [
        {"x": 0.5, "y": 0.5, "z": 0.5, "visibility": 1.0}
        for _ in range(21)
    ]
    mock.detect.return_value = {"landmarks": landmarks, "handedness": "Right"}
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
def clean_db(app_with_mocks):
    """Wipe all rows before each test for isolation."""
    from db.database import SessionLocal, Base, engine
    from db.models import User, Template, DemoLog
    db = SessionLocal()
    db.query(DemoLog).delete()
    db.query(Template).delete()
    db.query(User).delete()
    db.commit()
    app_with_mocks.app.state.cache._store.clear()  # Clear cache

    db.close()
    yield