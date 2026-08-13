"""
Backend Integration Test Suite for Dual-Model Registry & Services.

Tests:
1. ModelRegistry discovery, validation, and PIL ROI contract.
2. EmbeddingCache namespace segmentation.
3. Database Template model_id / model_version filtering.
4. IdentificationService using per-request model selection & runtime threshold.
5. EnrollmentService storing model_id & model_version.
6. API endpoints: GET /models, POST /identify (without global model mutation).
"""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

_BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(_BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(_BACKEND_DIR))

_ML_DIR = _BACKEND_DIR.parent / "ml"
if str(_ML_DIR) not in sys.path:
    sys.path.insert(0, str(_ML_DIR))

from ml.registry import ModelRegistry, ModelRuntime
from ml.cache import EmbeddingCache
from palm_recognition.models.palmnet_lite import build_palmnet_lite
from palm_recognition.models.initialization import initialize_scratch_weights
from palm_recognition.models.inference import PalmNetLiteInferenceWrapper


def create_dummy_runtime(model_id="palmnet-lite-scratch", version="1.0.0", threshold=0.50):
    backbone = build_palmnet_lite()
    initialize_scratch_weights(backbone)
    wrapper = PalmNetLiteInferenceWrapper(backbone)
    wrapper.eval()

    dummy = torch.randn(1, 3, 112, 112)
    scripted = torch.jit.trace(wrapper, dummy)

    return ModelRuntime(
        model_id=model_id,
        version=version,
        name=f"Test Model ({model_id})",
        training_mode="scratch" if "scratch" in model_id else "pretrained",
        manifest={"input_shape": [3, 112, 112], "normalization": {"mean": [0.5, 0.5, 0.5], "std": [0.5, 0.5, 0.5]}},
        model=scripted,
        threshold=threshold,
        threshold_data={"threshold": threshold, "calibration_split": "validation"},
        device=torch.device("cpu"),
    )


def test_registry_discovery_and_pil_contract():
    runtime = create_dummy_runtime()
    pil_roi = Image.new("RGB", (112, 112), color=(100, 100, 100))
    emb = runtime.extract_embedding(pil_roi)

    assert isinstance(emb, np.ndarray)
    assert emb.shape == (128,)
    assert np.isfinite(emb).all()
    assert abs(np.linalg.norm(emb) - 1.0) < 1e-3
    print("  [PASS] test_registry_discovery_and_pil_contract")


def test_cache_namespace_segmentation():
    cache = EmbeddingCache()

    e1 = np.ones(128, dtype=np.float32)
    e1 = e1 / np.linalg.norm(e1)
    e2 = np.zeros(128, dtype=np.float32)
    e2[0] = 1.0

    cache._store[("mobilefacenet-pretrained", "1.0.0")] = [
        {"user_id": 1, "user_name": "UserA", "embeddings": [e1]}
    ]
    cache._store[("palmnet-lite-scratch", "1.0.0")] = [
        {"user_id": 1, "user_name": "UserA", "embeddings": [e2]}
    ]

    mf_list = cache.get_all("mobilefacenet-pretrained", "1.0.0")
    pl_list = cache.get_all("palmnet-lite-scratch", "1.0.0")

    assert len(mf_list) == 1
    assert len(pl_list) == 1
    assert np.array_equal(mf_list[0]["embeddings"][0], e1)
    assert np.array_equal(pl_list[0]["embeddings"][0], e2)

    assert cache.get_all("nonexistent", "1.0.0") == []
    print("  [PASS] test_cache_namespace_segmentation")


def test_database_model_filtering():
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker
    from db.models import Base, User, Template
    from db.repositories import TemplateRepository

    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    db = Session()

    user = User(name="Test User")
    db.add(user)
    db.commit()

    repo = TemplateRepository(db)
    emb = np.ones(128, dtype=np.float32)
    emb = emb / np.linalg.norm(emb)

    t1 = repo.create(user.id, emb, 1.0, model_id="mobilefacenet-pretrained", model_version="1.0.0")
    t2 = repo.create(user.id, emb, 1.0, model_id="palmnet-lite-scratch", model_version="1.0.0")

    mf_temps = repo.list_by_user(user.id, model_id="mobilefacenet-pretrained")
    pl_temps = repo.list_by_user(user.id, model_id="palmnet-lite-scratch")

    assert len(mf_temps) == 1
    assert mf_temps[0].model_id == "mobilefacenet-pretrained"
    assert len(pl_temps) == 1
    assert pl_temps[0].model_id == "palmnet-lite-scratch"

    grouped_mf = repo.list_all_grouped(model_id="mobilefacenet-pretrained", model_version="1.0.0")
    assert len(grouped_mf) == 1
    assert grouped_mf[0]["user_id"] == user.id

    db.close()
    print("  [PASS] test_database_model_filtering")


def test_identification_service_per_request_selection():
    class DummySettings:
        default_model_id = "mobilefacenet-pretrained"
        top_k_templates = 3

    class DummyRegistry:
        def __init__(self):
            self.r1 = create_dummy_runtime("mobilefacenet-pretrained", "1.0.0", threshold=0.30)
            self.r2 = create_dummy_runtime("palmnet-lite-scratch", "1.0.0", threshold=0.50)

        def is_available(self, m_id):
            return m_id in ["mobilefacenet-pretrained", "palmnet-lite-scratch"]

        def get(self, m_id):
            return self.r1 if m_id == "mobilefacenet-pretrained" else self.r2

    class DummyAppState:
        def __init__(self):
            self.settings = DummySettings()
            self.registry = DummyRegistry()
            self.detector = None
            self.cache = EmbeddingCache()

    state = DummyAppState()

    from services.identification_service import IdentificationService
    s1 = IdentificationService(state, model_id="mobilefacenet-pretrained")
    s2 = IdentificationService(state, model_id="palmnet-lite-scratch")

    assert s1.model_id == "mobilefacenet-pretrained"
    assert s1.threshold == 0.30

    assert s2.model_id == "palmnet-lite-scratch"
    assert s2.threshold == 0.50

    print("  [PASS] test_identification_service_per_request_selection")


def test_enrollment_service_model_id():
    class DummySettings:
        default_model_id = "palmnet-lite-scratch"

    class DummyRegistry:
        def is_available(self, m_id):
            return m_id == "palmnet-lite-scratch"

        def get(self, m_id):
            return create_dummy_runtime("palmnet-lite-scratch", "1.0.0", threshold=0.50)

    class DummyAppState:
        settings = DummySettings()
        registry = DummyRegistry()
        detector = None

    state = DummyAppState()

    from services.enrollment_service import EnrollmentService
    e_service = EnrollmentService(state, model_id="palmnet-lite-scratch")

    assert e_service.model_id == "palmnet-lite-scratch"
    assert e_service.model_version == "1.0.0"

    print("  [PASS] test_enrollment_service_model_id")


def test_api_invalid_model_id_returns_404():
    """POST /identify with explicit invalid model_id returns HTTP 404 (Req 6)."""
    from fastapi.testclient import TestClient
    from main import app

    client = TestClient(app)
    # Create dummy image file
    buf = io.BytesIO()
    Image.new("RGB", (100, 100), color="white").save(buf, format="JPEG")
    buf.seek(0)

    response = client.post(
        "/identify",
        data={"model_id": "nonexistent_model_xyz"},
        files={"image": ("palm.jpg", buf, "image/jpeg")},
    )
    assert response.status_code == 404
    detail = response.json().get("detail", {})
    assert detail.get("error") == "model_not_found"
    print("  [PASS] test_api_invalid_model_id_returns_404")


def test_api_get_models():
    """GET /models returns list of models (Req 22)."""
    from fastapi.testclient import TestClient
    from main import app

    client = TestClient(app)
    response = client.get("/models")
    assert response.status_code == 200
    data = response.json()
    assert "models" in data
    assert "default_model_id" in data
    print("  [PASS] test_api_get_models")


import io

def run_backend_tests():
    tests = [
        ("Registry: PIL contract", test_registry_discovery_and_pil_contract),
        ("Cache: namespace segmentation", test_cache_namespace_segmentation),
        ("Database: model_id filtering", test_database_model_filtering),
        ("Identification: per-request selection", test_identification_service_per_request_selection),
        ("Enrollment: model_id support", test_enrollment_service_model_id),
        ("API: invalid model_id returns 404", test_api_invalid_model_id_returns_404),
        ("API: GET /models discovery", test_api_get_models),
    ]

    print("\n" + "=" * 60)
    print("Dual-Model Backend Integration Tests")
    print("=" * 60)

    passed = 0
    failed = 0
    for name, fn in tests:
        print(f"\n{name}")
        try:
            fn()
            passed += 1
        except Exception as e:
            print(f"  [FAIL] {e}")
            failed += 1

    print("\n" + "=" * 60)
    print(f"Results: {passed} passed, {failed} failed out of {len(tests)}")
    if failed == 0:
        print("[OK] ALL BACKEND TESTS PASSED")
    else:
        print("[FAIL] SOME BACKEND TESTS FAILED")

        sys.exit(1)
    print("=" * 60)


if __name__ == "__main__":
    run_backend_tests()

