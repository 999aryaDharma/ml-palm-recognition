"""Regression coverage for one-capture multi-model enrollment fan-out."""
from unittest.mock import MagicMock

import numpy as np


def _runtime(model_id: str, version: str = "1.0.0"):
    runtime = MagicMock()
    runtime.model_id = model_id
    runtime.version = version
    emb = np.ones(128, dtype=np.float32)
    if model_id == "palmnet-lite-scratch":
        emb[0] = 2.0
    emb /= np.linalg.norm(emb)
    runtime.extract_embedding.return_value = emb
    return runtime


def _install_two_model_registry(client):
    mobile = _runtime("mobilefacenet-pretrained")
    palmnet = _runtime("palmnet-lite-scratch")
    runtimes = {
        mobile.model_id: mobile,
        palmnet.model_id: palmnet,
    }

    registry = MagicMock()
    registry.list_available.return_value = [
        {"id": mobile.model_id, "version": mobile.version},
        {"id": palmnet.model_id, "version": palmnet.version},
    ]
    registry.get.side_effect = lambda model_id: runtimes[model_id]
    registry.is_available.side_effect = lambda model_id: model_id in runtimes
    client.app.state.registry = registry
    return mobile, palmnet


def _jpeg_bytes():
    import io
    from PIL import Image

    buf = io.BytesIO()
    Image.new("RGB", (320, 240), color=(180, 140, 100)).save(buf, format="JPEG")
    return buf.getvalue()


def test_one_multi_upload_creates_template_for_every_active_model(client):
    _install_two_model_registry(client)
    user_id = client.post("/users", json={"name": "Multi Model One"}).json()["id"]

    response = client.post(
        f"/users/{user_id}/templates/multi",
        files={"image": ("palm.jpg", _jpeg_bytes(), "image/jpeg")},
    )

    assert response.status_code == 200, response.text
    data = response.json()
    assert data["model_count"] == 2
    assert {item["model_id"] for item in data["templates"]} == {
        "mobilefacenet-pretrained",
        "palmnet-lite-scratch",
    }

    from db.database import SessionLocal
    from db.models import Template

    db = SessionLocal()
    try:
        rows = db.query(Template).filter(Template.user_id == user_id).all()
        assert len(rows) == 2
        assert {(row.model_id, row.model_version) for row in rows} == {
            ("mobilefacenet-pretrained", "1.0.0"),
            ("palmnet-lite-scratch", "1.0.0"),
        }
    finally:
        db.close()


def test_five_multi_uploads_create_five_templates_per_model(client):
    _install_two_model_registry(client)
    user_id = client.post("/users", json={"name": "Multi Model Five"}).json()["id"]

    for _ in range(5):
        response = client.post(
            f"/users/{user_id}/templates/multi",
            files={"image": ("palm.jpg", _jpeg_bytes(), "image/jpeg")},
        )
        assert response.status_code == 200, response.text

    ready = client.get(f"/users/{user_id}/verify-ready-all")
    assert ready.status_code == 200, ready.text
    data = ready.json()
    assert data["ready"] is True
    assert data["required"] == 5
    assert data["models"]["mobilefacenet-pretrained"]["template_count"] == 5
    assert data["models"]["palmnet-lite-scratch"]["template_count"] == 5


def test_verify_ready_all_false_when_one_model_is_missing_templates(client):
    _install_two_model_registry(client)
    user_id = client.post("/users", json={"name": "Partial Model User"}).json()["id"]

    for _ in range(5):
        response = client.post(
            f"/users/{user_id}/templates",
            data={"model_id": "mobilefacenet-pretrained"},
            files={"image": ("palm.jpg", _jpeg_bytes(), "image/jpeg")},
        )
        assert response.status_code == 200, response.text

    ready = client.get(f"/users/{user_id}/verify-ready-all")
    assert ready.status_code == 200, ready.text
    data = ready.json()
    assert data["ready"] is False
    assert data["models"]["mobilefacenet-pretrained"]["template_count"] == 5
    assert data["models"]["palmnet-lite-scratch"]["template_count"] == 0


def test_shared_roi_is_extracted_once_for_multi_model_upload(client, monkeypatch):
    mobile, palmnet = _install_two_model_registry(client)
    user_id = client.post("/users", json={"name": "Shared ROI User"}).json()["id"]

    import ml.roi

    calls = {"count": 0}
    original = ml.roi.extract_palm_roi

    def counting_extract(*args, **kwargs):
        calls["count"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(ml.roi, "extract_palm_roi", counting_extract)

    response = client.post(
        f"/users/{user_id}/templates/multi",
        files={"image": ("palm.jpg", _jpeg_bytes(), "image/jpeg")},
    )

    assert response.status_code == 200, response.text
    assert calls["count"] == 1
    mobile.extract_embedding.assert_called_once()
    palmnet.extract_embedding.assert_called_once()
