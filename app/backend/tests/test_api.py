"""
Integration tests for Palm Biometric API.
All ML calls are mocked — no model files required.
"""
import io
import numpy as np
from PIL import Image
import pytest


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_jpeg_bytes(w=320, h=240) -> bytes:
    img = Image.new("RGB", (w, h), color=(200, 150, 100))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


def _create_user(client, name="Test User") -> int:
    r = client.post("/users", json={"name": name})
    assert r.status_code == 200, r.text
    return r.json()["id"]


def _add_templates(client, user_id: int, n: int = 5):
    """Upload n templates — mocked pipeline always succeeds."""
    for _ in range(n):
        r = client.post(
            f"/users/{user_id}/templates",
            files={"image": ("palm.jpg", _make_jpeg_bytes(), "image/jpeg")},
        )
        assert r.status_code == 200, r.text


# ── Health ────────────────────────────────────────────────────────────────────

class TestHealth:
    def test_returns_ok(self, client):
        r = client.get("/health")
        assert r.status_code == 200
        data = r.json()
        assert data["status"] == "ok"
        assert "model_loaded" in data
        assert "detector_loaded" in data
        assert data["cached_users"] >= 0


# ── Users CRUD ────────────────────────────────────────────────────────────────

class TestUsers:
    def test_create_user(self, client):
        r = client.post("/users", json={"name": "Budi Santoso"})
        assert r.status_code == 200
        d = r.json()
        assert d["name"] == "Budi Santoso"
        assert d["template_count"] == 0
        assert "id" in d

    def test_list_users(self, client):
        _create_user(client, "Alice")
        _create_user(client, "Bob")
        r = client.get("/users")
        assert r.status_code == 200
        assert len(r.json()) == 2

    def test_get_user(self, client):
        uid = _create_user(client, "Charlie")
        r = client.get(f"/users/{uid}")
        assert r.status_code == 200
        assert r.json()["name"] == "Charlie"

    def test_get_user_not_found(self, client):
        r = client.get("/users/99999")
        assert r.status_code == 404
        assert r.json()["detail"]["error"] == "user_not_found"

    def test_delete_user(self, client):
        uid = _create_user(client)
        r = client.delete(f"/users/{uid}")
        assert r.status_code == 200
        assert r.json()["deleted"] is True
        # Verify gone
        assert client.get(f"/users/{uid}").status_code == 404

    def test_create_user_name_too_short(self, client):
        r = client.post("/users", json={"name": "A"})
        assert r.status_code == 422

    def test_template_upload_success(self, client):
        uid = _create_user(client)
        r = client.post(
            f"/users/{uid}/templates",
            files={"image": ("palm.jpg", _make_jpeg_bytes(), "image/jpeg")},
        )
        assert r.status_code == 200
        d = r.json()
        assert "template_id" in d
        assert d["quality_score"] >= 0

    def test_template_upload_user_not_found(self, client):
        r = client.post(
            "/users/99999/templates",
            files={"image": ("palm.jpg", _make_jpeg_bytes(), "image/jpeg")},
        )
        assert r.status_code == 404


# ── Identification ────────────────────────────────────────────────────────────

class TestIdentification:
    def test_identify_unknown_when_no_users(self, client):
        """With no enrolled users, identify must return 'unknown' (or error)."""
        r = client.post(
            "/identify",
            files={"image": ("palm.jpg", _make_jpeg_bytes(), "image/jpeg")},
        )
        # Either unknown or no_templates_enrolled error — both acceptable
        assert r.status_code in (200, 400)
        if r.status_code == 200:
            assert r.json()["status"] == "unknown"
        else:
            assert r.json()["detail"]["error"] == "no_templates_enrolled"

    def test_identify_returns_identified_after_enroll(self, client):
        """Seed embeddings that will match (all ones) and verify identification works."""
        uid = _create_user(client, "Identifiable User")
        _add_templates(client, uid)

        # Patch cache with the seeded embeddings so cosine similarity is 1.0
        from ml.cache import EmbeddingCache
        emb = np.ones(128, dtype=np.float32)
        emb /= np.linalg.norm(emb)
        client.app.state.cache._users = [{
            "user_id": uid,
            "user_name": "Identifiable User",
            "embeddings": [emb] * 5,
        }]

        r = client.post(
            "/identify",
            files={"image": ("palm.jpg", _make_jpeg_bytes(), "image/jpeg")},
        )
        assert r.status_code == 200
        d = r.json()
        assert d["status"] == "identified"
        assert d["user"]["id"] == uid
        assert d["score"] > 0.5
        assert d["latency_ms"] >= 0

    def test_identify_invalid_image(self, client):
        r = client.post(
            "/identify",
            files={"image": ("bad.txt", b"not an image", "text/plain")},
        )
        assert r.status_code in (400, 422)


# ── Demo Logs ─────────────────────────────────────────────────────────────────

class TestDemoLogs:
    def test_empty_logs(self, client):
        r = client.get("/demo-logs")
        assert r.status_code == 200
        assert r.json() == []

    def test_filter_by_type(self, client):
        uid = _create_user(client)
        client.post("/demos/payment/pay", json={
            "user_id": uid, "amount": 50000, "merchant": "Test", "match_score": 0.9
        })
        r = client.get("/demo-logs?demo_type=payment")
        assert r.status_code == 200
        logs = r.json()
        assert all(l["demo_type"] == "payment" for l in logs)


# ── Demo Endpoints ────────────────────────────────────────────────────────────

class TestDemos:
    def test_payment_pay(self, client):
        uid = _create_user(client)
        r = client.post("/demos/payment/pay", json={
            "user_id": uid, "amount": 127500, "merchant": "Toko Maju Jaya", "match_score": 0.87
        })
        assert r.status_code == 200
        d = r.json()
        assert d["status"] == "success"
        assert d["transaction_id"].startswith("PAY-")
        assert d["amount"] == 127500

    def test_payment_user_not_found(self, client):
        r = client.post("/demos/payment/pay", json={
            "user_id": 99999, "amount": 100, "merchant": "Test", "match_score": 0.9
        })
        assert r.status_code == 404

    def test_attendance_checkin(self, client):
        uid = _create_user(client)
        r = client.post("/demos/attendance/checkin", json={
            "user_id": uid, "mode": "checkin", "match_score": 0.85
        })
        assert r.status_code == 200
        assert r.json()["mode"] == "checkin"

    def test_attendance_checkout(self, client):
        uid = _create_user(client)
        r = client.post("/demos/attendance/checkin", json={
            "user_id": uid, "mode": "checkout", "match_score": 0.85
        })
        assert r.status_code == 200
        assert r.json()["mode"] == "checkout"

    def test_attendance_invalid_mode(self, client):
        uid = _create_user(client)
        r = client.post("/demos/attendance/checkin", json={
            "user_id": uid, "mode": "sleep", "match_score": 0.8
        })
        assert r.status_code == 422

    def test_patient_checkin(self, client):
        uid = _create_user(client)
        r = client.post("/demos/patient/checkin", json={
            "user_id": uid, "match_score": 0.91
        })
        assert r.status_code == 200
        d = r.json()
        assert d["status"] == "success"
        assert "patient" in d
        assert "nik" in d["patient"]

    def test_access_authorized_list_empty(self, client):
        r = client.get("/demos/access/authorized")
        assert r.status_code == 200
        assert isinstance(r.json(), list)

    def test_access_toggle_authorization(self, client):
        uid = _create_user(client)
        # Authorize
        r = client.put(f"/demos/access/authorized/{uid}?authorized=true")
        assert r.status_code == 200
        assert r.json()["authorized"] is True
        # Revoke
        r = client.put(f"/demos/access/authorized/{uid}?authorized=false")
        assert r.status_code == 200
        assert r.json()["authorized"] is False


# ── Seed ──────────────────────────────────────────────────────────────────────

class TestSeed:
    def test_seed_creates_users(self, client):
        r = client.post("/seed-demo-data")
        assert r.status_code == 200
        d = r.json()
        assert d["seeded_users"] > 0
        users = client.get("/users").json()
        assert len(users) == d["seeded_users"]

    def test_seed_idempotent(self, client):
        r1 = client.post("/seed-demo-data")
        r2 = client.post("/seed-demo-data")
        assert r2.json()["seeded_users"] == 0  # already seeded

    def test_reset_deletes_all(self, client):
        client.post("/seed-demo-data")
        r = client.delete("/seed-demo-data")
        assert r.status_code == 200
        assert r.json()["deleted"] is True
        assert client.get("/users").json() == []