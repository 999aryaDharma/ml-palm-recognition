#!/usr/bin/env python
"""
backend/scripts/smoke_enroll_5_templates.py
============================================
Smoke test: verifikasi bahwa 5 template bisa disimpan ke DB per user
tanpa error SQLite lock.

Jalankan dari folder backend/:
    conda run -n ml python scripts/smoke_enroll_5_templates.py

Exit code 0 = PASS, 1 = FAIL.
"""
import sys
import io
import os
import time

# Pastikan import dari backend/
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from PIL import Image
from unittest.mock import MagicMock, patch

# ─── Setup app + DB ──────────────────────────────────────────────────────────
def _make_jpeg_bytes(w: int = 320, h: int = 240) -> bytes:
    img = Image.new("RGB", (w, h), color=(190, 140, 100))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return buf.getvalue()


def _make_detector_mock():
    mock = MagicMock()
    landmarks = [{"x": 0.5, "y": 0.5, "z": 0.0, "visibility": 1.0} for _ in range(21)]
    mock.detect.return_value = {"landmarks": landmarks, "handedness": "Right"}
    mock.model = MagicMock()
    return mock


def _make_recognizer_mock():
    mock = MagicMock()
    emb = np.ones(128, dtype=np.float32)
    emb /= np.linalg.norm(emb)
    mock.extract_embedding.return_value = emb
    mock.model = MagicMock()
    return mock


def run():
    print("=" * 60)
    print("SMOKE TEST: 5-Template Enrollment")
    print("=" * 60)

    from fastapi.testclient import TestClient

    with (
        patch("ml.detection.HandDetector.__init__", return_value=None),
        patch("ml.recognizer.PalmRecognizer.__init__", return_value=None),
        patch("ml.roi.extract_palm_roi", return_value=Image.new("RGB", (112, 112))),
        patch("os.path.exists", return_value=True),
    ):
        from main import app

        with TestClient(app) as client:
            client.app.state.detector   = _make_detector_mock()
            client.app.state.recognizer = _make_recognizer_mock()

            from ml.cache import EmbeddingCache
            client.app.state.cache = EmbeddingCache()

            # Bersihkan DB
            from db.database import SessionLocal
            from db.models import User, Template, DemoLog
            db = SessionLocal()
            db.query(DemoLog).delete()
            db.query(Template).delete()
            db.query(User).delete()
            db.commit()
            client.app.state.cache._users = []
            db.close()

            # ── Test 1: Buat user ──────────────────────────────────────────
            print("\n[1] Membuat user...")
            r = client.post("/users", json={"name": "Smoke Test User"})
            assert r.status_code == 200, f"GAGAL buat user: {r.text}"
            user_id = r.json()["id"]
            print(f"    ✓ User dibuat → id={user_id}")

            # ── Test 2: Upload 5 template satu per satu ───────────────────
            print("\n[2] Mengunggah 5 template...")
            for i in range(5):
                jpeg = _make_jpeg_bytes()
                t0 = time.perf_counter()
                r = client.post(
                    f"/users/{user_id}/templates",
                    files={"image": ("palm.jpg", jpeg, "image/jpeg")},
                )
                elapsed = (time.perf_counter() - t0) * 1000
                assert r.status_code == 200, \
                    f"GAGAL upload template {i+1}: {r.text}"
                data = r.json()
                print(f"    ✓ Template {i+1}/5 → id={data['template_id']} "
                      f"(quality={data['quality_score']:.4f}, {elapsed:.0f}ms)")

            # ── Test 3: Verifikasi count di DB ────────────────────────────
            print("\n[3] Verifikasi via GET /users/{id}...")
            r = client.get(f"/users/{user_id}")
            assert r.status_code == 200, f"GAGAL get user: {r.text}"
            count = r.json()["template_count"]
            assert count == 5, f"GAGAL: Tersimpan {count}/5 template!"
            print(f"    ✓ template_count = {count}/5")

            # ── Test 4: Verifikasi via /verify-ready ──────────────────────
            print("\n[4] Verifikasi via /users/{id}/verify-ready...")
            r = client.get(f"/users/{user_id}/verify-ready")
            assert r.status_code == 200, f"GAGAL verify-ready: {r.text}"
            data = r.json()
            assert data["ready"] is True, f"GAGAL: ready={data['ready']}"
            assert data["template_count"] == 5, f"GAGAL: count={data['template_count']}"
            print(f"    ✓ ready=True, template_count={data['template_count']}")

            # ── Test 5: Cache berisi data yang benar ──────────────────────
            print("\n[5] Verifikasi in-memory cache...")
            cached = client.app.state.cache.get_user(user_id)
            assert cached is not None, "GAGAL: user tidak ada di cache!"
            emb_count = len(cached.get("embeddings", []))
            assert emb_count == 5, f"GAGAL: cache hanya punya {emb_count}/5 embeddings"
            print(f"    ✓ Cache memiliki {emb_count}/5 embeddings untuk user {user_id}")

            # ── Cleanup ───────────────────────────────────────────────────
            client.delete(f"/users/{user_id}")
            print("\n    (User smoke-test dihapus)")

    print("\n" + "=" * 60)
    print("HASIL: ✅ SEMUA TEST PASSED — 5 template tersimpan dengan benar")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(run())
