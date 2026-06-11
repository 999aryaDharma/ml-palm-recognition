"""
Regression test for 5-template enrollment bug.
Ensures that all 5 templates are stored in database during enrollment.
"""
import io
import numpy as np
from PIL import Image
import pytest


def _make_jpeg_bytes(w=320, h=240) -> bytes:
    """Create a dummy JPEG image for testing."""
    img = Image.new("RGB", (w, h), color=(200, 150, 100))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


class TestEnrollment5Templates:
    """Test that enrollment correctly stores 5 templates."""
    
    def test_store_5_templates_in_db(self, client):
        """
        Regression test: Verify that 5 templates are stored in database.
        This test reproduces the bug where only 1 template was stored.
        """
        # 1. Create user
        user_name = "Test User 5 Templates"
        r = client.post("/users", json={"name": user_name})
        assert r.status_code == 200, f"Failed to create user: {r.text}"
        user_id = r.json()["id"]
        
        # 2. Upload 5 templates
        for i in range(5):
            r = client.post(
                f"/users/{user_id}/templates",
                files={"image": ("palm.jpg", _make_jpeg_bytes(), "image/jpeg")},
            )
            assert r.status_code == 200, f"Failed to upload template {i+1}: {r.text}"
            assert "template_id" in r.json()
        
        # 3. Verify count in database
        r = client.get(f"/users/{user_id}")
        assert r.status_code == 200
        data = r.json()
        assert data["template_count"] == 5, f"Expected 5 templates, got {data['template_count']}"
    
    def test_verify_ready_returns_true_after_5_templates(self, client):
        """
        Test that verify-ready endpoint returns ready=true after 5 templates.
        """
        # 1. Create user
        user_name = "Test User Verify Ready"
        r = client.post("/users", json={"name": user_name})
        assert r.status_code == 200
        user_id = r.json()["id"]
        
        # 2. Upload 5 templates
        for i in range(5):
            r = client.post(
                f"/users/{user_id}/templates",
                files={"image": ("palm.jpg", _make_jpeg_bytes(), "image/jpeg")},
            )
            assert r.status_code == 200
        
        # 3. Check verify-ready endpoint
        r = client.get(f"/users/{user_id}/verify-ready")
        assert r.status_code == 200
        data = r.json()
        assert data["ready"] is True, f"Expected ready=true, got {data}"
        assert data["template_count"] == 5, f"Expected 5 templates, got {data['template_count']}"
        assert data["required"] == 5
    
    def test_verify_ready_returns_false_with_less_than_5_templates(self, client):
        """
        Test that verify-ready returns ready=false when templates < 5.
        """
        # 1. Create user
        user_name = "Test User Less Templates"
        r = client.post("/users", json={"name": user_name})
        assert r.status_code == 200
        user_id = r.json()["id"]
        
        # 2. Upload only 3 templates
        for i in range(3):
            r = client.post(
                f"/users/{user_id}/templates",
                files={"image": ("palm.jpg", _make_jpeg_bytes(), "image/jpeg")},
            )
            assert r.status_code == 200
        
        # 3. Check verify-ready endpoint
        r = client.get(f"/users/{user_id}/verify-ready")
        assert r.status_code == 200
        data = r.json()
        assert data["ready"] is False, f"Expected ready=false, got {data}"
        assert data["template_count"] == 3, f"Expected 3 templates, got {data['template_count']}"
    
    def test_cache_refreshed_after_each_template_upload(self, client):
        """
        Test that cache is properly refreshed after each template upload.
        """
        # 1. Create user
        user_name = "Test User Cache Refresh"
        r = client.post("/users", json={"name": user_name})
        assert r.status_code == 200
        user_id = r.json()["id"]
        
        # 2. Upload templates one by one and check cache
        for i in range(5):
            r = client.post(
                f"/users/{user_id}/templates",
                files={"image": ("palm.jpg", _make_jpeg_bytes(), "image/jpeg")},
            )
            assert r.status_code == 200
            
            # Check cache state
            cache = client.app.state.cache
            user_in_cache = cache.get_user(user_id)
            
            # After first template, user should be in cache
            # After subsequent templates, embeddings count should increase
            if user_in_cache:
                assert len(user_in_cache["embeddings"]) == i + 1, \
                    f"After upload {i+1}, expected {i+1} embeddings in cache, got {len(user_in_cache['embeddings'])}"
