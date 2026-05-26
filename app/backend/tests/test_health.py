def test_health_endpoint(client):
    """Test that the health check endpoint returns 200 and correct status."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    # Models should be initialized by lifespan
    assert data["model_loaded"] is not None
    assert data["detector_loaded"] is not None
    assert data["cached_users"] >= 0
