def test_health_endpoint(client):
    """Test that the health check endpoint returns 200 and correct status."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    # Models should be None initially as we haven't integrated ML yet in main.py lifespan for this phase
    assert data["model_loaded"] is False
    assert data["detector_loaded"] is False
