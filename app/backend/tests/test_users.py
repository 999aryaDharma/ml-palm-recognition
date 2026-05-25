import pytest

def test_create_user(client):
    """Test creating a new user."""
    response = client.post("/users", json={"name": "Unit Test User"})
    assert response.status_code == 200
    data = response.json()
    assert data["name"] == "Unit Test User"
    assert "id" in data
    return data["id"]

def test_list_users(client):
    """Test listing users."""
    # Ensure at least one user exists
    client.post("/users", json={"name": "List Test User"})
    
    response = client.get("/users")
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
    assert len(data) >= 1
    assert any(user["name"] == "List Test User" for user in data)

def test_get_user_not_found(client):
    """Test getting a non-existent user."""
    response = client.get("/users/99999")
    assert response.status_code == 404
    assert response.json()["detail"]["error"] == "user_not_found"

def test_delete_user(client):
    """Test creating and then deleting a user."""
    # 1. Create
    create_res = client.post("/users", json={"name": "Delete Me"})
    user_id = create_res.json()["id"]
    
    # 2. Delete
    delete_res = client.delete(f"/users/{user_id}")
    assert delete_res.status_code == 200
    assert delete_res.json()["deleted"] is True
    
    # 3. Verify gone
    get_res = client.get(f"/users/{user_id}")
    assert get_res.status_code == 404
