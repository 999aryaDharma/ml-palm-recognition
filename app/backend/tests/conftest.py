import pytest
from fastapi.testclient import TestClient
import sys
import os

# Add backend dir to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from main import app
from db.database import create_tables, engine, Base

@pytest.fixture(scope="session", autouse=True)
def setup_database():
    """Ensure tables are created before running tests."""
    create_tables()
    yield
    # Optional: cleanup after all tests
    # Base.metadata.drop_all(bind=engine)

@pytest.fixture
def client():
    """Provide a TestClient that triggers lifespan events."""
    with TestClient(app) as c:
        yield c
