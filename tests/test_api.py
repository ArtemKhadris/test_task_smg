"""
API tests for Madrid Housing Price Prediction FastAPI application.

This module contains unit tests for the FastAPI endpoints to ensure
proper functionality and response formats.
"""

from fastapi.testclient import TestClient
from src.app.main import app

# Initialize TestClient for FastAPI application testing
client = TestClient(app)


def test_health():
    """
    Test health check endpoint.

    Verifies:
    - Endpoint returns HTTP 200 status code
    - Response contains status field with value "ok"
    """
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_model_info():
    """
    Test model information endpoint.

    Verifies:
    - Endpoint returns HTTP 200 status code
    - Response contains version information
    """
    r = client.get("/model/info")
    assert r.status_code == 200
    assert "version" in r.json()


def test_predict():
    """
    Test single prediction endpoint with valid input.

    Verifies:
    - Endpoint returns HTTP 200 status code for valid input
    - Response contains prediction field with numeric value
    """
    # Valid input payload for housing price prediction
    payload = {
        "sq_mt_built": 120,  # Square meters built
        "sq_mt_useful": 100,  # Square meters useful
        "n_rooms": 3,  # Number of rooms
        "n_bathrooms": 2,  # Number of bathrooms
        "has_parking": True,  # Parking availability
    }

    r = client.post("/predict", json=payload)
    assert r.status_code == 200
    assert "prediction" in r.json()


def test_batch_predict():
    """
    Test batch prediction endpoint with multiple inputs.

    Verifies:
    - Endpoint returns HTTP 200 status code for valid batch input
    - Response contains predictions field as a list
    - List contains appropriate number of predictions
    """
    # Batch input with two housing instances for prediction
    payload = {
        "inputs": [
            {
                "sq_mt_built": 120,
                "sq_mt_useful": 100,
                "n_rooms": 3,
                "n_bathrooms": 2,
                "has_parking": True,
            },
            {
                "sq_mt_built": 80,
                "sq_mt_useful": 70,
                "n_rooms": 2,
                "n_bathrooms": 1,
                "has_parking": False,
            },
        ]
    }

    r = client.post("/batch_predict", json=payload)
    assert r.status_code == 200
    assert "predictions" in r.json()
    assert isinstance(r.json()["predictions"], list)
