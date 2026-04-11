import pytest
from fastapi.testclient import TestClient
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from api.main import app

client = TestClient(app)

def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert len(data["supported_cities"]) == 4

def test_cities_endpoint():
    response = client.get("/cities")
    assert response.status_code == 200
    data = response.json()
    cities = [c["name"] for c in data["cities"]]
    assert "Copenhagen" in cities
    assert "Aarhus" in cities

def test_prediction_validation():
    response = client.post("/predict", json={
        "temp": 15,
        "humidity": 60,
        "pressure": 1013,
        "wind_speed": 5,
        "hour": 14,
        "is_weekend": 0,
        "is_holiday": 0,
        "month": 4,
        "city": "Copenhagen"
    })
    assert response.status_code in [200, 503]  # 503 if model not loaded

def test_invalid_input():
    response = client.post("/predict", json={
        "temp": 15,
        "humidity": 150,  # Invalid >100
        "pressure": 1013,
        "wind_speed": 5,
        "hour": 14,
        "month": 4
    })
    assert response.status_code == 422