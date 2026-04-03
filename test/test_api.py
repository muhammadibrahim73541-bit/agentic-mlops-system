from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)

def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"

def test_predict():
    payload = {
        "temp": 20.0, "humidity": 50.0, "pressure": 1013.0,
        "wind_speed": 5.0, "hour": 12, "is_weekend": 0
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    assert "predicted_demand_mw" in response.json()