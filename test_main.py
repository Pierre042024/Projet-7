from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

def test_read_root():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"status": "API is running"}

def test_model_scores():
    response = client.get("/model_scores")
    assert response.status_code == 200
    assert "auc" in response.json()
    assert "accuracy" in response.json()

def test_get_client_ids():
    response = client.get("/clients")
    assert response.status_code == 200
    assert isinstance(response.json(), list)