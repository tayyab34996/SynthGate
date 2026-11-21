from api.app import app
from fastapi.testclient import TestClient


def test_health_endpoint() -> None:
    response = TestClient(app).get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_contract_endpoint_lists_both_scenarios() -> None:
    response = TestClient(app).get("/contracts")

    assert response.status_code == 200
    assert set(response.json()["contracts"]) == {
        "delivery_disruption",
        "normal_operations",
    }
