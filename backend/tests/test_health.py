"""Tests for the health-check endpoint."""

from fastapi.testclient import TestClient


def test_health_check_returns_ok(client: TestClient) -> None:
    response = client.get("/api/v1/health")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["database"] == "connected"
    assert body["service"] == "PhishGuard AI Backend"
    assert body["environment"] == "test"
    assert body["version"]
