import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient

from doccite.api import create_app


def test_api_answer_health_and_validation(index, monkeypatch):
    monkeypatch.setenv("DOCCITE_GENERATOR", "extractive")
    with TestClient(create_app(index)) as client:
        assert client.get("/health").json()["index"]["document_count"] == 12
        response = client.post(
            "/ask", json={"question": "What is the default timeout for network inactivity?"}
        )
        assert response.status_code == 200
        assert response.json()["status"] == "answered"
        assert response.json()["citations"]
        assert client.post("/ask", json={"question": " "}).status_code == 422
        assert client.post("/ask", json={"question": "timeout", "k": 100}).status_code == 422
        assert client.get("/openapi.json").status_code == 200


def test_busy_api_returns_429(index, monkeypatch):
    monkeypatch.setenv("DOCCITE_GENERATOR", "extractive")
    app = create_app(index)
    with TestClient(app) as client:
        app.state.lock.acquire()
        try:
            assert client.post("/ask", json={"question": "timeout"}).status_code == 429
        finally:
            app.state.lock.release()
