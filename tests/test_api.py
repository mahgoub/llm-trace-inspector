from fastapi.testclient import TestClient

from llm_trace_inspector.api import app
from llm_trace_inspector.io import load_trace


def test_health() -> None:
    client = TestClient(app)
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_eval_endpoint() -> None:
    client = TestClient(app)
    trace = load_trace("examples/rag_trace_good.json")
    response = client.post("/eval", json=trace.model_dump())
    assert response.status_code == 200
    assert response.json()["trace_id"] == "rag-good-001"

