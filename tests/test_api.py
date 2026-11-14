from fastapi.testclient import TestClient

from rag_demo.api import create_app
from rag_demo.config import Settings


def test_health_ingest_search():
    app = create_app(Settings(chunk_size=50, chunk_overlap=10))
    with TestClient(app) as client:
        assert client.get("/healthz").json() == {"status": "ok"}
        response = client.post("/v1/documents", json={"source_id": "test", "text": "hello vector search"})
        assert response.status_code == 200, response.text
        response = client.post("/v1/search", json={"query": "hello", "top_k": 3})
        assert response.status_code == 200, response.text
        assert response.json()["hits"][0]["chunk"]["source_id"] == "test"


def test_validation():
    with TestClient(create_app()) as client:
        assert client.post("/v1/documents", json={"source_id": "", "text": "x"}).status_code == 422
        assert client.post("/v1/search", json={"query": "", "top_k": 0}).status_code == 422
