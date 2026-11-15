from fastapi.testclient import TestClient

from rag_demo.api import create_app
from rag_demo.config import Settings


def test_public_demo_is_read_only_and_seeded():
    settings = Settings(
        _env_file=None, demo_mode=True, store_backend="memory",
        llm_provider="disabled", embedding_provider="hash", default_knowledge_base="tayoca"
    )
    with TestClient(create_app(settings)) as client:
        assert client.get("/").status_code == 200
        assert client.get("/healthz").json() == {"status": "ok"}
        hits = client.post("/v1/search", json={
            "knowledge_base": "tayoca", "query": "retrieval evidence"
        })
        assert hits.status_code == 200
        assert hits.json()["hits"][0]["chunk"]["source_id"] == "tayoca:public-knowledge"
        assert client.post("/v1/documents", json={
            "source_id": "attacker", "text": "untrusted"
        }).status_code == 403
        assert client.post("/v1/ask", json={"query": "test"}).status_code == 503
