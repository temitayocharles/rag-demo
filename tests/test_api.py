from fastapi.testclient import TestClient

from rag_demo.api import create_app
from rag_demo.config import Settings


def test_health_ingest_search():
    app = create_app(Settings(chunk_size=50, chunk_overlap=10, store_backend="memory"))
    with TestClient(app) as client:
        assert client.get("/healthz").json() == {"status": "ok"}
        response = client.post("/v1/documents", json={
            "source_id": "test", "text": "hello vector search"
        })
        assert response.status_code == 200, response.text
        response = client.post("/v1/search", json={"query": "hello", "top_k": 3})
        assert response.status_code == 200, response.text
        assert response.json()["hits"][0]["chunk"]["source_id"] == "test"


def test_validation():
    with TestClient(create_app(Settings(store_backend="memory"))) as client:
        assert client.post("/v1/documents", json={"source_id": "", "text": "x"}).status_code == 422
        assert client.post("/v1/search", json={"query": "", "top_k": 0}).status_code == 422


def test_demo_llm_is_opt_in_and_never_enables_document_uploads():
    import pytest

    with pytest.raises(ValueError, match="RAG_LLM_API_KEY"):
        Settings(
            _env_file=None, demo_mode=True, store_backend="memory",
            llm_provider="openai_compatible", llm_model="test-chat",
            llm_base_url="https://example.com/v1",
            llm_api_key=None,
        )

    settings = Settings(
        _env_file=None, demo_mode=True, store_backend="memory",
        llm_provider="openai_compatible", llm_model="test-chat",
        llm_base_url="https://example.com/v1",
        llm_api_key="dummy-for-local-test-only",
    )
    with TestClient(create_app(settings)) as client:
        assert client.get("/healthz").status_code == 200
        assert client.post("/v1/documents", json={
            "source_id": "test", "text": "No public uploads permitted."
        }).status_code == 403
        assert client.get("/v1/knowledge-bases").status_code == 200


def test_demo_default_still_disables_llm_generation():
    settings = Settings(_env_file=None, demo_mode=True, store_backend="memory")
    with TestClient(create_app(settings)) as client:
        assert client.post("/v1/ask", json={
            "query": "What does Tayoca do?", "knowledge_base": "tayoca"
        }).status_code == 503
