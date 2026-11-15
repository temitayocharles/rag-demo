import asyncio
import json

import httpx
import pytest
from fastapi.testclient import TestClient

from rag_demo.api import create_app
from rag_demo.config import Settings
from rag_demo.embedder import OpenAICompatibleEmbedder
from rag_demo.generator import ChatConfig, OpenAICompatibleGenerator
from rag_demo.models import Chunk, SearchHit


def test_provider_switch_only_configuration():
    a = Settings(
        _env_file=None, llm_provider="openai_compatible",
        llm_base_url="https://openrouter.ai/api/v1",
        llm_model="vendor/model-a", llm_api_key="dummy",
        embedding_provider="hash",
    )
    b = Settings(
        _env_file=None, llm_provider="openai_compatible",
        llm_base_url="https://integrate.api.nvidia.com/v1",
        llm_model="vendor/model-b", llm_api_key="dummy",
        embedding_provider="hash",
    )
    assert a.llm_model != b.llm_model
    assert a.llm_base_url != b.llm_base_url


def test_insecure_credential_destination_rejected():
    with pytest.raises(ValueError, match="HTTPS"):
        Settings(_env_file=None, llm_provider="openai_compatible",
                 llm_base_url="http://remote.example/v1", llm_model="some-model")


def test_generator_uses_configured_endpoint_and_key(monkeypatch):
    seen = {}

    def reply(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["authorization"] = request.headers.get("authorization")
        seen["model"] = json.loads(request.content)["model"]
        return httpx.Response(200, json={
            "choices": [{"message": {"content": "Grounded answer [1]."}}]
        })

    real_client = httpx.AsyncClient
    transport = httpx.MockTransport(reply)

    def client_factory(*args, **kwargs):
        return real_client(*args, transport=transport, **kwargs)

    monkeypatch.setattr("rag_demo.generator.httpx.AsyncClient", client_factory)
    config = ChatConfig("https://openrouter.ai/api/v1", "some/model", "dummy",
                        0.0, 128, 5.0)
    hit = SearchHit(score=0.8, chunk=Chunk(
        chunk_id="c", source_id="source", text="Evidence",
        position=0, start=0, end=8,
    ))
    result = asyncio.run(OpenAICompatibleGenerator(config).answer("Question?", [hit]))
    assert result == "Grounded answer [1]."
    assert seen == {
        "url": "https://openrouter.ai/api/v1/chat/completions",
        "authorization": "Bearer dummy", "model": "some/model",
    }


def test_embedding_dimensions_are_optional(monkeypatch):
    seen = {}

    def reply(request: httpx.Request) -> httpx.Response:
        seen.update(json.loads(request.content))
        return httpx.Response(200, json={
            "data": [{"index": 0, "embedding": [1.0, 0.0, 0.0, 0.0]}]
        })

    real_client = httpx.AsyncClient
    monkeypatch.setattr(
        "rag_demo.embedder.httpx.AsyncClient",
        lambda *args, **kwargs: real_client(
            *args, transport=httpx.MockTransport(reply), **kwargs
        ),
    )
    embedder = OpenAICompatibleEmbedder(
        None, "embedding-model", 4, "http://127.0.0.1:8001/v1", False
    )
    assert asyncio.run(embedder.embed(["test"])) == [[1.0, 0.0, 0.0, 0.0]]
    assert "dimensions" not in seen


def test_ask_unconfigured_has_clear_error():
    with TestClient(create_app(Settings(_env_file=None, store_backend="memory"))) as client:
        result = client.post("/v1/ask", json={"query": "hello"})
        assert result.status_code == 503
