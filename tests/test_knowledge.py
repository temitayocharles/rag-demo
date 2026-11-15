import json

import pytest
from fastapi.testclient import TestClient

from rag_demo.api import create_app
from rag_demo.config import Settings
from rag_demo.loaders import as_document, extract_text


def test_env_example_and_knowledge_config(tmp_path, monkeypatch):
    config = tmp_path / "knowledge.json"
    config.write_text(json.dumps({"knowledge_bases": {
        "first": {"sources": []}, "second": {"sources": []}
    }}))
    monkeypatch.setenv("RAG_KNOWLEDGE_BASES_FILE", str(config))
    monkeypatch.setenv("RAG_DEFAULT_KNOWLEDGE_BASE", "first")
    settings = Settings(_env_file=None, store_backend="sqlite",
                        sqlite_path=str(tmp_path / "db.sqlite3"))
    assert settings.default_knowledge_base == "first"
    with TestClient(create_app(settings)) as client:
        response = client.get("/v1/knowledge-bases")
        assert {x["name"] for x in response.json()["knowledge_bases"]} == {"first", "second"}
        for kb in ["first", "second"]:
            res = client.post("/v1/documents", json={
                "source_id": "shared", "text": f"{kb} specialword", "knowledge_base": kb
            })
            assert res.status_code == 200, res.text
        for kb in ["first", "second"]:
            res = client.post("/v1/search", json={"query": "specialword", "knowledge_base": kb})
            assert res.status_code == 200
            assert len(res.json()["hits"]) == 1
            assert res.json()["hits"][0]["chunk"]["metadata"]["knowledge_base"] == kb
        assert client.post("/v1/search", json={
            "query": "test", "knowledge_base": "unknown"
        }).status_code == 422

    # Prove records survive app restart.
    with TestClient(create_app(settings)) as client:
        res = client.post("/v1/search", json={"query": "specialword", "knowledge_base": "first"})
        assert len(res.json()["hits"]) == 1


def test_file_loader_limits_and_utf8(tmp_path):
    settings = Settings(_env_file=None)
    doc = as_document("x", "hello.md", b"\xef\xbb\xbfHello, world", settings)
    assert doc.text == "Hello, world"
    with pytest.raises(ValueError, match="unsupported"):
        extract_text("archive.zip", b"garbage", settings)
    with pytest.raises(ValueError, match="too large"):
        extract_text("big.txt", b"x" * (settings.max_source_bytes + 1), settings)
