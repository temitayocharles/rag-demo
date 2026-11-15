"""MCP protocol, authorization, business persistence and safety tests."""
import json

import pytest
from fastapi.testclient import TestClient

from rag_demo.api import create_app
from rag_demo.config import Settings
from rag_demo.mcp_store import McpDataStore

READ_TOKEN = "r" * 48
WRITE_TOKEN = "w" * 48


def demo_settings(**overrides):
    return Settings(
        _env_file=None, demo_mode=True, store_backend="memory",
        mcp_read_token=READ_TOKEN, mcp_write_token=WRITE_TOKEN,
        **overrides,
    )


def rpc(name, arguments=None):
    return {
        "jsonrpc": "2.0", "id": 1, "method": "tools/call",
        "params": {"name": name, "arguments": arguments or {}},
    }


def post(client, data, token=READ_TOKEN):
    return client.post(
        "/mcp", json=data,
        headers={"Authorization": f"Bearer {token}",
                 "Accept": "application/json, text/event-stream",
                 "Content-Type": "application/json",
                 "MCP-Protocol-Version": "2025-11-25"},
    )


def test_mcp_disabled_without_credentials():
    settings = Settings(_env_file=None, store_backend="memory", demo_mode=True)
    with TestClient(create_app(settings)) as client:
        response = client.post("/mcp", json={"jsonrpc": "2.0", "id": 1,
                                             "method": "tools/list"})
        assert response.status_code == 503


def test_mcp_token_validation():
    with pytest.raises(ValueError, match="at least 32"):
        Settings(_env_file=None, mcp_read_token="short")
    with pytest.raises(ValueError, match="must differ"):
        Settings(_env_file=None, mcp_read_token=READ_TOKEN,
                 mcp_write_token=READ_TOKEN)


def test_mcp_requires_bearer_and_rejects_demo_writes():
    with TestClient(create_app(demo_settings())) as client:
        assert client.get("/healthz").status_code == 200
        bad = client.post("/mcp", json={"jsonrpc": "2.0",
                                        "id": 1, "method": "tools/list"})
        assert bad.status_code == 401
        assert post(client, rpc("save_memory", {
            "subject": "u", "content": "test", "source": "s",
        })).status_code == 403
        assert post(client, rpc("save_memory", {
            "subject": "u", "content": "test", "source": "s",
        }), WRITE_TOKEN).status_code == 403


def test_mcp_lists_tools_and_returns_public_knowledge():
    with TestClient(create_app(demo_settings())) as client:
        resp = post(client, {"jsonrpc": "2.0", "id": 1, "method": "initialize",
                             "params": {"protocolVersion": "2025-11-25",
                                        "capabilities": {},
                                        "clientInfo": {"name": "test", "version": "1"}}})
        assert resp.status_code == 200, resp.text[:500]
        listing = post(client, {"jsonrpc": "2.0", "id": 2, "method": "tools/list"})
        assert listing.status_code == 200, listing.text[:500]
        names = {tool["name"] for tool in listing.json()["result"]["tools"]}
        assert names == {
            "list_knowledge_bases", "search_knowledge", "ask_knowledge",
            "get_document", "get_source_metadata",
            "ingest_document", "create_business_record",
            "trigger_workflow", "save_memory",
        }
        result = post(client, rpc("search_knowledge", {
            "query": "What does Tayoca do?", "knowledge_base": "tayoca",
        }))
        assert result.status_code == 200, result.text[:500]
        assert "tayoca:public-knowledge" in result.text
        assert "mcp_read_token" not in result.text
        document = post(client, rpc("get_source_metadata", {
            "source_id": "tayoca:public-knowledge",
        }))
        assert document.status_code == 200
        assert "bundled-public-document" in document.text


def test_local_records_and_memory_persist_across_connections(tmp_path):
    db_file = str(tmp_path / "mcp.sqlite3")
    db = McpDataStore(db_file)
    record = db.create_record("lead", {"email": "example@example.com"})
    memory = db.save_memory("customer-1", "Prefers email", "customer form")
    assert record["record_id"]
    assert memory["verified"] is False
    reopened = McpDataStore(db_file)
    found = reopened.db.execute(
        "SELECT payload FROM mcp_records WHERE record_id=?",
        (record["record_id"],),
    ).fetchone()
    assert json.loads(found[0])["email"] == "example@example.com"
    found_memory = reopened.db.execute(
        "SELECT verified FROM mcp_memories WHERE memory_id=?",
        (memory["memory_id"],),
    ).fetchone()
    assert found_memory == (0,)
    assert reopened.db.execute("SELECT COUNT(*) FROM mcp_audit").fetchone()[0] == 2


def test_write_requires_durable_sqlite_even_with_write_token():
    with TestClient(create_app(demo_settings())) as client:
        for name in [
            "ingest_document", "create_business_record", "trigger_workflow", "save_memory"
        ]:
            response = post(client, rpc(name), WRITE_TOKEN)
            assert response.status_code == 403


def test_local_write_requires_write_token(tmp_path):
    settings = Settings(
        _env_file=None, demo_mode=False, store_backend="sqlite",
        sqlite_path=str(tmp_path / "local.sqlite3"),
        mcp_read_token=READ_TOKEN, mcp_write_token=WRITE_TOKEN,
    )
    with TestClient(create_app(settings)) as client:
        assert post(client, rpc("create_business_record", {
            "kind": "lead", "payload": {"customer": "test"}
        }), READ_TOKEN).status_code == 403
        result = post(client, rpc("create_business_record", {
            "kind": "lead", "payload": {"customer": "test"}
        }), WRITE_TOKEN)
        assert result.status_code == 200, result.text[:500]
        assert "record_id" in result.text
