"""Authenticated Streamable HTTP MCP tools backed by the existing RAG service.

This demo is single-organization. Read tools work with the bundled public KB.
Writes require local durable SQLite and a separate write bearer token; never Vercel.
"""
import hashlib
import hmac
import json
from pathlib import Path
from typing import Any

import httpx
from mcp.server.mcpserver import MCPServer
from mcp.server.transport_security import TransportSecuritySettings
from starlette.responses import JSONResponse

from rag_demo.config import Settings
from rag_demo.generator import OpenAICompatibleGenerator
from rag_demo.knowledge import KnowledgeBases
from rag_demo.mcp_store import McpDataStore
from rag_demo.models import DocumentIn
from rag_demo.service import RagService

WRITE_TOOLS = frozenset({
    "ingest_document", "create_business_record", "trigger_workflow", "save_memory"
})
BUNDLED_SOURCE = "tayoca:public-knowledge"
BUNDLED_PATH = (
    Path(__file__).resolve().parents[2] / "knowledge" / "tayoca_public_demo_knowledge.md"
)


def _require_kb(knowledge: KnowledgeBases, settings: Settings, name: str) -> str:
    kb = knowledge.require(name)
    if settings.demo_mode and kb != "tayoca":
        raise ValueError("Public demo only exposes the tayoca knowledge base")
    return kb


def create_mcp_gateway(
    settings: Settings, service: RagService, knowledge: KnowledgeBases,
    generator: OpenAICompatibleGenerator | None,
) -> tuple[MCPServer, Any]:
    """Return the MCP server and an authenticated ASGI app to mount at /."""
    durable = not settings.demo_mode and settings.store_backend == "sqlite"
    registry = McpDataStore(settings.sqlite_path) if durable else None
    mcp = MCPServer(
        name="rag-demo",
        description="Scoped knowledge retrieval and approved business operations",
        version="0.4.0",
    )

    def require_write() -> McpDataStore:
        if registry is None:
            raise ValueError("Writes unavailable: a durable SQLite host is required")
        return registry

    @mcp.tool(description="List configured knowledge bases. The public demo only exposes Tayoca.")
    def list_knowledge_bases() -> dict:
        if settings.demo_mode:
            return {"knowledge_bases": [
                item for item in knowledge.summary() if item["name"] == "tayoca"
            ]}
        return {"knowledge_bases": knowledge.summary()}

    @mcp.tool(description="Search an authorized knowledge base, returning scored source chunks.")
    async def search_knowledge(
        query: str, knowledge_base: str = "tayoca", top_k: int = 5,
    ) -> dict:
        if not 1 <= len(query) <= 4000 or not 1 <= top_k <= 20:
            raise ValueError("Invalid query or top_k")
        kb = _require_kb(knowledge, settings, knowledge_base)
        hits = await service.search(query, top_k, kb)
        return {"knowledge_base": kb, "hits": [hit.model_dump() for hit in hits]}

    @mcp.tool(description="Answer from retrieved evidence, only if an LLM is configured.")
    async def ask_knowledge(
        query: str, knowledge_base: str = "tayoca", top_k: int = 5,
    ) -> dict:
        if not 1 <= len(query) <= 4000 or not 1 <= top_k <= 20:
            raise ValueError("Invalid query or top_k")
        kb = _require_kb(knowledge, settings, knowledge_base)
        if generator is None:
            return {"enabled": False, "error": "LLM not configured",
                    "knowledge_base": kb}
        hits = await service.search(query, top_k, kb)
        answer = await generator.answer(query, hits)
        return {"enabled": True, "knowledge_base": kb, "answer": answer,
                "sources": [{"source_id": h.chunk.source_id, "chunk_id": h.chunk.chunk_id}
                            for h in hits]}

    @mcp.tool(description="Retrieve a known document by source ID and knowledge-base scope.")
    def get_document(source_id: str, knowledge_base: str = "tayoca") -> dict:
        kb = _require_kb(knowledge, settings, knowledge_base)
        if source_id == BUNDLED_SOURCE and kb == "tayoca":
            return {"found": True, "source_id": source_id, "knowledge_base": kb,
                    "text": BUNDLED_PATH.read_text(encoding="utf-8"),
                    "metadata": {"origin": "bundled-public-document", "safe_to_share": True}}
        record = registry.get_document(kb, source_id) if registry else None
        return {"found": bool(record), "document": record}

    @mcp.tool(description="Return verified provenance for a known source.")
    def get_source_metadata(source_id: str, knowledge_base: str = "tayoca") -> dict:
        kb = _require_kb(knowledge, settings, knowledge_base)
        if kb == "tayoca" and source_id == BUNDLED_SOURCE:
            return {"found": True, "source_id": source_id, "knowledge_base": kb,
                    "origin": "bundled-public-document",
                    "source_url": "https://tayoca.com/", "safe_to_share": True}
        record = registry.get_document(kb, source_id) if registry else None
        if record:
            return {"found": True, "source_id": source_id, "knowledge_base": kb,
                    "metadata": record["metadata"], "updated_at": record["updated_at"]}
        return {"found": False, "source_id": source_id, "knowledge_base": kb}

    @mcp.tool(description="Persist and index an authorized text document (write permission).")
    async def ingest_document(
        source_id: str, text: str, knowledge_base: str = "local",
    ) -> dict:
        db = require_write()
        kb = _require_kb(knowledge, settings, knowledge_base)
        if not 1 <= len(source_id) <= 256 or source_id == BUNDLED_SOURCE:
            raise ValueError("Invalid or reserved source_id")
        if not text.strip() or len(text.encode("utf-8")) > settings.max_document_bytes:
            raise ValueError("Document is empty or exceeds configured byte limit")
        document = DocumentIn(source_id=source_id, text=text)
        result = await service.ingest(document, kb)
        saved = db.save_document(kb, document)
        return {**saved, "chunks_upserted": result.chunks_upserted}

    @mcp.tool(description="Create an immutable business record; no arbitrary SQL (write permission).")
    def create_business_record(kind: str, payload: dict[str, Any]) -> dict:
        db = require_write()
        if not 1 <= len(kind) <= 64 or not kind.replace("_", "").isalnum():
            raise ValueError("kind must be alphanumeric or underscores")
        encoded = json.dumps(payload, allow_nan=False)
        if len(encoded.encode("utf-8")) > 16384:
            raise ValueError("Record exceeds 16 KiB")
        return db.create_record(kind, payload)

    @mcp.tool(description="Record an unverified memory for later review (write permission).")
    def save_memory(subject: str, content: str, source: str) -> dict:
        db = require_write()
        if not 1 <= len(subject) <= 128 or not 1 <= len(source) <= 256:
            raise ValueError("Invalid subject or provenance")
        if not 1 <= len(content) <= 4000:
            raise ValueError("Memory must be between 1 and 4000 characters")
        return db.save_memory(subject, content, source)

    @mcp.tool(description="Send an audited event to the configured n8n webhook (write permission).")
    async def trigger_workflow(
        workflow: str, payload: dict[str, Any],
    ) -> dict:
        db = require_write()
        if not settings.mcp_workflow_url or not settings.mcp_workflow_secret:
            raise ValueError("Approved workflow endpoint and signing secret not configured")
        if workflow != settings.mcp_workflow_name:
            raise ValueError("Workflow is not allowlisted")
        body = json.dumps(payload, allow_nan=False).encode()
        if len(body) > 16384:
            raise ValueError("Workflow payload exceeds 16 KiB")
        event_id = db.queue_workflow(workflow, payload)
        request_body = json.dumps({
            "event_id": event_id, "workflow": workflow, "payload": payload,
        }, separators=(",", ":"), allow_nan=False).encode()
        signature = hmac.new(
            settings.mcp_workflow_secret.encode(), request_body, hashlib.sha256
        ).hexdigest()
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                response = await client.post(
                    settings.mcp_workflow_url,
                    content=request_body,
                    headers={"Content-Type": "application/json",
                             "X-RAG-Signature": "sha256=" + signature,
                             "X-RAG-Event-ID": event_id},
                    follow_redirects=False,
                )
                response.raise_for_status()
        except httpx.HTTPError:
            db.workflow_status(event_id, "delivery_failed")
            return {"event_id": event_id, "status": "delivery_failed",
                    "note": "Inspect the audited event; safe retry is not automatic"}
        db.workflow_status(event_id, "delivered")
        return {"event_id": event_id, "status": "delivered",
                "note": "Destination accepted event; workflow completion is not confirmed"}

    mcp_app = mcp.streamable_http_app(
        streamable_http_path="/mcp", stateless_http=True, json_response=True,
        max_request_body_size=131072,
        transport_security=TransportSecuritySettings(
            allowed_hosts=[
                "localhost:*", "127.0.0.1:*", "testserver", "rag-demo-ruddy.vercel.app",
                "*.vercel.app",
            ],
        ),
    )
    return mcp, McpAuthGateway(mcp_app, settings, durable)


class McpAuthGateway:
    """Bounded, fail-closed bearer auth; read-only tokens cannot call write tools."""

    def __init__(self, app: Any, settings: Settings, durable: bool) -> None:
        self.app = app
        self.settings = settings
        self.durable = durable

    async def __call__(self, scope: dict, receive: Any, send: Any) -> None:
        if scope["type"] != "http":
            await JSONResponse({"error": "Unsupported"}, status_code=400)(
                scope, receive, send
            )
            return
        if scope.get("path") != "/mcp":
            await JSONResponse({"error": "Not found"}, status_code=404)(
                scope, receive, send
            )
            return
        if not (self.settings.mcp_read_token or self.settings.mcp_write_token):
            await JSONResponse({"error": "MCP not configured"}, status_code=503)(
                scope, receive, send
            )
            return
        headers = {k.lower(): v for k, v in scope.get("headers", [])}
        header = headers.get(b"authorization", b"").decode("latin-1")
        token = header[7:] if header.startswith("Bearer ") else ""
        role = None
        for label, secret in [
            ("write", self.settings.mcp_write_token),
            ("read", self.settings.mcp_read_token),
        ]:
            if secret and hmac.compare_digest(token, secret):
                role = label
        if role is None:
            await JSONResponse({"error": "Unauthorized"}, status_code=401,
                               headers={"WWW-Authenticate": "Bearer"})(
                scope, receive, send
            )
            return
        if scope.get("method") != "POST":
            await self.app(scope, receive, send)
            return
        fragments = []
        size = 0
        more = True
        while more:
            message = await receive()
            if message["type"] == "http.disconnect":
                return
            part = message.get("body", b"")
            size += len(part)
            if size > 131072:
                await JSONResponse({"error": "Request too large"}, status_code=413)(
                    scope, receive, send
                )
                return
            fragments.append(part)
            more = message.get("more_body", False)
        body = b"".join(fragments)
        try:
            envelope = json.loads(body)
        except (ValueError, UnicodeDecodeError):
            await JSONResponse({"error": "Invalid JSON"}, status_code=400)(
                scope, receive, send
            )
            return
        messages = envelope if isinstance(envelope, list) else [envelope]
        for message in messages:
            if not isinstance(message, dict):
                continue
            if message.get("method") == "tools/call":
                params = message.get("params") or {}
                name = params.get("name") if isinstance(params, dict) else None
                if name in WRITE_TOOLS and (role != "write" or not self.durable):
                    await JSONResponse({"error": "Writes require authorized durable host"},
                                       status_code=403)(scope, receive, send)
                    return
        received = False

        async def replay() -> dict:
            nonlocal received
            if not received:
                received = True
                return {"type": "http.request", "body": body, "more_body": False}
            return await receive()

        await self.app(scope, replay, send)
