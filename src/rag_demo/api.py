from contextlib import asynccontextmanager
import logging
import os
from pathlib import Path

import httpx
from fastapi import FastAPI, HTTPException, Request

from rag_demo.chunker import Chunker
from rag_demo.config import Settings
from rag_demo.embedder import HashEmbedder, OpenAICompatibleEmbedder
from rag_demo.generator import ChatConfig, OpenAICompatibleGenerator
from rag_demo.ingestor import Ingestor
from rag_demo.knowledge import KnowledgeBases
from rag_demo.models import (
    AskRequest,
    AskResponse,
    Citation,
    DocumentIn,
    IngestResult,
    SearchRequest,
    SearchResponse,
)
from rag_demo.service import RagService
from rag_demo.sqlite_store import SQLiteVectorStore
from rag_demo.store import InMemoryVectorStore
from rag_demo.telegram import (\n    process_update,\n    register_cloud_webhook,\n    telegram_ready,\n    verify_webhook,\n)


logger = logging.getLogger(__name__)\n\n\ndef create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or Settings()
    if settings.demo_mode and settings.default_knowledge_base == "local":
        settings = settings.model_copy(update={"default_knowledge_base": "tayoca"})
    knowledge = KnowledgeBases(settings.knowledge_bases_file)
    knowledge.require(settings.default_knowledge_base)
    if settings.embedding_provider == "hash":
        embedder = HashEmbedder(settings.embedding_dimensions)
    else:
        legacy = settings.embedding_provider == "openai"
        embedder = OpenAICompatibleEmbedder(
            settings.embedding_api_key or (settings.openai_api_key if legacy else None),
            settings.openai_embedding_model if legacy else settings.embedding_model,
            settings.embedding_dimensions,
            settings.openai_base_url if legacy else settings.embedding_base_url,
            settings.embedding_send_dimensions,
        )
    store = (SQLiteVectorStore(settings.embedding_dimensions, settings.sqlite_path)
             if settings.store_backend == "sqlite"
             else InMemoryVectorStore(settings.embedding_dimensions))
    service = RagService(
        Ingestor(settings.max_document_bytes),
        Chunker(settings.chunk_size, settings.chunk_overlap),
        embedder, store,
    )
    generator = (
        OpenAICompatibleGenerator(ChatConfig(
            base_url=settings.llm_base_url, model=settings.llm_model,
            api_key=settings.llm_api_key, temperature=settings.llm_temperature,
            max_tokens=settings.llm_max_tokens, timeout_seconds=settings.llm_timeout_seconds,
        ))
        if settings.llm_provider == "openai_compatible" else None
    )

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.rag_service = service
        if settings.demo_mode:
            package = (
                Path(__file__).resolve().parents[2]
                / "knowledge"
                / "tayoca_public_demo_knowledge.md"
            )
            await service.ingest(
                DocumentIn(
                    source_id="tayoca:public-knowledge",
                    text=package.read_text(encoding="utf-8"),
                    metadata={
                        "origin": "bundled-public-document",
                        "filename": package.name,
                        "source_url": "https://tayoca.com/",
                        "safe_to_share": True,
                    },
                ),
                "tayoca",
            )
        if (
            settings.telegram_auto_register
            and telegram_ready(settings)
            and os.environ.get("VERCEL_ENV") == "production"
        ):
            try:
                await register_cloud_webhook(settings)
            except (httpx.HTTPError, ValueError) as exc:
                # Never log exception details: provider URLs contain the bot token.
                logger.warning(
                    "Telegram webhook registration failed (%s)", type(exc).__name__
                )
        yield

    app = FastAPI(title="RAG Demo API", version="0.3.0", lifespan=lifespan)

    @app.get("/")
    async def root() -> dict[str, str]:
        return {
            "name": "RAG Demo API",
            "mode": "public-demo" if settings.demo_mode else "local",
            "docs": "/docs",
            "health": "/healthz",
            "knowledge_bases": "/v1/knowledge-bases",
            "search": "/v1/search",
            "telegram": "ready" if telegram_ready(settings) else "unconfigured",
        }

    @app.get("/healthz")
    async def healthz() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/v1/knowledge-bases")
    async def knowledge_bases() -> dict:
        return {"knowledge_bases": knowledge.summary()}

    @app.post("/v1/documents", response_model=IngestResult)
    async def ingest(document: DocumentIn, request: Request) -> IngestResult:
        if settings.demo_mode:
            raise HTTPException(status_code=403, detail="Uploads disabled in public demo")
        try:
            kb = knowledge.require(document.knowledge_base or settings.default_knowledge_base)
            return await request.app.state.rag_service.ingest(document, kb)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.post("/v1/search", response_model=SearchResponse)
    async def search(query: SearchRequest, request: Request) -> SearchResponse:
        try:
            kb = knowledge.require(query.knowledge_base or settings.default_knowledge_base)
            return SearchResponse(
                hits=await request.app.state.rag_service.search(query.query, query.top_k, kb)
            )
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.post("/v1/ask", response_model=AskResponse)
    async def ask(query: AskRequest, request: Request) -> AskResponse:
        if generator is None:
            raise HTTPException(status_code=503, detail="LLM generation is not configured")
        try:
            kb = knowledge.require(query.knowledge_base or settings.default_knowledge_base)
            hits = await request.app.state.rag_service.search(query.query, query.top_k, kb)
            answer = await generator.answer(query.query, hits)
            return AskResponse(
                answer=answer, knowledge_base=kb, model=settings.llm_model,
                citations=[
                    Citation(number=i, source_id=hit.chunk.source_id,
                             chunk_id=hit.chunk.chunk_id, metadata=hit.chunk.metadata)
                    for i, hit in enumerate(hits, start=1)
                ],
            )
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except (httpx.HTTPError, KeyError, IndexError, TypeError) as exc:
            # Never return upstream response bodies or credentials to API consumers.
            raise HTTPException(
                status_code=502, detail="LLM or embedding provider failure"
            ) from exc

    @app.post("/integrations/telegram/webhook", include_in_schema=False)
    async def telegram_webhook(request: Request) -> dict[str, bool]:
        if not telegram_ready(settings):
            raise HTTPException(status_code=503, detail="Telegram not configured")
        if not verify_webhook(
            settings.telegram_bot_token,
            request.headers.get("X-Telegram-Bot-Api-Secret-Token"),
        ):
            raise HTTPException(status_code=403, detail="Invalid Telegram webhook secret")
        if int(request.headers.get("content-length", "0") or "0") > 32768:
            raise HTTPException(status_code=413, detail="Telegram update too large")
        try:
            payload = await request.json()
        except ValueError as exc:
            raise HTTPException(status_code=400, detail="Invalid JSON") from exc
        await process_update(payload, settings, request.app.state.rag_service, generator)
        return {"ok": True}

    return app


app = create_app()
