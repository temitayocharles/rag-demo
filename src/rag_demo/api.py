import logging
import os
from contextlib import AsyncExitStack, asynccontextmanager
from pathlib import Path

import httpx
from fastapi import BackgroundTasks, FastAPI, HTTPException, Request

from rag_demo.chunker import Chunker
from rag_demo.config import Settings
from rag_demo.embedder import HashEmbedder, OpenAICompatibleEmbedder
from rag_demo.generator import ChatConfig, OpenAICompatibleGenerator
from rag_demo.ingestor import Ingestor
from rag_demo.knowledge import KnowledgeBases
from rag_demo.mcp_gateway import create_mcp_gateway
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
from rag_demo.telegram import (
    process_update,
    register_cloud_webhook,
    telegram_ready,
    verify_webhook,
)

logger = logging.getLogger(__name__)


def create_app(settings: Settings | None = None) -> FastAPI:
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

    mcp_server, mcp_asgi = create_mcp_gateway(settings, service, knowledge, generator)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.rag_service = service
        if settings.demo_mode:
            package = (
                Path(__file__).resolve().parents[2]
                / "knowledge"
                / "tayoca_public_demo_knowledge.md"
            )
            app.state.public_demo_knowledge = package.read_text(encoding="utf-8")
            await service.ingest(
                DocumentIn(
                    source_id="tayoca:public-knowledge",
                    text=app.state.public_demo_knowledge,
                    metadata={
                        "origin": "bundled-public-document",
                        "filename": package.name,
                        "source_url": "https://tayoca.com/",
                        "safe_to_share": True,
                    },
                ),
                "tayoca",
            )
        app.state.telegram_registration_triggered = False
        async with AsyncExitStack() as stack:
            await stack.enter_async_context(mcp_server.session_manager.run())
            yield

    app = FastAPI(title="RAG Demo API", version="0.3.0", lifespan=lifespan)

    async def register_after_response() -> None:
        try:
            await register_cloud_webhook(settings)
        except (httpx.HTTPError, ValueError) as exc:
            # Never expose bot token through outbound exception URLs.
            logger.warning(
                "Telegram webhook registration failed (%s)", type(exc).__name__
            )

    @app.get("/")
    async def root(
        request: Request, background_tasks: BackgroundTasks
    ) -> dict[str, str]:
        if (
            settings.telegram_auto_register
            and telegram_ready(settings)
            and os.environ.get("VERCEL_ENV") == "production"
            and not request.app.state.telegram_registration_triggered
        ):
            request.app.state.telegram_registration_triggered = True
            background_tasks.add_task(register_after_response)
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
    async def telegram_webhook(request: Request) -> dict:
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
        response = await process_update(
            payload, settings, request.app.state.rag_service, generator,
            getattr(request.app.state, "public_demo_knowledge", None),
        )
        return response or {"ok": True}

    app.mount("/", mcp_asgi)
    return app


app = create_app()
