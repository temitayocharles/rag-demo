from contextlib import asynccontextmanager

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


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or Settings()
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
        yield

    app = FastAPI(title="RAG Demo API", version="0.3.0", lifespan=lifespan)

    @app.get("/healthz")
    async def healthz() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/v1/knowledge-bases")
    async def knowledge_bases() -> dict:
        return {"knowledge_bases": knowledge.summary()}

    @app.post("/v1/documents", response_model=IngestResult)
    async def ingest(document: DocumentIn, request: Request) -> IngestResult:
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

    return app


app = create_app()
