from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request

from rag_demo.chunker import Chunker
from rag_demo.config import Settings
from rag_demo.embedder import HashEmbedder, OpenAIEmbedder
from rag_demo.ingestor import Ingestor
from rag_demo.knowledge import KnowledgeBases
from rag_demo.models import DocumentIn, IngestResult, SearchRequest, SearchResponse
from rag_demo.service import RagService
from rag_demo.sqlite_store import SQLiteVectorStore
from rag_demo.store import InMemoryVectorStore


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or Settings()
    knowledge = KnowledgeBases(settings.knowledge_bases_file)
    knowledge.require(settings.default_knowledge_base)
    if settings.embedding_provider == "openai":
        assert settings.openai_api_key is not None
        embedder = OpenAIEmbedder(
            settings.openai_api_key, settings.openai_embedding_model,
            settings.embedding_dimensions, settings.openai_base_url,
        )
    else:
        embedder = HashEmbedder(settings.embedding_dimensions)
    store = (SQLiteVectorStore(settings.embedding_dimensions, settings.sqlite_path)
             if settings.store_backend == "sqlite"
             else InMemoryVectorStore(settings.embedding_dimensions))
    service = RagService(
        Ingestor(settings.max_document_bytes),
        Chunker(settings.chunk_size, settings.chunk_overlap),
        embedder, store,
    )

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.rag_service = service
        yield

    app = FastAPI(title="RAG Demo API", version="0.2.0", lifespan=lifespan)

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

    return app


app = create_app()
