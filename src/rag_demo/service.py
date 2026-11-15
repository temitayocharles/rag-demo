from rag_demo.chunker import Chunker
from rag_demo.embedder import Embedder
from rag_demo.ingestor import Ingestor
from rag_demo.models import DocumentIn, IngestResult, SearchHit, StoredChunk
from rag_demo.store import VectorStore


class RagService:
    def __init__(
        self, ingestor: Ingestor, chunker: Chunker, embedder: Embedder, store: VectorStore
    ) -> None:
        self.ingestor = ingestor
        self.chunker = chunker
        self.embedder = embedder
        self.store = store

    async def ingest(self, document: DocumentIn, knowledge_base: str = "local") -> IngestResult:
        tagged = document.model_copy(update={
            "metadata": {**document.metadata, "knowledge_base": knowledge_base}
        })
        normalized = self.ingestor.normalize(tagged)
        chunks = self.chunker.split(normalized)
        vectors = await self.embedder.embed([chunk.text for chunk in chunks])
        if len(vectors) != len(chunks):
            raise ValueError("embedding count mismatch")
        await self.store.replace_source(
            normalized.source_id,
            [StoredChunk(chunk=c, vector=v) for c, v in zip(chunks, vectors, strict=True)],
            knowledge_base,
        )
        return IngestResult(source_id=normalized.source_id, chunks_upserted=len(chunks))

    async def search(self, query: str, top_k: int,
                     knowledge_base: str = "local") -> list[SearchHit]:
        vectors = await self.embedder.embed([query])
        if len(vectors) != 1:
            raise ValueError("embedding provider returned invalid query vector")
        return await self.store.search(vectors[0], top_k, knowledge_base)
