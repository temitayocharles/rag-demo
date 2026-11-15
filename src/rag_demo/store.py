import math
from collections.abc import Sequence
from typing import Protocol

from rag_demo.models import SearchHit, StoredChunk


class VectorStore(Protocol):
    async def replace_source(self, source_id: str, chunks: Sequence[StoredChunk],
                             knowledge_base: str = "local") -> None: ...
    async def search(self, vector: list[float], limit: int,
                     knowledge_base: str = "local") -> list[SearchHit]: ...


class InMemoryVectorStore:
    """Process-local example store. Restarting the server clears stored documents."""

    def __init__(self, dimensions: int) -> None:
        self.dimensions = dimensions
        self._items: dict[tuple[str, str], StoredChunk] = {}

    async def replace_source(self, source_id: str, chunks: Sequence[StoredChunk],
                             knowledge_base: str = "local") -> None:
        if any(x.chunk.source_id != source_id for x in chunks):
            raise ValueError("source_id mismatch")
        if any(len(x.vector) != self.dimensions for x in chunks):
            raise ValueError("vector dimensions mismatch")
        self._items = {
            key: value for key, value in self._items.items()
            if not (key[0] == knowledge_base and value.chunk.source_id == source_id)
        }
        self._items.update({(knowledge_base, chunk.chunk.chunk_id): chunk for chunk in chunks})

    async def search(self, vector: list[float], limit: int,
                     knowledge_base: str = "local") -> list[SearchHit]:
        if len(vector) != self.dimensions:
            raise ValueError("query dimensions mismatch")
        query_norm = math.sqrt(sum(x * x for x in vector))
        hits = []
        for (scope, _), stored in self._items.items():
            if scope != knowledge_base:
                continue
            vec_norm = math.sqrt(sum(x * x for x in stored.vector))
            dot = sum(a * b for a, b in zip(vector, stored.vector, strict=True))
            score = dot / (query_norm * vec_norm) if query_norm and vec_norm else 0.0
            hits.append(SearchHit(score=score, chunk=stored.chunk))
        return sorted(hits, key=lambda hit: (-hit.score, hit.chunk.chunk_id))[:limit]
