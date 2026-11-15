import json
import math
import sqlite3
from collections.abc import Sequence
from pathlib import Path

from rag_demo.models import Chunk, SearchHit, StoredChunk


class SQLiteVectorStore:
    """Durable, single-machine proof-of-concept vector store. Not for Vercel."""

    def __init__(self, dimensions: int, path: str):
        self.dimensions = dimensions
        self.path = path
        if path != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(path, check_same_thread=False, timeout=30)
        self.db.execute("PRAGMA busy_timeout=30000")
        self.db.execute("""CREATE TABLE IF NOT EXISTS chunks (
            knowledge_base TEXT NOT NULL, source_id TEXT NOT NULL,
            chunk_id TEXT NOT NULL, payload TEXT NOT NULL, vector TEXT NOT NULL,
            PRIMARY KEY(knowledge_base, chunk_id))""")
        self.db.execute(
            "CREATE INDEX IF NOT EXISTS idx_chunks_scope ON chunks(knowledge_base, source_id)"
        )
        self.db.commit()

    async def replace_source(self, source_id: str, chunks: Sequence[StoredChunk],
                             knowledge_base: str = "local") -> None:
        if any(x.chunk.source_id != source_id for x in chunks):
            raise ValueError("source_id mismatch")
        if any(len(x.vector) != self.dimensions for x in chunks):
            raise ValueError("vector dimensions mismatch")
        with self.db:
            self.db.execute("DELETE FROM chunks WHERE knowledge_base=? AND source_id=?",
                            (knowledge_base, source_id))
            self.db.executemany(
                "INSERT INTO chunks VALUES (?, ?, ?, ?, ?)",
                [(knowledge_base, source_id, x.chunk.chunk_id,
                  x.chunk.model_dump_json(), json.dumps(x.vector)) for x in chunks],
            )

    async def search(self, vector: list[float], limit: int,
                     knowledge_base: str = "local") -> list[SearchHit]:
        if len(vector) != self.dimensions:
            raise ValueError("query dimensions mismatch")
        query_norm = math.sqrt(sum(x*x for x in vector))
        hits = []
        for payload, stored_json in self.db.execute(
            "SELECT payload, vector FROM chunks WHERE knowledge_base=?", (knowledge_base,)
        ):
            stored = json.loads(stored_json)
            if len(stored) != self.dimensions:
                raise ValueError("stored vector dimensions differ from configured model; reindex")
            norm = math.sqrt(sum(x*x for x in stored))
            dot = sum(a*b for a, b in zip(vector, stored, strict=True))
            hits.append(SearchHit(score=dot/(query_norm*norm) if query_norm and norm else 0,
                                  chunk=Chunk.model_validate_json(payload)))
        return sorted(hits, key=lambda x: (-x.score, x.chunk.chunk_id))[:limit]
