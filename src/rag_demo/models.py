from typing import Any

from pydantic import BaseModel, Field


class DocumentIn(BaseModel):
    source_id: str = Field(min_length=1, max_length=256)
    text: str = Field(min_length=1)
    metadata: dict[str, Any] = Field(default_factory=dict)
    knowledge_base: str | None = None


class Chunk(BaseModel):
    chunk_id: str
    source_id: str
    text: str
    position: int
    start: int
    end: int
    metadata: dict[str, Any] = Field(default_factory=dict)


class StoredChunk(BaseModel):
    chunk: Chunk
    vector: list[float]


class IngestResult(BaseModel):
    source_id: str
    chunks_upserted: int


class SearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=4000)
    top_k: int = Field(default=5, ge=1, le=50)
    knowledge_base: str | None = None


class SearchHit(BaseModel):
    score: float
    chunk: Chunk


class SearchResponse(BaseModel):
    hits: list[SearchHit]
