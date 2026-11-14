import hashlib
import math
import re
from typing import Protocol

import httpx


class Embedder(Protocol):
    dimensions: int

    async def embed(self, texts: list[str]) -> list[list[float]]: ...


class HashEmbedder:
    """Deterministic local demo embeddings; NOT a semantic embedding model."""

    def __init__(self, dimensions: int = 128) -> None:
        self.dimensions = dimensions

    async def embed(self, texts: list[str]) -> list[list[float]]:
        vectors = []
        for text in texts:
            vector = [0.0] * self.dimensions
            for token in re.findall(r"\w+", text.lower()):
                digest = hashlib.sha256(token.encode("utf-8")).digest()
                bucket = int.from_bytes(digest[:8], "big") % self.dimensions
                vector[bucket] += 1 if digest[8] & 1 else -1
            norm = math.sqrt(sum(x * x for x in vector))
            vectors.append([x / norm for x in vector] if norm else vector)
        return vectors


class OpenAIEmbedder:
    """Embedding adapter for the OpenAI embeddings API."""

    def __init__(self, api_key: str, model: str, dimensions: int, base_url: str) -> None:
        self.api_key = api_key
        self.model = model
        self.dimensions = dimensions
        self.base_url = base_url.rstrip("/")

    async def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                f"{self.base_url}/embeddings",
                headers={"Authorization": f"Bearer {self.api_key}"},
                json={"model": self.model, "input": texts, "dimensions": self.dimensions},
            )
            response.raise_for_status()
            payload = response.json()
        ordered = sorted(payload["data"], key=lambda item: item["index"])
        if len(ordered) != len(texts):
            raise ValueError("embedding provider returned unexpected number of vectors")
        vectors = [item["embedding"] for item in ordered]
        if any(len(vector) != self.dimensions for vector in vectors):
            raise ValueError("embedding dimension mismatch")
        return vectors
