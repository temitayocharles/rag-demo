import hashlib

from rag_demo.models import Chunk, DocumentIn


class Chunker:
    """Deterministic, overlapping text chunks with stable IDs and character offsets."""

    def __init__(self, size: int, overlap: int) -> None:
        if size <= 0 or overlap < 0 or overlap >= size:
            raise ValueError("invalid chunk size/overlap")
        self.size = size
        self.overlap = overlap

    def split(self, document: DocumentIn) -> list[Chunk]:
        text = document.text
        result: list[Chunk] = []
        start = 0
        while start < len(text):
            end = min(start + self.size, len(text))
            # Prefer nearby whitespace to avoid slicing words, without tiny chunks.
            if end < len(text):
                boundary = text.rfind(" ", start + self.size // 2, end)
                if boundary > start:
                    end = boundary + 1
            content = text[start:end]
            digest = hashlib.sha256(
                f"{document.source_id}\0{start}\0{end}\0{content}".encode()
            ).hexdigest()[:24]
            result.append(
                Chunk(
                    chunk_id=digest,
                    source_id=document.source_id,
                    text=content,
                    position=len(result),
                    start=start,
                    end=end,
                    metadata=document.metadata,
                )
            )
            if end == len(text):
                break
            next_start = end - self.overlap
            start = max(next_start, start + 1)
        return result
