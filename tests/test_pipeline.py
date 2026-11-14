import asyncio

import pytest

from rag_demo.chunker import Chunker
from rag_demo.embedder import HashEmbedder
from rag_demo.ingestor import Ingestor
from rag_demo.models import DocumentIn
from rag_demo.service import RagService
from rag_demo.store import InMemoryVectorStore


def test_chunking_preserves_text_and_offsets():
    document = DocumentIn(source_id="doc-1", text="alpha beta gamma delta epsilon zeta")
    chunks = Chunker(size=15, overlap=4).split(document)
    assert len(chunks) > 1
    assert all(document.text[c.start : c.end] == c.text for c in chunks)
    assert chunks[-1].end == len(document.text)


def test_ingestor_normalizes_and_rejects_empty():
    ingestor = Ingestor(1024)
    assert ingestor.normalize(DocumentIn(source_id="x", text=" a\r\nb ")).text == "a\nb"
    with pytest.raises(ValueError, match="empty"):
        ingestor.normalize(DocumentIn(source_id="x", text=" \x00 "))


def test_ingest_replace_and_search():
    async def run():
        service = RagService(Ingestor(1024), Chunker(50, 10), HashEmbedder(32),
                             InMemoryVectorStore(32))
        result = await service.ingest(DocumentIn(source_id="one", text="oranges apples pears"))
        assert result.chunks_upserted == 1
        hits = await service.search("oranges", 5)
        assert len(hits) == 1
        assert hits[0].chunk.source_id == "one"
        await service.ingest(DocumentIn(source_id="one", text="bananas"))
        hits = await service.search("bananas", 5)
        assert len(hits) == 1
        assert hits[0].chunk.text == "bananas"
    asyncio.run(run())


def test_vector_dimension_contract():
    async def run():
        store = InMemoryVectorStore(32)
        with pytest.raises(ValueError, match="dimensions"):
            await store.search([1.0], 5)
    asyncio.run(run())
