import unicodedata

from rag_demo.models import DocumentIn


class Ingestor:
    """Normalize plain text documents without losing source metadata."""

    def __init__(self, max_document_bytes: int) -> None:
        self.max_document_bytes = max_document_bytes

    def normalize(self, document: DocumentIn) -> DocumentIn:
        text = unicodedata.normalize("NFC", document.text.replace("\r\n", "\n"))
        text = text.replace("\r", "\n").replace("\x00", "").strip()
        if not text:
            raise ValueError("document is empty after normalization")
        if len(text.encode("utf-8")) > self.max_document_bytes:
            raise ValueError("document exceeds configured byte limit")
        return document.model_copy(update={"text": text})
