"""Local/remote file-content extraction. Call only on trusted or bounded inputs."""
import io
from pathlib import Path

from rag_demo.config import Settings
from rag_demo.models import DocumentIn

TEXT_SUFFIXES = {".txt", ".md", ".markdown", ".csv", ".json"}
SUPPORTED_SUFFIXES = TEXT_SUFFIXES | {".html", ".htm", ".pdf", ".docx", ".png", ".jpg", ".jpeg", ".tif", ".tiff"}


def extract_text(name: str, content: bytes, settings: Settings) -> str:
    if len(content) > settings.max_source_bytes:
        raise ValueError(f"source too large: {name}")
    suffix = Path(name).suffix.lower()
    if suffix in TEXT_SUFFIXES:
        return content.decode("utf-8-sig")
    if suffix in {".html", ".htm"}:
        from bs4 import BeautifulSoup
        return BeautifulSoup(content, "html.parser").get_text(" ", strip=True)
    if suffix == ".docx":
        from docx import Document
        doc = Document(io.BytesIO(content))
        return "\n".join(p.text for p in doc.paragraphs if p.text.strip())
    if suffix == ".pdf":
        import pymupdf
        pdf = pymupdf.open(stream=content, filetype="pdf")
        try:
            if len(pdf) > settings.max_pdf_pages:
                raise ValueError("PDF exceeds configured page limit")
            pages = []
            for page in pdf:
                text = page.get_text("text")
                if len(text.strip()) < 20:
                    try:
                        text = page.get_text(textpage=page.get_textpage_ocr(
                            language=settings.ocr_language))
                    except Exception as exc:
                        raise RuntimeError(
                            "OCR failed; install Tesseract with language data, or check PDF"
                        ) from exc
                pages.append(text)
            return "\n\n".join(pages)
        finally:
            pdf.close()
    if suffix in {".png", ".jpg", ".jpeg", ".tif", ".tiff"}:
        import pytesseract
        from PIL import Image
        with Image.open(io.BytesIO(content)) as image:
            if image.width * image.height > 25_000_000:
                raise ValueError("image pixel limit exceeded")
            return pytesseract.image_to_string(image, lang=settings.ocr_language)
    raise ValueError(f"unsupported file type: {suffix}")


def as_document(source_id: str, name: str, content: bytes,
                settings: Settings, metadata: dict | None = None) -> DocumentIn:
    text = extract_text(name, content, settings)
    return DocumentIn(
        source_id=source_id,
        text=text,
        metadata={"filename": name, **(metadata or {})},
    )
