from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="RAG_", env_file=".env", extra="ignore")

    embedding_provider: Literal["hash", "openai"] = "hash"
    embedding_dimensions: int = Field(default=128, ge=8, le=8192)
    chunk_size: int = Field(default=800, ge=50, le=10000)
    chunk_overlap: int = Field(default=120, ge=0)
    max_document_bytes: int = Field(default=1048576, ge=1024, le=10485760)
    max_source_bytes: int = Field(default=10485760, ge=1024, le=104857600)
    max_pdf_pages: int = Field(default=40, ge=1, le=500)
    ocr_language: str = "eng"
    store_backend: Literal["memory", "sqlite"] = "sqlite"
    sqlite_path: str = "data/rag.sqlite3"
    knowledge_bases_file: Path = Path("knowledge_bases.json")
    default_knowledge_base: str = "local"
    openai_api_key: str | None = None
    openai_base_url: str = "https://api.openai.com/v1"
    openai_embedding_model: str = "text-embedding-3-small"
    google_service_account_file: Path | None = None
    notion_token: str | None = None

    @model_validator(mode="after")
    def validate_configuration(self) -> "Settings":
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be less than chunk_size")
        if self.embedding_provider == "openai" and not self.openai_api_key:
            raise ValueError("RAG_OPENAI_API_KEY required for openai provider")
        return self
