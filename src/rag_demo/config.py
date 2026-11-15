from pathlib import Path
from typing import Literal
from urllib.parse import urlparse

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


def validate_api_url(value: str, label: str) -> None:
    url = urlparse(value)
    if url.username or url.password or url.query or url.fragment:
        raise ValueError(f"{label} must not contain credentials, query or fragment")
    if url.scheme == "https" and url.hostname:
        return
    if url.scheme == "http" and url.hostname in {"localhost", "127.0.0.1", "::1"}:
        return
    raise ValueError(f"{label} requires HTTPS or HTTP on loopback")


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="RAG_", env_file=".env", extra="ignore")

    embedding_provider: Literal["hash", "openai_compatible", "openai"] = "hash"
    embedding_dimensions: int = Field(default=128, ge=8, le=8192)
    embedding_api_key: str | None = None
    embedding_base_url: str = "https://api.openai.com/v1"
    embedding_model: str = "text-embedding-3-small"
    embedding_send_dimensions: bool = True

    llm_provider: Literal["disabled", "openai_compatible"] = "disabled"
    llm_api_key: str | None = None
    llm_base_url: str = "https://api.openai.com/v1"
    llm_model: str = "set-model-in-env"
    llm_temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    llm_max_tokens: int = Field(default=512, ge=16, le=8192)
    llm_timeout_seconds: float = Field(default=45, ge=1, le=180)

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
    demo_mode: bool = False
    telegram_bot_token: str | None = None
    telegram_chat_id: str | None = None
    telegram_auto_register: bool = True
    telegram_webhook_origin: str = "https://rag-demo-ruddy.vercel.app"

    # Backward compatible aliases for existing installations.
    openai_api_key: str | None = None
    openai_base_url: str = "https://api.openai.com/v1"
    openai_embedding_model: str = "text-embedding-3-small"

    google_service_account_file: Path | None = None
    notion_token: str | None = None

    @model_validator(mode="after")
    def validate_configuration(self) -> "Settings":
        if self.telegram_auto_register and self.telegram_bot_token:
            validate_api_url(self.telegram_webhook_origin, "RAG_TELEGRAM_WEBHOOK_ORIGIN")
        if self.telegram_chat_id and not (
            self.telegram_chat_id.removeprefix("-").isdigit()
        ):
            raise ValueError("RAG_TELEGRAM_CHAT_ID must be a numeric Telegram chat ID")
        if self.demo_mode and self.store_backend != "memory":
            raise ValueError("RAG_DEMO_MODE requires RAG_STORE_BACKEND=memory")
        if self.demo_mode and self.llm_provider != "disabled":
            raise ValueError("RAG_DEMO_MODE requires RAG_LLM_PROVIDER=disabled")
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be less than chunk_size")
        if self.embedding_provider == "openai":
            if not (self.embedding_api_key or self.openai_api_key):
                raise ValueError("embedding API key required for legacy openai provider")
            validate_api_url(self.openai_base_url, "RAG_OPENAI_BASE_URL")
        if self.embedding_provider == "openai_compatible":
            validate_api_url(self.embedding_base_url, "RAG_EMBEDDING_BASE_URL")
        if self.llm_provider != "disabled":
            validate_api_url(self.llm_base_url, "RAG_LLM_BASE_URL")
            if self.llm_model == "set-model-in-env":
                raise ValueError("RAG_LLM_MODEL must be configured when LLM is enabled")
        return self
