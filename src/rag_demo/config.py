from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="RAG_", env_file=".env", extra="ignore")

    embedding_provider: str = "hash"
    embedding_dimensions: int = Field(default=128, ge=8, le=8192)
    chunk_size: int = Field(default=800, ge=50, le=10000)
    chunk_overlap: int = Field(default=120, ge=0)
    max_document_bytes: int = Field(default=1048576, ge=1024, le=10485760)
    openai_api_key: str | None = None
    openai_base_url: str = "https://api.openai.com/v1"
    openai_embedding_model: str = "text-embedding-3-small"

    @model_validator(mode="after")
    def validate_configuration(self) -> "Settings":
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be less than chunk_size")
        if self.embedding_provider not in {"hash", "openai"}:
            raise ValueError("unsupported embedding provider")
        if self.embedding_provider == "openai" and not self.openai_api_key:
            raise ValueError("RAG_OPENAI_API_KEY required for openai provider")
        return self
