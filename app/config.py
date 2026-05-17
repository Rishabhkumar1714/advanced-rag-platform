"""
Application Configuration
Centralized settings management using Pydantic BaseSettings.
"""

from functools import lru_cache
from typing import List, Optional

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── Application ──────────────────────────────────────────────────────────
    app_name: str = Field(default="Advanced RAG Platform")
    app_version: str = Field(default="1.0.0")
    app_env: str = Field(default="development")
    debug: bool = Field(default=False)
    secret_key: str = Field(default="change-me-in-production")

    # ── Server ───────────────────────────────────────────────────────────────
    host: str = Field(default="0.0.0.0")
    port: int = Field(default=8000)
    workers: int = Field(default=4)

    # ── OpenAI ───────────────────────────────────────────────────────────────
    openai_api_key: str = Field(default="")
    openai_model: str = Field(default="gpt-4o")
    openai_embedding_model: str = Field(default="text-embedding-3-large")
    openai_max_tokens: int = Field(default=2048)
    openai_temperature: float = Field(default=0.1)

    # ── Vector Store ─────────────────────────────────────────────────────────
    faiss_index_path: str = Field(default="./data/processed/faiss_index")
    faiss_index_dimension: int = Field(default=3072)
    embedding_batch_size: int = Field(default=100)

    # ── Retrieval ────────────────────────────────────────────────────────────
    retrieval_top_k: int = Field(default=5)
    hybrid_search_alpha: float = Field(default=0.5)
    similarity_threshold: float = Field(default=0.7)
    max_context_tokens: int = Field(default=4096)
    chunk_size: int = Field(default=512)
    chunk_overlap: int = Field(default=64)
    reranker_enabled: bool = Field(default=True)
    reranker_top_n: int = Field(default=3)

    # ── Database ─────────────────────────────────────────────────────────────
    database_url: str = Field(
        default="postgresql+asyncpg://raguser:ragpass@localhost:5432/ragdb"
    )
    db_pool_size: int = Field(default=10)
    db_max_overflow: int = Field(default=20)

    # ── Redis ────────────────────────────────────────────────────────────────
    redis_url: str = Field(default="redis://localhost:6379/0")
    cache_ttl: int = Field(default=3600)

    # ── Document Processing ──────────────────────────────────────────────────
    max_file_size_mb: int = Field(default=50)
    allowed_extensions: str = Field(default=".pdf,.docx,.txt,.md,.html")
    ingestion_batch_size: int = Field(default=10)

    # ── Logging ──────────────────────────────────────────────────────────────
    log_level: str = Field(default="INFO")
    log_format: str = Field(default="json")

    # ── CORS ─────────────────────────────────────────────────────────────────
    allowed_origins: str = Field(default="http://localhost:3000")
    allowed_methods: str = Field(default="GET,POST,PUT,DELETE,OPTIONS")
    allowed_headers: str = Field(default="*")

    # ── Rate Limiting ────────────────────────────────────────────────────────
    rate_limit_per_minute: int = Field(default=60)
    rate_limit_burst: int = Field(default=10)

    # ── Metrics ──────────────────────────────────────────────────────────────
    metrics_enabled: bool = Field(default=True)
    metrics_port: int = Field(default=9090)

    @field_validator("hybrid_search_alpha")
    @classmethod
    def validate_alpha(cls, v: float) -> float:
        if not 0.0 <= v <= 1.0:
            raise ValueError("hybrid_search_alpha must be between 0 and 1")
        return v

    @property
    def allowed_extensions_list(self) -> List[str]:
        return [ext.strip() for ext in self.allowed_extensions.split(",")]

    @property
    def allowed_origins_list(self) -> List[str]:
        return [origin.strip() for origin in self.allowed_origins.split(",")]

    @property
    def is_production(self) -> bool:
        return self.app_env.lower() == "production"


@lru_cache()
def get_settings() -> Settings:
    """Return cached application settings."""
    return Settings()


settings = get_settings()
