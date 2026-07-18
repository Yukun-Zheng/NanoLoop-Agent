"""Validated process configuration with repository-relative defaults."""

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    app_env: Literal["development", "test", "production"] = "development"
    database_url: str = "sqlite:///./data/nanoloop.db"
    output_root: Path = Path("./outputs")
    model_registry_path: Path = Path("./model_artifacts/registry.yaml")
    model_snapshot_root: Path = Path("./data/model-snapshots")
    model_device: str = "auto"
    knowledge_source_dir: Path = Path("./knowledge_base/sources")
    faiss_index_path: Path = Path("./knowledge_base/index/faiss.index")
    embedding_model: str = "BAAI/bge-small-zh-v1.5"
    embedding_model_revision: str | None = Field(
        default=None,
        pattern=r"^[0-9a-f]{40,64}$",
    )
    knowledge_max_pdf_pages: int = Field(default=2_000, ge=1, le=100_000)
    knowledge_max_extracted_chars: int = Field(
        default=10_000_000,
        ge=1_000,
        le=100_000_000,
    )
    knowledge_max_chunks_per_document: int = Field(
        default=20_000,
        ge=1,
        le=100_000,
    )
    knowledge_max_vector_index_chunks: int = Field(
        default=100_000,
        ge=1,
        le=1_000_000,
    )
    embedding_index_batch_size: int = Field(default=128, ge=1, le=4_096)
    data_distribution_evidence_limit: int = Field(default=200, ge=10, le=1_000)
    llm_provider: Literal["openai_compatible", "extractive"] = "extractive"
    llm_base_url: str | None = None
    llm_api_key: str | None = None
    llm_model: str | None = None
    max_upload_mb: int = Field(default=200, ge=1, le=2048)
    max_request_mb: int = Field(default=512, ge=1, le=4096)
    analysis_worker_count: int = Field(default=2, ge=1, le=16)
    analysis_queue_capacity: int = Field(default=32, ge=1, le=1000)
    analysis_scheduler_poll_seconds: float = Field(default=0.5, ge=0.05, le=60)
    shutdown_timeout_seconds: float = Field(default=30.0, ge=0, le=300)
    log_level: str = "INFO"
    api_prefix: str = "/api/v1"
    trusted_hosts: str = "localhost,127.0.0.1,testserver"
    cors_allow_origins: str = ""

    @field_validator(
        "embedding_model_revision",
        "llm_base_url",
        "llm_api_key",
        "llm_model",
        mode="before",
    )
    @classmethod
    def empty_string_is_none(cls, value: object) -> object:
        return None if value == "" else value

    @field_validator("log_level")
    @classmethod
    def normalize_log_level(cls, value: str) -> str:
        normalized = value.upper()
        if normalized not in {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}:
            raise ValueError("unsupported LOG_LEVEL")
        return normalized

    @model_validator(mode="after")
    def request_limit_covers_one_upload(self) -> "Settings":
        if self.max_request_mb < self.max_upload_mb:
            raise ValueError("MAX_REQUEST_MB must be at least MAX_UPLOAD_MB")
        return self


@lru_cache
def get_settings() -> Settings:
    return Settings()
