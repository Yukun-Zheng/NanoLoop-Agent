"""Validated process configuration with repository-relative defaults."""

import re
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from app.contracts.identity import AuthMode

_API_KEY_PATTERN = re.compile(r"^[A-Za-z0-9_-]{32,128}$")


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
        hide_input_in_errors=True,
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
    auth_mode: Literal["auto", "disabled", "shared_key", "principal"] = "auto"
    nanoloop_api_key: SecretStr | None = None
    credential_pepper: SecretStr | None = None
    api_rate_limit_requests: int = Field(default=0, ge=0, le=1_000_000)
    api_rate_limit_window_seconds: float = Field(default=60.0, gt=0, le=3_600)
    api_principal_preauth_rate_limit_requests: int = Field(
        default=600,
        ge=0,
        le=1_000_000,
    )
    api_principal_preauth_rate_limit_window_seconds: float = Field(
        default=60.0,
        gt=0,
        le=3_600,
    )
    api_rate_limit_max_buckets: int = Field(default=4_096, ge=1, le=1_000_000)

    @field_validator(
        "embedding_model_revision",
        "llm_base_url",
        "llm_api_key",
        "llm_model",
        "nanoloop_api_key",
        "credential_pepper",
        mode="before",
    )
    @classmethod
    def empty_string_is_none(cls, value: object) -> object:
        if isinstance(value, SecretStr) and value.get_secret_value() == "":
            return None
        return None if value == "" else value

    @field_validator("log_level")
    @classmethod
    def normalize_log_level(cls, value: str) -> str:
        normalized = value.upper()
        if normalized not in {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}:
            raise ValueError("unsupported LOG_LEVEL")
        return normalized

    @field_validator("nanoloop_api_key")
    @classmethod
    def validate_api_key(cls, value: SecretStr | None) -> SecretStr | None:
        if value is None:
            return None
        if not _API_KEY_PATTERN.fullmatch(value.get_secret_value()):
            raise ValueError(
                "NANOLOOP_API_KEY must contain 32-128 URL-safe letters, digits, '_' or '-'"
            )
        return value

    @model_validator(mode="after")
    def request_limit_covers_one_upload(self) -> "Settings":
        if self.max_request_mb < self.max_upload_mb:
            raise ValueError("MAX_REQUEST_MB must be at least MAX_UPLOAD_MB")
        if self.effective_auth_mode is AuthMode.SHARED_KEY and self.nanoloop_api_key is None:
            raise ValueError("NANOLOOP_API_KEY is required when AUTH_MODE=shared_key")
        if self.effective_auth_mode is AuthMode.PRINCIPAL:
            if self.credential_pepper is None:
                raise ValueError("CREDENTIAL_PEPPER is required when AUTH_MODE=principal")
            if len(self.credential_pepper.get_secret_value().encode("utf-8")) < 32:
                raise ValueError(
                    "CREDENTIAL_PEPPER must contain at least 32 bytes when AUTH_MODE=principal"
                )
        return self

    @property
    def effective_auth_mode(self) -> AuthMode:
        """Resolve the compatibility ``auto`` mode without changing explicit modes."""

        if self.auth_mode == "auto":
            return (
                AuthMode.SHARED_KEY
                if self.nanoloop_api_key is not None
                else AuthMode.DISABLED
            )
        return AuthMode(self.auth_mode)


@lru_cache
def get_settings() -> Settings:
    return Settings()
