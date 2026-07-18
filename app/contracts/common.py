"""Shared response envelopes and scalar conventions."""

from datetime import UTC, datetime
from typing import Any, Generic, Literal, TypeVar

from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.contracts.enums import ApiStatus

T = TypeVar("T")


def utc_now() -> datetime:
    """Return a timezone-aware UTC timestamp."""

    return datetime.now(UTC)


class ContractModel(BaseModel):
    """Base contract with strict input and stable JSON serialization."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True, use_enum_values=False)


class ApiErrorPayload(ContractModel):
    code: str = Field(min_length=1, max_length=80)
    message: str = Field(min_length=1, max_length=1000)
    details: dict[str, Any] = Field(default_factory=dict)
    retryable: bool = False


class ApiResponse(ContractModel, Generic[T]):
    request_id: str = Field(min_length=1, max_length=100)
    status: ApiStatus
    data: T | None = None
    error: ApiErrorPayload | None = None

    @model_validator(mode="after")
    def validate_success_error_shape(self) -> "ApiResponse[T]":
        if self.status == ApiStatus.ERROR:
            if self.error is None or self.data is not None:
                raise ValueError("error responses require error and forbid data")
        elif self.error is not None:
            raise ValueError("successful responses cannot contain error")
        return self


class Pagination(ContractModel):
    limit: int = Field(default=50, ge=1, le=200)
    offset: int = Field(default=0, ge=0)
    total: int = Field(ge=0)


class HealthComponent(ContractModel):
    status: Literal["healthy", "degraded", "unavailable"]
    detail: str | None = None


class HealthData(ContractModel):
    service: HealthComponent
    database: HealthComponent
    model_registry: HealthComponent
    rag_index: HealthComponent
    version: str
