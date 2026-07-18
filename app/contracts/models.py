"""Model registry, health, bundle, and recommendation contracts."""

from pathlib import PurePosixPath
from typing import Any, Literal

from pydantic import Field, model_validator

from app.contracts.common import ContractModel
from app.contracts.enums import (
    DevicePreference,
    ModelFamily,
    ModelStatus,
    ModelVariant,
    QualityTier,
    RoiMode,
)


class ModelMetadata(ContractModel):
    model_id: str = Field(min_length=1, max_length=160)
    family: ModelFamily
    variant: ModelVariant
    quality_tier: QualityTier
    version: str
    status: ModelStatus
    supports_box_prompt: bool
    default_threshold: float | None = Field(default=None, ge=0, le=1)
    preprocess_profile: str
    postprocess_profile: str
    applicable_materials: list[str] = Field(default_factory=list)
    metrics: dict[str, float] = Field(default_factory=dict)
    metric_context: dict[str, Any] = Field(default_factory=dict)
    notes: str = ""
    health_error: str | None = None
    adapter_path: str | None = None
    weight_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    config_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    model_card_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    adapter_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")


class ModelBundleReference(ContractModel):
    """Persistent, content-addressed references sufficient to reopen one frozen model bundle."""

    schema_version: Literal[1] = 1
    bundle_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    manifest_ref: str
    weight_ref: str
    config_ref: str
    model_card_ref: str
    adapter_ref: str
    adapter_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def artifact_references_are_relative_and_normalized(self) -> "ModelBundleReference":
        for name in (
            "manifest_ref",
            "weight_ref",
            "config_ref",
            "model_card_ref",
            "adapter_ref",
        ):
            value = getattr(self, name)
            path = PurePosixPath(value)
            if path.is_absolute() or ".." in path.parts or str(path) != value:
                raise ValueError(f"{name} must be a normalized relative artifact reference")
        return self


class ModelHealth(ContractModel):
    model_id: str
    status: ModelStatus
    error_summary: str | None = None
    device: str | None = None
    weight_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")


class ModelListData(ContractModel):
    models: list[ModelMetadata] = Field(default_factory=list)


class ModelRecommendationRequest(ContractModel):
    image_id: str
    roi_mode: RoiMode
    target_profile: ModelVariant = ModelVariant.GENERAL
    prefer: str = Field(default="accuracy", pattern=r"^(speed|balance|accuracy)$")
    device: DevicePreference = DevicePreference.AUTO
    max_gpu_memory_mb: int | None = Field(default=None, ge=0)


class ModelCandidate(ContractModel):
    model_id: str
    score: float = Field(ge=0, le=1)
    reasons: list[str] = Field(default_factory=list)


class ModelRecommendationData(ContractModel):
    candidates: list[ModelCandidate] = Field(default_factory=list)
    requires_user_confirmation: bool = True
