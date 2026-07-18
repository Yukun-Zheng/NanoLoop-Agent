"""Analysis jobs, image metadata, ROI, run, measurement, and export DTOs."""

from datetime import datetime
from typing import Any, Literal

from pydantic import Field, model_validator

from app.contracts.analysis_config import (
    ExecutionBuildProvenance,
    MorphometryConfig,
    PostprocessProfile,
    QualityGateConfig,
)
from app.contracts.common import ContractModel
from app.contracts.enums import DevicePreference, JobStatus, QualityStatus, RoiMode, ScaleMode
from app.contracts.execution import ExecutionRuntimeProvenance
from app.contracts.models import ModelBundleReference


class ScaleInput(ContractModel):
    mode: ScaleMode
    value: float | None = Field(default=None, gt=0)

    @model_validator(mode="after")
    def validate_mode_value(self) -> "ScaleInput":
        if self.mode == ScaleMode.NM_PER_PIXEL and self.value is None:
            raise ValueError("nm_per_pixel mode requires a positive value")
        if self.mode == ScaleMode.PIXEL_ONLY and self.value is not None:
            raise ValueError("pixel_only mode cannot include a physical scale")
        return self


class ImageMetadataInput(ContractModel):
    filename: str = Field(min_length=1, max_length=255)
    sample_id: str = Field(min_length=1, max_length=120)
    material_name: str | None = Field(default=None, max_length=255)
    material_formula: str | None = Field(default=None, max_length=255)
    experiment_conditions: dict[str, Any] = Field(default_factory=dict)
    scale: ScaleInput = Field(default_factory=lambda: ScaleInput(mode=ScaleMode.PIXEL_ONLY))


class CreateAnalysisMetadata(ContractModel):
    job_name: str = Field(min_length=1, max_length=255)
    images: list[ImageMetadataInput] = Field(min_length=1, max_length=20)

    @model_validator(mode="after")
    def validate_unique_filenames(self) -> "CreateAnalysisMetadata":
        names = [item.filename for item in self.images]
        if len(names) != len(set(names)):
            raise ValueError("metadata filenames must be unique within a job")
        return self


class AnalysisJobDTO(ContractModel):
    job_id: str
    name: str
    status: JobStatus
    config: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime
    updated_at: datetime
    error_code: str | None = None


class ImageAssetDTO(ContractModel):
    image_id: str
    job_id: str
    filename: str
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    bit_depth: int = Field(gt=0)
    sample_id: str
    material_name: str | None = None
    material_formula: str | None = None
    experiment_conditions: dict[str, Any] = Field(default_factory=dict)
    scale_nm_per_pixel: float | None = Field(default=None, gt=0)
    analysis_roi: "AnalysisROI"
    original_download_url: str | None = None


class PixelRect(ContractModel):
    """A half-open rectangle in original image pixels."""

    x1: int = Field(ge=0)
    y1: int = Field(ge=0)
    x2: int = Field(gt=0)
    y2: int = Field(gt=0)

    @model_validator(mode="after")
    def validate_coordinate_order(self) -> "PixelRect":
        if self.x1 >= self.x2 or self.y1 >= self.y2:
            raise ValueError("PixelRect requires x1 < x2 and y1 < y2")
        return self


class InvalidPixelRegion(PixelRect):
    reason: str = Field(default="instrument_bar", max_length=120)


class AnalysisROI(ContractModel):
    schema_version: Literal[1] = 1
    coordinate_space: Literal["original_px"] = "original_px"
    valid_rect: PixelRect
    invalid_rects: list[InvalidPixelRegion] = Field(default_factory=list)
    source: Literal["none", "manual", "detected"] = "none"
    revision: int = Field(default=1, ge=1)


class ROIBox(ContractModel):
    box_id: str | None = None
    label: str = Field(default="", max_length=120)
    x1: int = Field(ge=0)
    y1: int = Field(ge=0)
    x2: int = Field(gt=0)
    y2: int = Field(gt=0)
    active: bool = True

    @model_validator(mode="after")
    def validate_coordinate_order(self) -> "ROIBox":
        if self.x1 >= self.x2 or self.y1 >= self.y2:
            raise ValueError("ROIBox uses half-open coordinates and requires x1 < x2, y1 < y2")
        return self


class BoxSetDTO(ContractModel):
    image_id: str
    revision: int = Field(ge=0)
    boxes: list[ROIBox] = Field(default_factory=list, max_length=20)


class ReplaceBoxesRequest(ContractModel):
    expected_revision: int = Field(ge=0)
    boxes: list[ROIBox] = Field(default_factory=list, max_length=20)


class InferenceOptions(ContractModel):
    threshold: float | None = Field(default=None, ge=0, le=1)
    min_area_px: int = Field(default=8, ge=0)
    watershed_enabled: bool = False
    exclude_border: bool = True
    device: DevicePreference = DevicePreference.AUTO
    seed: int = 42


class RunConfiguration(ContractModel):
    schema_version: Literal[1, 2, 3] = 1
    provenance_status: Literal["complete", "legacy_fallback"] = "legacy_fallback"
    provenance_warnings: list[str] = Field(
        default_factory=lambda: ["legacy_run_configuration_incomplete"]
    )
    model_id: str
    model_version: str
    adapter_path: str | None = None
    weight_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    config_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    model_card_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    adapter_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    model_bundle: ModelBundleReference | None = None
    roi_mode: RoiMode
    box_revision: int | None = Field(default=None, ge=0)
    boxes: list[ROIBox] = Field(default_factory=list, max_length=20)
    analysis_roi: AnalysisROI
    inference: InferenceOptions
    preprocess_profile: str
    postprocess_profile: str
    image_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    scale_nm_per_pixel: float | None = Field(default=None, gt=0)
    resolved_postprocess: PostprocessProfile | None = None
    resolved_morphometry: MorphometryConfig | None = None
    resolved_quality_gate: QualityGateConfig | None = None
    execution_build: ExecutionBuildProvenance | None = None
    roi_context_px: int = Field(default=16, ge=0, le=4096)
    review_source: Literal["model_inference", "corrected_mask"] = "model_inference"
    corrected_mask_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    created_at: datetime

    @model_validator(mode="after")
    def validate_reproducibility_snapshot(self) -> "RunConfiguration":
        if self.provenance_status == "complete":
            missing = [
                name
                for name in (
                    "adapter_path",
                    "weight_sha256",
                    "config_sha256",
                    "model_card_sha256",
                    "image_sha256",
                    "resolved_postprocess",
                    "resolved_morphometry",
                    "resolved_quality_gate",
                    "execution_build",
                )
                if getattr(self, name) is None
            ]
            if self.schema_version == 3:
                missing.extend(
                    name
                    for name in ("adapter_sha256", "model_bundle")
                    if getattr(self, name) is None
                )
                if self.execution_build is not None:
                    missing.extend(
                        f"execution_build.{name}"
                        for name in (
                            "installed_dependencies_sha256",
                            "application_source_sha256",
                        )
                        if getattr(self.execution_build, name) is None
                    )
                if (
                    self.model_bundle is not None
                    and self.adapter_sha256 is not None
                    and self.model_bundle.adapter_sha256 != self.adapter_sha256
                ):
                    raise ValueError(
                        "model_bundle adapter digest must match frozen adapter_sha256"
                    )
            if self.schema_version not in {2, 3} or missing:
                raise ValueError(
                    "complete run provenance requires schema_version=2/3 and all "
                    "resolved settings; "
                    f"missing={missing}"
                )
        elif "legacy_run_configuration_incomplete" not in self.provenance_warnings:
            self.provenance_warnings.append("legacy_run_configuration_incomplete")
        return self


class CreateRunsRequest(ContractModel):
    image_ids: list[str] = Field(min_length=1, max_length=20)
    model_ids: list[str] = Field(min_length=1, max_length=3)
    roi_mode: RoiMode
    box_revisions: dict[str, int] = Field(default_factory=dict)
    inference: InferenceOptions = Field(default_factory=InferenceOptions)

    @model_validator(mode="before")
    @classmethod
    def upgrade_single_image_box_revision(cls, data: Any) -> Any:
        """Accept the v2.0 single-image shorthand without exposing an ambiguous schema."""

        if not isinstance(data, dict) or "box_revision" not in data:
            return data
        upgraded = dict(data)
        legacy_revision = upgraded.pop("box_revision")
        image_ids = upgraded.get("image_ids", [])
        if len(image_ids) != 1:
            raise ValueError("legacy box_revision is valid only when image_ids has one item")
        if "box_revisions" in upgraded:
            raise ValueError("send box_revisions or box_revision, not both")
        upgraded["box_revisions"] = {image_ids[0]: legacy_revision}
        return upgraded

    @model_validator(mode="after")
    def validate_box_revision(self) -> "CreateRunsRequest":
        if len(set(self.image_ids)) != len(self.image_ids):
            raise ValueError("image_ids must be unique")
        if len(set(self.model_ids)) != len(self.model_ids):
            raise ValueError("model_ids must be unique")
        if self.roi_mode == RoiMode.BOXES:
            if set(self.box_revisions) != set(self.image_ids):
                raise ValueError("boxes mode requires one saved revision for every image_id")
            if any(revision < 0 for revision in self.box_revisions.values()):
                raise ValueError("box revisions cannot be negative")
        elif self.box_revisions:
            raise ValueError("full_image mode does not accept box_revisions")
        return self


class CreateRunsData(ContractModel):
    run_ids: list[str]


class RunArtifacts(ContractModel):
    mask_url: str | None = None
    overlay_url: str | None = None
    probability_url: str | None = None
    instances_url: str | None = None
    labeled_particles_url: str | None = None
    particles_csv_url: str | None = None
    quality_report_url: str | None = None
    execution_provenance_url: str | None = None


class ParticleRecordDTO(ContractModel):
    particle_id: str
    run_id: str
    instance_index: int = Field(ge=1)
    area_px: float = Field(ge=0)
    perimeter_px: float = Field(ge=0)
    equivalent_diameter_px: float = Field(ge=0)
    equivalent_diameter_nm: float | None = Field(default=None, ge=0)
    circularity: float | None = Field(default=None, ge=0, le=1)
    bbox: tuple[int, int, int, int]
    confidence: float | None = Field(default=None, ge=0, le=1)


class ImageSummaryDTO(ContractModel):
    run_id: str
    particle_count: int = Field(ge=0)
    roi_area_px: int = Field(ge=0)
    number_density_px2: float = Field(ge=0)
    number_density_um2: float | None = Field(default=None, ge=0)
    mean_equivalent_diameter_px: float | None = Field(default=None, ge=0)
    mean_equivalent_diameter_nm: float | None = Field(default=None, ge=0)
    coverage_ratio: float = Field(ge=0, le=1)
    perimeter_density_px: float = Field(ge=0)
    perimeter_density_um: float | None = Field(default=None, ge=0)
    quality_status: QualityStatus


class QualityReportDTO(ContractModel):
    status: QualityStatus
    reasons: list[str] = Field(default_factory=list)
    metrics: dict[str, float | int | str | None] = Field(default_factory=dict)
    recommendations: list[str] = Field(default_factory=list)


class RunStatusEventDTO(ContractModel):
    event_id: int = Field(ge=1)
    from_status: JobStatus | None = None
    to_status: JobStatus
    error_code: str | None = None
    error_message: str | None = None
    created_at: datetime


class SegmentationRunDTO(ContractModel):
    run_id: str
    job_id: str
    image_id: str
    model_id: str
    status: JobStatus
    roi_mode: RoiMode
    box_revision: int | None = None
    threshold: float | None = None
    inference: InferenceOptions
    configuration: RunConfiguration
    parent_run_id: str | None = None
    artifacts: RunArtifacts = Field(default_factory=RunArtifacts)
    summary: ImageSummaryDTO | None = None
    quality: QualityReportDTO | None = None
    execution: ExecutionRuntimeProvenance | None = None
    runtime_ms: int | None = Field(default=None, ge=0)
    error_code: str | None = None
    error_message: str | None = None
    status_history: list[RunStatusEventDTO] = Field(default_factory=list)
    created_at: datetime
    updated_at: datetime


class JobDetailDTO(ContractModel):
    job: AnalysisJobDTO
    images: list[ImageAssetDTO] = Field(default_factory=list)
    runs: list[SegmentationRunDTO] = Field(default_factory=list)
    partial_failures: list["RunFailureDTO"] = Field(default_factory=list)


class RunFailureDTO(ContractModel):
    run_id: str
    image_id: str
    model_id: str
    error_code: str
    error_message: str | None = None


class ReviewRunRequest(ContractModel):
    threshold: float | None = Field(default=None, ge=0, le=1)
    min_area_px: int | None = Field(default=None, ge=0)
    watershed_enabled: bool | None = None
    exclude_border: bool | None = None
    corrected_mask_token: str | None = None

    @model_validator(mode="after")
    def validate_has_change(self) -> "ReviewRunRequest":
        if all(value is None for value in self.__dict__.values()):
            raise ValueError(
                "review must change at least one parameter or provide a corrected mask"
            )
        return self


class ReviewRunData(ContractModel):
    parent_run_id: str
    run_id: str


class CorrectedMaskUploadData(ContractModel):
    corrected_mask_token: str
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    width: int = Field(gt=0)
    height: int = Field(gt=0)


class ExportData(ContractModel):
    job_id: str
    download_url: str
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    filename: str
