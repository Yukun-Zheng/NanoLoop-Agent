"""Structured, previewable scientific-report contracts."""

from datetime import datetime
from typing import Literal

from pydantic import Field, model_validator

from app.contracts.common import ContractModel
from app.contracts.enums import QualityStatus
from app.contracts.queries import Citation, ToolEvidence


class ScientificReportRequest(ContractModel):
    """Select the completed runs that define one report snapshot."""

    run_ids: list[str] = Field(min_length=1, max_length=20)
    require_qwen: bool = True

    @model_validator(mode="after")
    def validate_unique_runs(self) -> "ScientificReportRequest":
        if len(set(self.run_ids)) != len(self.run_ids):
            raise ValueError("run_ids must be unique")
        return self


class ReportArtifactData(ContractModel):
    filename: str
    download_url: str
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    size_bytes: int = Field(ge=1)
    media_type: str


class ReportMetricDTO(ContractModel):
    key: str
    label: str
    display_value: str
    unit: str | None = None
    definition: str
    source_run_ids: list[str] = Field(default_factory=list)


class ReportFindingDTO(ContractModel):
    title: str
    interpretation: str
    severity: Literal["info", "caution", "review"]
    evidence_ids: list[str] = Field(default_factory=list)
    source_run_ids: list[str] = Field(default_factory=list)


class ReportRecommendationDTO(ContractModel):
    priority: int = Field(ge=1)
    action: str
    rationale: str
    verification: str
    source_run_ids: list[str] = Field(default_factory=list)


class ReportRunSummaryDTO(ContractModel):
    run_id: str
    image_id: str
    filename: str | None = None
    sample_id: str | None = None
    model_id: str
    quality_status: QualityStatus
    scale_nm_per_pixel: float | None = Field(default=None, gt=0)
    particle_count: int = Field(ge=0)
    roi_area: str
    number_density: str
    mean_equivalent_diameter: str
    coverage: str
    perimeter_density: str
    quality_reasons: list[str] = Field(default_factory=list)


class BatchMetricDistributionDTO(ContractModel):
    key: str
    label: str
    unit: str
    sample_count: int = Field(ge=1)
    mean: float
    std_dev: float = Field(ge=0)
    minimum: float
    q1: float
    median: float
    q3: float
    maximum: float
    coefficient_of_variation: float | None = Field(default=None, ge=0)


class BatchOutlierDTO(ContractModel):
    run_id: str
    image_id: str
    metric_key: str
    metric_label: str
    value: float
    unit: str
    direction: Literal["low", "high"]


class BatchResultSummaryDTO(ContractModel):
    image_count: int = Field(ge=2)
    run_count: int = Field(ge=2)
    model_count: int = Field(ge=1)
    total_particle_count: int = Field(ge=0)
    quality_pass_count: int = Field(ge=0)
    quality_warning_count: int = Field(ge=0)
    quality_review_count: int = Field(ge=0)
    distributions: list[BatchMetricDistributionDTO]
    outliers: list[BatchOutlierDTO] = Field(default_factory=list)


class ReportProvenanceDTO(ContractModel):
    run_id: str
    model_id: str
    model_version: str
    image_sha256: str | None = None
    model_bundle_sha256: str | None = None
    scale_nm_per_pixel: float | None = Field(default=None, gt=0)
    roi_mode: str
    box_revision: int | None = Field(default=None, ge=0)
    threshold: float | None = Field(default=None, ge=0, le=1)
    min_area_px: int = Field(ge=0)


class ScientificReportPreviewDTO(ContractModel):
    report_id: str
    job_id: str
    title: str
    generated_at: datetime
    selected_run_ids: list[str]
    analysis_mode: Literal["single_image", "batch"]
    quality_status: QualityStatus
    scale_status: Literal["physical", "mixed", "pixel_only"]
    technical_summary: str
    synthesis_provider: Literal["local_llm", "deterministic_fallback"]
    synthesis_model: str | None = None
    fallback_used: bool
    headline_metrics: list[ReportMetricDTO]
    findings: list[ReportFindingDTO]
    run_summaries: list[ReportRunSummaryDTO]
    batch_summary: BatchResultSummaryDTO | None = None
    recommendations: list[ReportRecommendationDTO]
    methodology: list[str]
    limitations: list[str]
    further_questions: list[str]
    data_evidence: list[ToolEvidence]
    knowledge_citations: list[Citation]
    provenance: list[ReportProvenanceDTO]


class ScientificReportData(ScientificReportPreviewDTO):
    docx: ReportArtifactData
    pdf: ReportArtifactData
