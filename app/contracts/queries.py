"""Evidence-preserving data, knowledge, and mixed-query contracts."""

from datetime import datetime
from typing import Any, Literal

from pydantic import Field

from app.contracts.common import ContractModel
from app.contracts.enums import QueryType
from app.contracts.limits import MAX_MATERIAL_ALIASES, MaterialAlias


class MaterialContext(ContractModel):
    formula: str | None = Field(default=None, max_length=255)
    name: str | None = Field(default=None, max_length=255)
    aliases: list[MaterialAlias] = Field(
        default_factory=list,
        max_length=MAX_MATERIAL_ALIASES,
    )
    source: Literal["request", "image_metadata", "user_confirmation"] = "request"


class Citation(ContractModel):
    citation_id: str
    doc_id: str
    title: str
    page: int | None = Field(default=None, ge=1)
    chunk_id: str
    excerpt: str = Field(max_length=160)
    retrieval_score: float = Field(ge=0)
    source_type: str | None = None
    citation_text: str | None = None


class ToolEvidence(ContractModel):
    tool_name: str
    validated_arguments: dict[str, Any]
    rows: list[dict[str, Any]] = Field(default_factory=list)
    aggregates: dict[str, Any] = Field(default_factory=dict)
    units: dict[str, str] = Field(default_factory=dict)
    source_run_ids: list[str] = Field(default_factory=list)
    quality_warnings: list[str] = Field(default_factory=list)
    chart_url: str | None = None


class ToolCallLog(ContractModel):
    tool_name: str
    arguments: dict[str, Any]
    outcome: Literal["success", "insufficient_data", "error"]
    source_run_ids: list[str] = Field(default_factory=list)


class UnifiedQueryRequest(ContractModel):
    question: str = Field(min_length=1, max_length=2000)
    query_type: QueryType = QueryType.AUTO
    image_id: str | None = None
    run_ids: list[str] = Field(default_factory=list, max_length=50)
    material_context: MaterialContext | None = None


class UnifiedQueryResponse(ContractModel):
    query_type: QueryType
    answer: str
    data_evidence: list[ToolEvidence] = Field(default_factory=list)
    citations: list[Citation] = Field(default_factory=list)
    tool_calls: list[ToolCallLog] = Field(default_factory=list)
    material_context: MaterialContext | None = None
    confidence: Literal["low", "medium", "high"]
    limitations: list[str] = Field(default_factory=list)
    needs_clarification: bool = False
    outcome_code: Literal["OK", "INSUFFICIENT_EVIDENCE"] = "OK"


class QueryAuditRecordDTO(ContractModel):
    query_id: str
    job_id: str
    image_id: str | None = None
    request: UnifiedQueryRequest
    response: UnifiedQueryResponse
    created_at: datetime


class GetMetricArgs(ContractModel):
    scope_type: Literal["job", "image", "run", "sample", "material"]
    scope_id: str
    metric: str
    aggregation: Literal["value", "mean", "median", "min", "max", "sum", "count"] = "value"


class RankSamplesArgs(ContractModel):
    job_id: str
    metric: str
    group_by: Literal["image", "sample", "material"] = "image"
    order: Literal["asc", "desc"] = "desc"
    top_k: int = Field(default=10, ge=1, le=100)
    filters: dict[str, Any] = Field(default_factory=dict)


class CompareGroupsArgs(ContractModel):
    job_id: str
    metric: str
    group_by: Literal["sample", "material"]
    groups: list[str] = Field(min_length=2, max_length=20)
    statistic: Literal["mean", "median", "min", "max"] = "mean"


class DescribeDistributionArgs(ContractModel):
    job_id: str
    metric: str
    bins: int = Field(default=20, ge=2, le=200)
    filters: dict[str, Any] = Field(default_factory=dict)


class FindReviewArgs(ContractModel):
    job_id: str
    reasons: list[str] = Field(default_factory=list)


class CompareModelsArgs(ContractModel):
    image_id: str
    run_ids: list[str] = Field(min_length=2, max_length=3)
    metric: str
