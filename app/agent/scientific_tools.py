"""Bounded adapters from agent actions to NanoLoop's deterministic services."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Literal

from pydantic import BaseModel, Field, ValidationError
from sqlalchemy import select
from sqlalchemy.orm import Session, selectinload

from app.agent.data_tools import SqlAlchemyDataToolService
from app.agent.protocols import AgentToolContext
from app.agent.tool_registry import AgentToolRegistry, RegisteredAgentTool
from app.agent.unified_query import DataQuery
from app.analysis.application import AnalysisApplicationService
from app.analysis.authorization import require_read
from app.analysis.reporting import JobExportSnapshot, ReportWriter
from app.analysis.scientific_reports import ScientificReportBuilder
from app.contracts.agent_runtime import (
    AgentToolObservation,
    AgentToolOutcome,
    AgentToolRisk,
    AgentToolSpec,
)
from app.contracts.analyses import CreateRunsRequest, ReviewRunRequest
from app.contracts.common import ContractModel
from app.contracts.enums import JobStatus, ModelVariant, RoiMode
from app.contracts.file_artifacts import FileArtifactKind
from app.contracts.models import ModelMetadata, ModelRecommendationRequest
from app.core.errors import ExportNotReadyError, ResourceNotFoundError
from app.db.models import AnalysisJob, SegmentationRun
from app.db.repositories import SqlAlchemyRepositorySet
from app.files import FileArtifactAccessService
from app.rag.providers import OpenAICompatibleProvider
from app.storage import LocalFileStore

SessionFactory = Callable[[], Session]
_TERMINAL_RUN_STATUSES = {
    JobStatus.COMPLETED.value,
    JobStatus.COMPLETED_WITH_WARNINGS.value,
    JobStatus.FAILED.value,
}


def _spec(
    name: str,
    description: str,
    arguments_model: type[BaseModel],
    *,
    risk: AgentToolRisk,
    requires_approval: bool,
    idempotent: bool,
) -> AgentToolSpec:
    return AgentToolSpec(
        name=name,
        description=description,
        input_schema=arguments_model.model_json_schema(),
        risk=risk,
        requires_approval=requires_approval,
        idempotent=idempotent,
    )


class InspectJobArguments(ContractModel):
    include_models: bool = True
    include_runs: bool = True


class InspectRunsArguments(ContractModel):
    run_ids: list[str] = Field(default_factory=list, max_length=20)


class RecommendModelsArguments(ContractModel):
    image_id: str = Field(min_length=1, max_length=64)
    roi_mode: RoiMode = RoiMode.FULL_IMAGE
    target_profile: ModelVariant = ModelVariant.GENERAL
    prefer: Literal["speed", "balance", "accuracy"] = "balance"


class QueryResultsArguments(ContractModel):
    question: str = Field(min_length=1, max_length=2000)
    image_id: str | None = Field(default=None, max_length=64)
    run_ids: list[str] = Field(default_factory=list, max_length=20)


class CreateRunsArguments(CreateRunsRequest):
    pass


class CreateReviewRunArguments(ReviewRunRequest):
    parent_run_id: str = Field(min_length=1, max_length=64)


class GenerateScientificReportArguments(ContractModel):
    run_ids: list[str] = Field(min_length=1, max_length=20)
    require_report_model: bool = False


class ExportReproducibilityBundleArguments(ContractModel):
    run_ids: list[str] = Field(min_length=1, max_length=20)


_SCIENTIFIC_ARGUMENT_MODELS: dict[str, type[BaseModel]] = {
    "inspect_job": InspectJobArguments,
    "inspect_runs": InspectRunsArguments,
    "recommend_models": RecommendModelsArguments,
    "query_results": QueryResultsArguments,
    "create_analysis_runs": CreateRunsArguments,
    "create_review_run": CreateReviewRunArguments,
    "generate_scientific_report": GenerateScientificReportArguments,
    "export_reproducibility_bundle": ExportReproducibilityBundleArguments,
}

_SCIENTIFIC_TOOL_SPECS = {
    spec.name: spec
    for spec in (
        _spec(
            "inspect_job",
            "检查当前任务的图像、比例尺、ROI、运行状态和可用本地分割模型。",
            InspectJobArguments,
            risk=AgentToolRisk.READ_ONLY,
            requires_approval=False,
            idempotent=True,
        ),
        _spec(
            "inspect_runs",
            "读取指定运行或当前任务最近运行的进度、质量状态、失败原因和核心统计。",
            InspectRunsArguments,
            risk=AgentToolRisk.READ_ONLY,
            requires_approval=False,
            idempotent=True,
        ),
        _spec(
            "recommend_models",
            "依据图像、ROI 模式和目标画像返回候选模型；结果仍需用户确认后才能创建运行。",
            RecommendModelsArguments,
            risk=AgentToolRisk.READ_ONLY,
            requires_approval=False,
            idempotent=True,
        ),
        _spec(
            "query_results",
            "用确定性数据工具查询颗粒计数、粒径、覆盖率、密度、异常、分布或模型比较。",
            QueryResultsArguments,
            risk=AgentToolRisk.READ_ONLY,
            requires_approval=False,
            idempotent=True,
        ),
        _spec(
            "create_analysis_runs",
            "创建不可变的图像分割运行。该写操作会消耗计算资源，执行前必须人工批准。",
            CreateRunsArguments,
            risk=AgentToolRisk.CONTROLLED_WRITE,
            requires_approval=True,
            idempotent=False,
        ),
        _spec(
            "create_review_run",
            "基于已有运行调整阈值、后处理、比例尺或人工掩膜并创建复核子运行；必须人工批准。",
            CreateReviewRunArguments,
            risk=AgentToolRisk.CONTROLLED_WRITE,
            requires_approval=True,
            idempotent=False,
        ),
        _spec(
            "generate_scientific_report",
            "从已完成运行生成可复核的 DOCX/PDF 科研报告；写入制品前必须人工批准。",
            GenerateScientificReportArguments,
            risk=AgentToolRisk.CONTROLLED_WRITE,
            requires_approval=True,
            idempotent=False,
        ),
        _spec(
            "export_reproducibility_bundle",
            "把指定完成运行、输入、配置、统计、质量和审计制品打成内容寻址 ZIP；必须人工批准。",
            ExportReproducibilityBundleArguments,
            risk=AgentToolRisk.CONTROLLED_WRITE,
            requires_approval=True,
            idempotent=False,
        ),
    )
}


def scientific_tool_specs() -> list[AgentToolSpec]:
    """Return the exact production tool contracts for prompts and evaluations."""

    return [
        spec.model_copy(deep=True)
        for spec in _SCIENTIFIC_TOOL_SPECS.values()
    ]


def scientific_tool_arguments_are_valid(
    tool_name: str,
    arguments: dict[str, Any],
) -> bool:
    """Validate a proposed action with the same Pydantic model used at runtime."""

    model = _SCIENTIFIC_ARGUMENT_MODELS.get(tool_name)
    if model is None:
        return False
    try:
        model.model_validate(arguments)
    except ValidationError:
        return False
    return True


class InspectJobTool:
    def __init__(self, session_factory: SessionFactory, inference_gateway: Any) -> None:
        self._session_factory = session_factory
        self._inference_gateway = inference_gateway

    @property
    def spec(self) -> AgentToolSpec:
        return _SCIENTIFIC_TOOL_SPECS["inspect_job"]

    def execute(
        self,
        context: AgentToolContext,
        arguments: dict[str, Any],
    ) -> AgentToolObservation:
        parsed = InspectJobArguments.model_validate(arguments)
        tenant_id = _tenant_id(context)
        session = self._session_factory()
        try:
            scope = SqlAlchemyRepositorySet(session).jobs.get_scope(
                context.job_id,
                tenant_id=tenant_id,
            )
            require_read(context.principal, scope)
            job = session.scalar(
                select(AnalysisJob)
                .where(
                    AnalysisJob.job_id == context.job_id,
                    AnalysisJob.tenant_id == tenant_id,
                )
                .options(
                    selectinload(AnalysisJob.images),
                    selectinload(AnalysisJob.runs).selectinload(SegmentationRun.summary),
                )
            )
            if job is None:
                raise LookupError("analysis job is not available")
            images = [
                {
                    "image_id": image.image_id,
                    "filename": image.filename,
                    "sample_id": image.sample_id,
                    "material_name": image.material_name,
                    "material_formula": image.material_formula,
                    "width": image.width,
                    "height": image.height,
                    "scale_nm_per_pixel": image.scale_nm_per_pixel,
                    "scale_source": image.scale_source,
                    "box_revision": image.box_revision,
                    "analysis_roi": image.analysis_roi_json,
                }
                for image in job.images[:20]
            ]
            runs = [_compact_run(run) for run in job.runs[-50:]] if parsed.include_runs else []
        finally:
            session.close()

        models: list[dict[str, Any]] = []
        if parsed.include_models:
            raw_models = self._inference_gateway.list_models(only_ready=False)
            models = [
                ModelMetadata.model_validate(item).model_dump(
                    mode="json",
                    include={
                        "model_id",
                        "family",
                        "variant",
                        "quality_tier",
                        "status",
                        "supports_box_prompt",
                        "applicable_materials",
                        "notes",
                        "health_error",
                    },
                )
                for item in raw_models[:30]
            ]
        active_run_ids = [
            run["run_id"] for run in runs if run["status"] not in _TERMINAL_RUN_STATUSES
        ]
        active_count = len(active_run_ids)
        return AgentToolObservation(
            outcome=AgentToolOutcome.OK,
            summary=(
                f"任务包含 {len(images)} 张图像、{len(runs)} 个已返回运行记录、"
                f"{len(models)} 个模型；活动运行 {active_count} 个。"
            ),
            data={
                "job": {
                    "job_id": job.job_id,
                    "name": job.name,
                    "status": job.status,
                },
                "images": images,
                "runs": runs,
                "models": models,
            },
            evidence_refs=[
                *[image["image_id"] for image in images],
                *[run["run_id"] for run in runs],
            ],
            suggested_poll_after_seconds=2 if active_count else None,
            continuation_tool="inspect_runs" if active_count else None,
            continuation_arguments={"run_ids": active_run_ids} if active_count else {},
        )


class InspectRunsTool:
    def __init__(self, session_factory: SessionFactory) -> None:
        self._session_factory = session_factory

    @property
    def spec(self) -> AgentToolSpec:
        return _SCIENTIFIC_TOOL_SPECS["inspect_runs"]

    def execute(
        self,
        context: AgentToolContext,
        arguments: dict[str, Any],
    ) -> AgentToolObservation:
        parsed = InspectRunsArguments.model_validate(arguments)
        tenant_id = _tenant_id(context)
        session = self._session_factory()
        try:
            scope = SqlAlchemyRepositorySet(session).jobs.get_scope(
                context.job_id,
                tenant_id=tenant_id,
            )
            require_read(context.principal, scope)
            statement = (
                select(SegmentationRun)
                .join(AnalysisJob, AnalysisJob.job_id == SegmentationRun.job_id)
                .where(
                    SegmentationRun.job_id == context.job_id,
                    AnalysisJob.tenant_id == tenant_id,
                )
                .options(selectinload(SegmentationRun.summary))
                .order_by(SegmentationRun.created_at.desc())
                .limit(20)
            )
            if parsed.run_ids:
                statement = statement.where(SegmentationRun.run_id.in_(parsed.run_ids))
            runs = list(session.scalars(statement))
            compact = [_compact_run(run) for run in runs]
        finally:
            session.close()
        active_count = sum(run["status"] not in _TERMINAL_RUN_STATUSES for run in compact)
        failed_count = sum(run["status"] == JobStatus.FAILED.value for run in compact)
        return AgentToolObservation(
            outcome=AgentToolOutcome.OK,
            summary=(
                f"读取 {len(compact)} 个运行：活动 {active_count} 个，失败 {failed_count} 个。"
            ),
            data={"runs": compact},
            evidence_refs=[run["run_id"] for run in compact],
            suggested_poll_after_seconds=2 if active_count else None,
            continuation_tool="inspect_runs" if active_count else None,
            continuation_arguments={
                "run_ids": [
                    run["run_id"]
                    for run in compact
                    if run["status"] not in _TERMINAL_RUN_STATUSES
                ]
            }
            if active_count
            else {},
        )


class RecommendModelsTool:
    def __init__(self, inference_gateway: Any) -> None:
        self._inference_gateway = inference_gateway

    @property
    def spec(self) -> AgentToolSpec:
        return _SCIENTIFIC_TOOL_SPECS["recommend_models"]

    def execute(
        self,
        context: AgentToolContext,
        arguments: dict[str, Any],
    ) -> AgentToolObservation:
        parsed = RecommendModelsArguments.model_validate(arguments)
        request = ModelRecommendationRequest(
            job_id=context.job_id,
            image_id=parsed.image_id,
            roi_mode=parsed.roi_mode,
            target_profile=parsed.target_profile,
            prefer=parsed.prefer,
        )
        candidates = self._inference_gateway.recommend(request)
        normalized = [
            item.model_dump(mode="json") if hasattr(item, "model_dump") else dict(item)
            for item in candidates
        ]
        return AgentToolObservation(
            outcome=AgentToolOutcome.OK,
            summary=f"得到 {len(normalized)} 个候选模型，创建运行前仍需人工批准。",
            data={"candidates": normalized, "requires_user_confirmation": True},
            evidence_refs=[str(item.get("model_id")) for item in normalized],
        )


class QueryResultsTool:
    def __init__(self, data_tools: SqlAlchemyDataToolService) -> None:
        self._data_tools = data_tools

    @property
    def spec(self) -> AgentToolSpec:
        return _SCIENTIFIC_TOOL_SPECS["query_results"]

    def execute(
        self,
        context: AgentToolContext,
        arguments: dict[str, Any],
    ) -> AgentToolObservation:
        parsed = QueryResultsArguments.model_validate(arguments)
        result = self._data_tools.answer(
            DataQuery(
                job_id=context.job_id,
                tenant_id=_tenant_id(context),
                question=parsed.question,
                image_id=parsed.image_id,
                run_ids=tuple(parsed.run_ids),
            )
        )
        outcome = (
            AgentToolOutcome.OK
            if result.evidence and not result.needs_clarification
            else AgentToolOutcome.ERROR
        )
        return AgentToolObservation(
            outcome=outcome,
            summary=result.answer or "数据工具没有返回可用结论。",
            data={
                "evidence": [item.model_dump(mode="json") for item in result.evidence],
                "limitations": list(result.limitations),
                "needs_clarification": result.needs_clarification,
            },
            retryable=result.needs_clarification,
            evidence_refs=list(
                dict.fromkeys(
                    run_id
                    for item in result.evidence
                    for run_id in item.source_run_ids
                )
            ),
        )


class CreateRunsTool:
    def __init__(self, analysis_service: AnalysisApplicationService) -> None:
        self._analysis_service = analysis_service

    @property
    def spec(self) -> AgentToolSpec:
        return _SCIENTIFIC_TOOL_SPECS["create_analysis_runs"]

    def execute(
        self,
        context: AgentToolContext,
        arguments: dict[str, Any],
    ) -> AgentToolObservation:
        request = CreateRunsRequest.model_validate(arguments)
        run_ids = self._analysis_service.create_runs(
            context.job_id,
            request,
            principal=context.principal,
        )
        return AgentToolObservation(
            outcome=AgentToolOutcome.OK,
            summary=f"已创建 {len(run_ids)} 个不可变分析运行。",
            data={"run_ids": run_ids},
            evidence_refs=run_ids,
            suggested_poll_after_seconds=2,
            continuation_tool="inspect_runs",
            continuation_arguments={"run_ids": run_ids},
        )


class CreateReviewRunTool:
    def __init__(self, analysis_service: AnalysisApplicationService) -> None:
        self._analysis_service = analysis_service

    @property
    def spec(self) -> AgentToolSpec:
        return _SCIENTIFIC_TOOL_SPECS["create_review_run"]

    def execute(
        self,
        context: AgentToolContext,
        arguments: dict[str, Any],
    ) -> AgentToolObservation:
        parsed = CreateReviewRunArguments.model_validate(arguments)
        request = ReviewRunRequest.model_validate(
            parsed.model_dump(mode="python", exclude={"parent_run_id"})
        )
        run_id = self._analysis_service.create_review_run(
            parsed.parent_run_id,
            request,
            principal=context.principal,
        )
        return AgentToolObservation(
            outcome=AgentToolOutcome.OK,
            summary="已创建一个不可变复核子运行。",
            data={"parent_run_id": parsed.parent_run_id, "run_id": run_id},
            evidence_refs=[parsed.parent_run_id, run_id],
            suggested_poll_after_seconds=2,
            continuation_tool="inspect_runs",
            continuation_arguments={"run_ids": [run_id]},
        )


class GenerateScientificReportTool:
    def __init__(
        self,
        *,
        session_factory: SessionFactory,
        file_store: LocalFileStore,
        file_access: FileArtifactAccessService,
        api_prefix: str,
        data_tools: SqlAlchemyDataToolService,
        llm_provider: OpenAICompatibleProvider | None,
    ) -> None:
        self._session_factory = session_factory
        self._file_access = file_access
        self._api_prefix = api_prefix.rstrip("/")
        self._builder = ScientificReportBuilder(
            file_store=file_store,
            data_tools=data_tools,
            llm_provider=llm_provider,
        )

    @property
    def spec(self) -> AgentToolSpec:
        return _SCIENTIFIC_TOOL_SPECS["generate_scientific_report"]

    def execute(
        self,
        context: AgentToolContext,
        arguments: dict[str, Any],
    ) -> AgentToolObservation:
        parsed = GenerateScientificReportArguments.model_validate(arguments)
        snapshot = _load_export_snapshot(
            self._session_factory,
            context,
            parsed.run_ids,
            include_queries=False,
        )
        preview, docx_file, pdf_file = self._builder.build(
            snapshot=snapshot,
            tenant_id=_tenant_id(context),
            run_ids=parsed.run_ids,
            require_qwen=parsed.require_report_model,
        )
        docx_token = self._file_access.issue_download_token(
            principal=context.principal,
            job_id=context.job_id,
            artifact_kind=FileArtifactKind.ANALYSIS_EXPORT,
            storage_path=docx_file.relative_path,
            filename=docx_file.filename,
            media_type=(
                "application/vnd.openxmlformats-officedocument."
                "wordprocessingml.document"
            ),
            expected_sha256=docx_file.sha256,
            expected_size_bytes=docx_file.size_bytes,
        )
        pdf_token = self._file_access.issue_download_token(
            principal=context.principal,
            job_id=context.job_id,
            artifact_kind=FileArtifactKind.ANALYSIS_EXPORT,
            storage_path=pdf_file.relative_path,
            filename=pdf_file.filename,
            media_type="application/pdf",
            expected_sha256=pdf_file.sha256,
            expected_size_bytes=pdf_file.size_bytes,
        )
        return AgentToolObservation(
            outcome=AgentToolOutcome.OK,
            summary=f"已基于 {len(parsed.run_ids)} 个完成运行生成 DOCX/PDF 科研报告。",
            data={
                "report_id": preview.report_id,
                "run_ids": parsed.run_ids,
                "synthesis_provider": preview.synthesis_provider,
                "synthesis_model": preview.synthesis_model,
                "fallback_used": preview.fallback_used,
                "docx": {
                    "filename": docx_file.filename,
                    "download_url": f"{self._api_prefix}/files/{docx_token}",
                    "sha256": docx_file.sha256,
                    "size_bytes": docx_file.size_bytes,
                },
                "pdf": {
                    "filename": pdf_file.filename,
                    "download_url": f"{self._api_prefix}/files/{pdf_token}",
                    "sha256": pdf_file.sha256,
                    "size_bytes": pdf_file.size_bytes,
                },
            },
            evidence_refs=[*parsed.run_ids, preview.report_id],
        )


class ExportReproducibilityBundleTool:
    def __init__(
        self,
        *,
        session_factory: SessionFactory,
        file_store: LocalFileStore,
        file_access: FileArtifactAccessService,
        api_prefix: str,
    ) -> None:
        self._session_factory = session_factory
        self._writer = ReportWriter(file_store)
        self._file_access = file_access
        self._api_prefix = api_prefix.rstrip("/")

    @property
    def spec(self) -> AgentToolSpec:
        return _SCIENTIFIC_TOOL_SPECS["export_reproducibility_bundle"]

    def execute(
        self,
        context: AgentToolContext,
        arguments: dict[str, Any],
    ) -> AgentToolObservation:
        parsed = ExportReproducibilityBundleArguments.model_validate(arguments)
        snapshot = _load_export_snapshot(
            self._session_factory,
            context,
            parsed.run_ids,
            include_queries=True,
        )
        exported = self._writer.build_job_export(
            context.job_id,
            run_ids=set(parsed.run_ids),
            snapshot=snapshot,
        )
        token = self._file_access.issue_download_token(
            principal=context.principal,
            job_id=context.job_id,
            artifact_kind=FileArtifactKind.ANALYSIS_EXPORT,
            storage_path=exported.relative_path,
            filename=exported.filename,
            media_type="application/zip",
            expected_sha256=exported.sha256,
            expected_size_bytes=exported.size_bytes,
        )
        return AgentToolObservation(
            outcome=AgentToolOutcome.OK,
            summary=f"已把 {len(parsed.run_ids)} 个完成运行导出为复现 ZIP。",
            data={
                "run_ids": parsed.run_ids,
                "filename": exported.filename,
                "download_url": f"{self._api_prefix}/files/{token}",
                "sha256": exported.sha256,
                "size_bytes": exported.size_bytes,
            },
            evidence_refs=[*parsed.run_ids, exported.sha256],
        )


def build_scientific_tool_registry(
    *,
    session_factory: SessionFactory,
    inference_gateway: Any,
    analysis_service: AnalysisApplicationService,
    data_tools: SqlAlchemyDataToolService,
    file_store: LocalFileStore,
    file_access: FileArtifactAccessService,
    api_prefix: str,
    report_llm_provider: OpenAICompatibleProvider | None = None,
) -> AgentToolRegistry:
    tools = [
        _registered(InspectJobTool(session_factory, inference_gateway), InspectJobArguments),
        _registered(InspectRunsTool(session_factory), InspectRunsArguments),
        _registered(RecommendModelsTool(inference_gateway), RecommendModelsArguments),
        _registered(QueryResultsTool(data_tools), QueryResultsArguments),
        _registered(CreateRunsTool(analysis_service), CreateRunsArguments),
        _registered(CreateReviewRunTool(analysis_service), CreateReviewRunArguments),
        _registered(
            GenerateScientificReportTool(
                session_factory=session_factory,
                file_store=file_store,
                file_access=file_access,
                api_prefix=api_prefix,
                data_tools=data_tools,
                llm_provider=report_llm_provider,
            ),
            GenerateScientificReportArguments,
        ),
        _registered(
            ExportReproducibilityBundleTool(
                session_factory=session_factory,
                file_store=file_store,
                file_access=file_access,
                api_prefix=api_prefix,
            ),
            ExportReproducibilityBundleArguments,
        ),
    ]
    return AgentToolRegistry(tools)


def _registered(tool: Any, arguments_model: type[BaseModel]) -> RegisteredAgentTool:
    return RegisteredAgentTool(tool=tool, arguments_model=arguments_model)


def _compact_run(run: SegmentationRun) -> dict[str, Any]:
    summary = run.summary
    return {
        "run_id": run.run_id,
        "image_id": run.image_id,
        "model_id": run.model_id,
        "status": run.status,
        "roi_mode": run.roi_mode,
        "threshold": run.threshold,
        "parent_run_id": run.parent_run_id,
        "runtime_ms": run.runtime_ms,
        "error_code": run.error_code,
        "error_message": run.error_message,
        "summary": (
            {
                "particle_count": summary.particle_count,
                "mean_equivalent_diameter_px": summary.mean_equivalent_diameter_px,
                "mean_equivalent_diameter_nm": summary.mean_equivalent_diameter_nm,
                "coverage_ratio": summary.coverage_ratio,
                "quality_status": summary.quality_status,
                "quality": summary.quality_json,
            }
            if summary is not None
            else None
        ),
    }


def _tenant_id(context: AgentToolContext) -> str:
    tenant_id = context.principal.tenant_id
    if tenant_id is None:
        raise ValueError("agent tool context requires a tenant ID")
    return tenant_id


def _load_export_snapshot(
    session_factory: SessionFactory,
    context: AgentToolContext,
    run_ids: list[str],
    *,
    include_queries: bool,
) -> JobExportSnapshot:
    if len(set(run_ids)) != len(run_ids):
        raise ValueError("run_ids must be unique")
    tenant_id = _tenant_id(context)
    session = session_factory()
    try:
        repositories = SqlAlchemyRepositorySet(session)
        scope = repositories.jobs.get_scope(context.job_id, tenant_id=tenant_id)
        require_read(context.principal, scope)
        images = repositories.images.list_by_job_scoped(
            context.job_id,
            tenant_id=tenant_id,
        )
        box_revisions = repositories.boxes.list_by_job_scoped(
            context.job_id,
            tenant_id=tenant_id,
        )
        all_runs = {
            run.run_id: run
            for run in repositories.runs.list_by_job_scoped(
                context.job_id,
                tenant_id=tenant_id,
            )
        }
        missing = [run_id for run_id in run_ids if run_id not in all_runs]
        if missing:
            raise ResourceNotFoundError(
                details={
                    "resource": "run",
                    "job_id": context.job_id,
                    "run_ids": missing,
                }
            )
        terminal = {JobStatus.COMPLETED, JobStatus.COMPLETED_WITH_WARNINGS}
        not_ready = [run_id for run_id in run_ids if all_runs[run_id].status not in terminal]
        if not_ready:
            raise ExportNotReadyError(
                details={
                    "job_id": context.job_id,
                    "run_ids": not_ready,
                    "reason": "runs_not_complete",
                }
            )
        selected = tuple(all_runs[run_id] for run_id in run_ids)
        if any(run.summary is None or run.quality is None for run in selected):
            raise ExportNotReadyError(
                details={
                    "job_id": context.job_id,
                    "run_ids": run_ids,
                    "reason": "summary_or_quality_missing",
                }
            )
        queries = (
            tuple(
                repositories.queries.list_by_job_scoped(
                    context.job_id,
                    tenant_id=tenant_id,
                )
            )
            if include_queries
            else ()
        )
        return JobExportSnapshot(
            job=scope.job,
            images=tuple(images),
            runs=selected,
            queries=queries,
            box_revisions=tuple(box_revisions),
            image_storage_paths=tuple(
                sorted(
                    repositories.images.get_storage_path(image.image_id)
                    for image in images
                )
            ),
            run_artifact_paths=tuple(
                (run_id, artifact_path)
                for run_id in run_ids
                for artifact_path in sorted(
                    path
                    for path in repositories.runs.get_artifact_paths(run_id).values()
                    if path is not None
                )
            ),
        )
    finally:
        session.close()
