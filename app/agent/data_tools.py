"""Deterministic, read-only experiment-data tools for unified queries.

This module deliberately exposes a small intent allowlist. It never accepts SQL or delegates
arithmetic to an LLM: job ownership, filters, aggregates, units, and quality warnings are all
resolved from persisted SQLAlchemy records here.
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from enum import StrEnum
from math import ceil, isclose
from statistics import fmean, median
from typing import Any

from sqlalchemy import Integer, SQLColumnExpression, case, cast, func, select
from sqlalchemy.orm import Session, joinedload

from app.agent.unified_query import DataQuery, DataQueryResult
from app.contracts.enums import JobStatus, QualityStatus
from app.contracts.queries import ToolCallLog, ToolEvidence
from app.db.models import AnalysisJob, ImageAsset, ParticleRecord, SegmentationRun

SessionFactory = Callable[[], Session]

_COMPLETED_STATUSES = {
    JobStatus.COMPLETED.value,
    JobStatus.COMPLETED_WITH_WARNINGS.value,
}
_RANK_SIGNALS = (
    "最高",
    "最低",
    "最大",
    "最小",
    "排名",
    "排序",
    "哪组",
    "哪张",
    "top",
    "highest",
    "lowest",
    "rank",
)
DEFAULT_DISTRIBUTION_EVIDENCE_LIMIT = 200


class _IntentKind(StrEnum):
    OVERVIEW = "overview"
    PARTICLE_COUNT = "particle_count"
    NUMBER_DENSITY = "number_density"
    MEAN_DIAMETER = "mean_diameter"
    COVERAGE = "coverage"
    PERIMETER_DENSITY = "perimeter_density"
    REVIEW = "review"
    ANOMALIES = "anomalies"
    RANK = "rank"
    COMPARE_GROUPS = "compare_groups"
    DISTRIBUTION = "distribution"
    COMPARE_MODELS = "compare_models"
    UNSUPPORTED = "unsupported"


class _Metric(StrEnum):
    PARTICLE_COUNT = "particle_count"
    NUMBER_DENSITY_PX2 = "number_density_px2"
    MEAN_DIAMETER_PX = "mean_equivalent_diameter_px"
    COVERAGE_RATIO = "coverage_ratio"
    PERIMETER_DENSITY_PX = "perimeter_density_px"


class _GroupBy(StrEnum):
    IMAGE = "image"
    SAMPLE = "sample"
    MATERIAL = "material"


@dataclass(frozen=True, slots=True)
class _Intent:
    kind: _IntentKind
    metric: _Metric | None = None
    group_by: _GroupBy | None = None
    order: str = "desc"
    top_k: int = 10
    statistic: str = "mean"
    bins: int = 20
    error: str | None = None


@dataclass(frozen=True, slots=True)
class _RunDatum:
    run_id: str
    image_id: str
    filename: str
    sample_id: str
    material_name: str | None
    material_formula: str | None
    scale_nm_per_pixel: float | None
    model_id: str
    status: str
    particle_count: int
    stored_particle_rows: int
    stored_particle_nm_rows: int
    roi_area_px: int
    number_density_px2: float
    number_density_um2: float | None
    mean_diameter_px: float | None
    mean_diameter_nm: float | None
    coverage_ratio: float
    perimeter_density_px: float
    perimeter_density_um: float | None
    quality_status: str
    quality_reasons: tuple[str, ...]
    recommendations: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class _Scope:
    runs: tuple[_RunDatum, ...]
    selected_run_count: int
    status_counts: dict[str, int]
    warnings: tuple[str, ...]
    normalized_run_ids: tuple[str, ...]


class _ScopeError(ValueError):
    pass


class SqlAlchemyDataToolService:
    """Read-only implementation of ``DataToolService`` over persisted analysis results."""

    def __init__(
        self,
        session_factory: SessionFactory,
        *,
        distribution_evidence_limit: int = DEFAULT_DISTRIBUTION_EVIDENCE_LIMIT,
    ) -> None:
        if distribution_evidence_limit <= 0:
            raise ValueError("distribution_evidence_limit must be positive")
        self._session_factory = session_factory
        self._distribution_evidence_limit = distribution_evidence_limit

    def answer(self, query: DataQuery) -> DataQueryResult:
        intent = _resolve_intent(query.question)
        arguments = _base_arguments(query, intent)
        if intent.kind == _IntentKind.UNSUPPORTED:
            reason = intent.error or "unsupported or ambiguous data intent"
            return _insufficient_result(
                tool_name="resolve_data_intent",
                arguments=arguments,
                reason=reason,
                outcome="insufficient_data",
                needs_clarification=True,
            )

        session = self._session_factory()
        try:
            try:
                scope = _load_scope(session, query)
            except _ScopeError as error:
                return _insufficient_result(
                    tool_name=_tool_name(intent),
                    arguments=arguments,
                    reason=str(error),
                    outcome="error",
                )

            arguments["run_ids"] = list(scope.normalized_run_ids)
            ambiguous = _ambiguous_images(scope)
            if (
                ambiguous
                and not scope.normalized_run_ids
                and intent.kind
                in {
                    _IntentKind.OVERVIEW,
                    _IntentKind.PARTICLE_COUNT,
                    _IntentKind.NUMBER_DENSITY,
                    _IntentKind.MEAN_DIAMETER,
                    _IntentKind.COVERAGE,
                    _IntentKind.PERIMETER_DENSITY,
                    _IntentKind.RANK,
                    _IntentKind.COMPARE_GROUPS,
                    _IntentKind.DISTRIBUTION,
                }
            ):
                choices = ", ".join(
                    f"{image_id}=[{', '.join(run_ids)}]" for image_id, run_ids in ambiguous.items()
                )
                return _insufficient_result(
                    tool_name=_tool_name(intent),
                    arguments=arguments,
                    reason=(
                        "同一图像存在多个完成运行，不能默认把替代模型或复核运行重复汇总；"
                        f"请用 run_ids 每张图像选择一个运行。候选：{choices}"
                    ),
                    outcome="insufficient_data",
                    warnings=scope.warnings,
                    sources=_source_ids(scope.runs),
                    needs_clarification=True,
                )
            if intent.kind == _IntentKind.OVERVIEW or intent.metric in {
                _Metric.PARTICLE_COUNT,
                _Metric.NUMBER_DENSITY_PX2,
                _Metric.MEAN_DIAMETER_PX,
                _Metric.PERIMETER_DENSITY_PX,
            }:
                incomplete_runs = [
                    run for run in scope.runs if run.stored_particle_rows != run.particle_count
                ]
                if incomplete_runs:
                    details = ", ".join(
                        f"{run.run_id}={run.stored_particle_rows}/{run.particle_count}"
                        for run in incomplete_runs
                    )
                    return _insufficient_result(
                        tool_name=_tool_name(intent),
                        arguments=arguments,
                        reason=(
                            "颗粒明细行数与汇总 particle_count 不一致，"
                            "不能判定颗粒数或用残缺子集计算粒径/周长密度；"
                            f"请修复或重跑这些运行：{details}"
                        ),
                        outcome="insufficient_data",
                        warnings=scope.warnings,
                        sources=_source_ids(scope.runs),
                    )
            if intent.kind == _IntentKind.OVERVIEW:
                return _overview_result(scope, arguments)
            if intent.kind == _IntentKind.PARTICLE_COUNT:
                return _particle_count_result(scope, arguments)
            if intent.kind == _IntentKind.NUMBER_DENSITY:
                return _number_density_result(scope, arguments)
            if intent.kind == _IntentKind.MEAN_DIAMETER:
                return _mean_diameter_result(scope, arguments)
            if intent.kind == _IntentKind.COVERAGE:
                return _coverage_result(scope, arguments)
            if intent.kind == _IntentKind.PERIMETER_DENSITY:
                return _perimeter_density_result(scope, arguments)
            if intent.kind == _IntentKind.REVIEW:
                return _review_result(scope, arguments)
            if intent.kind == _IntentKind.ANOMALIES:
                return _anomaly_result(scope, arguments)
            if intent.kind == _IntentKind.RANK:
                return _ranking_result(scope, arguments, intent)
            if intent.kind == _IntentKind.COMPARE_GROUPS:
                return _compare_groups_result(scope, arguments, intent)
            if intent.kind == _IntentKind.DISTRIBUTION:
                return _distribution_result(
                    session,
                    scope,
                    arguments,
                    intent,
                    evidence_limit=self._distribution_evidence_limit,
                )
            if intent.kind == _IntentKind.COMPARE_MODELS:
                return _compare_models_result(scope, arguments, intent)
            raise AssertionError(f"unhandled data intent: {intent.kind}")
        finally:
            session.close()


def _resolve_intent(question: str) -> _Intent:
    normalized = question.casefold().strip()
    metrics = _mentioned_metrics(normalized)
    model_compare_signals = (
        "模型对比",
        "模型比较",
        "比较模型",
        "不同模型",
        "compare models",
        "model comparison",
    )
    if any(signal in normalized for signal in model_compare_signals):
        if len(metrics) != 1:
            return _Intent(
                _IntentKind.UNSUPPORTED,
                error=(
                    "模型对比必须明确一个指标：颗粒数、颗粒数密度、"
                    "平均粒径、覆盖率或周长密度"
                ),
            )
        return _Intent(_IntentKind.COMPARE_MODELS, metric=metrics[0])
    distribution_signals = (
        "分布",
        "直方图",
        "四分位",
        "中位数",
        "distribution",
        "histogram",
        "quantile",
    )
    if any(signal in normalized for signal in distribution_signals):
        if len(metrics) != 1:
            return _Intent(
                _IntentKind.UNSUPPORTED,
                error=(
                    "分布问题必须明确一个指标：颗粒数、颗粒数密度、"
                    "平均粒径、覆盖率或周长密度"
                ),
            )
        return _Intent(
            _IntentKind.DISTRIBUTION,
            metric=metrics[0],
            bins=_requested_bins(normalized),
        )
    if any(word in normalized for word in ("异常", "离群", "outlier", "anomal")):
        return _Intent(_IntentKind.ANOMALIES)
    compare_signals = ("比较", "对比", "差异", " versus ", " vs ", "compare")
    if any(signal in normalized for signal in compare_signals):
        if len(metrics) != 1:
            return _Intent(
                _IntentKind.UNSUPPORTED,
                error=(
                    "分组比较必须明确一个指标：颗粒数、颗粒数密度、"
                    "平均粒径、覆盖率或周长密度"
                ),
            )
        group_by = _ranking_group(normalized)
        if group_by not in {_GroupBy.SAMPLE, _GroupBy.MATERIAL}:
            return _Intent(
                _IntentKind.UNSUPPORTED,
                error="分组比较必须明确按样品或材料分组",
            )
        return _Intent(
            _IntentKind.COMPARE_GROUPS,
            metric=metrics[0],
            group_by=group_by,
            statistic=_requested_statistic(normalized),
        )
    if any(signal in normalized for signal in _RANK_SIGNALS):
        if len(metrics) != 1:
            return _Intent(
                _IntentKind.UNSUPPORTED,
                error=(
                    "排名问题必须明确一个指标：颗粒数、颗粒数密度、"
                    "平均粒径、覆盖率或周长密度"
                ),
            )
        group_by = _ranking_group(normalized)
        if group_by is None:
            return _Intent(
                _IntentKind.UNSUPPORTED,
                error="排名问题必须明确按图像、样品或材料分组",
            )
        order = "asc" if any(word in normalized for word in ("最低", "最小", "lowest")) else "desc"
        return _Intent(
            _IntentKind.RANK,
            metric=metrics[0],
            group_by=group_by,
            order=order,
            top_k=_top_k(normalized),
        )
    if any(word in normalized for word in ("需复核", "需要复核", "复核结果", "review required")):
        return _Intent(_IntentKind.REVIEW)
    overview_signals = (
        "任务概览",
        "结果概览",
        "数据概览",
        "结果汇总",
        "当前结果",
        "当前数据",
        "overview",
        "summary",
    )
    if any(word in normalized for word in overview_signals):
        return _Intent(_IntentKind.OVERVIEW)
    if len(metrics) > 1:
        return _Intent(_IntentKind.OVERVIEW)
    if metrics == [_Metric.PARTICLE_COUNT]:
        return _Intent(_IntentKind.PARTICLE_COUNT, metric=metrics[0])
    if metrics == [_Metric.NUMBER_DENSITY_PX2]:
        return _Intent(_IntentKind.NUMBER_DENSITY, metric=metrics[0])
    if metrics == [_Metric.MEAN_DIAMETER_PX]:
        return _Intent(_IntentKind.MEAN_DIAMETER, metric=metrics[0])
    if metrics == [_Metric.COVERAGE_RATIO]:
        return _Intent(_IntentKind.COVERAGE, metric=metrics[0])
    if metrics == [_Metric.PERIMETER_DENSITY_PX]:
        return _Intent(_IntentKind.PERIMETER_DENSITY, metric=metrics[0])
    return _Intent(
        _IntentKind.UNSUPPORTED,
        error=("仅支持任务概览、指标、分布、异常/复核、明确分组排名或比较，以及模型对比"),
    )


def _mentioned_metrics(question: str) -> list[_Metric]:
    metrics: list[_Metric] = []
    perimeter_density_signals = (
        "周长密度",
        "边界密度",
        "perimeter density",
        "boundary density",
    )
    if any(word in question for word in perimeter_density_signals):
        metrics.append(_Metric.PERIMETER_DENSITY_PX)
    density_signals = (
        "颗粒数密度",
        "颗粒密度",
        "粒子数密度",
        "粒子密度",
        "number density",
    )
    density_mentioned = any(word in question for word in density_signals)
    if density_mentioned:
        metrics.append(_Metric.NUMBER_DENSITY_PX2)
    if not density_mentioned and any(
        word in question for word in ("颗粒数", "粒子数", "particle count")
    ):
        metrics.append(_Metric.PARTICLE_COUNT)
    diameter_signals = ("平均粒径", "粒径均值", "mean diameter", "average diameter")
    if any(word in question for word in diameter_signals):
        metrics.append(_Metric.MEAN_DIAMETER_PX)
    if any(word in question for word in ("覆盖率", "coverage")):
        metrics.append(_Metric.COVERAGE_RATIO)
    return metrics


def _ranking_group(question: str) -> _GroupBy | None:
    if any(word in question for word in ("图像", "图片", "哪张", "image")):
        return _GroupBy.IMAGE
    if any(word in question for word in ("样品", "样本", "哪组", "sample")):
        return _GroupBy.SAMPLE
    if any(word in question for word in ("材料", "material")):
        return _GroupBy.MATERIAL
    return None


def _top_k(question: str) -> int:
    match = re.search(r"(?:前\s*|top\s*)(\d{1,3})", question)
    if match is None:
        return 10
    return min(100, max(1, int(match.group(1))))


def _requested_bins(question: str) -> int:
    match = re.search(r"(\d{1,3})\s*(?:个)?(?:箱|组|bins?)", question)
    if match is None:
        return 20
    return min(200, max(2, int(match.group(1))))


def _requested_statistic(question: str) -> str:
    if any(word in question for word in ("中位数", "median")):
        return "median"
    if any(word in question for word in ("最小值", "minimum", " min ")):
        return "min"
    if any(word in question for word in ("最大值", "maximum", " max ")):
        return "max"
    return "mean"


def _base_arguments(query: DataQuery, intent: _Intent) -> dict[str, Any]:
    arguments: dict[str, Any] = {
        "job_id": query.job_id,
        "image_id": query.image_id,
        "run_ids": list(dict.fromkeys(query.run_ids)),
        "intent": intent.kind.value,
    }
    if intent.metric is not None:
        arguments["metric"] = intent.metric.value
    if intent.group_by is not None:
        arguments.update(
            {
                "group_by": intent.group_by.value,
                "order": intent.order,
                "top_k": intent.top_k,
            }
        )
    if intent.kind == _IntentKind.COMPARE_GROUPS:
        arguments["statistic"] = intent.statistic
    if intent.kind == _IntentKind.DISTRIBUTION:
        arguments["bins"] = intent.bins
    return arguments


def _load_scope(session: Session, query: DataQuery) -> _Scope:
    job = session.scalar(
        select(AnalysisJob).where(
            AnalysisJob.job_id == query.job_id,
            AnalysisJob.tenant_id == query.tenant_id,
        )
    )
    if job is None:
        raise _ScopeError("指定任务不存在，无法读取实验数据")

    if query.image_id is not None:
        image = session.scalar(
            select(ImageAsset)
            .join(AnalysisJob, AnalysisJob.job_id == ImageAsset.job_id)
            .where(
                ImageAsset.image_id == query.image_id,
                ImageAsset.job_id == query.job_id,
                AnalysisJob.tenant_id == query.tenant_id,
            )
        )
        if image is None:
            raise _ScopeError("image_id 不属于指定任务")

    normalized_run_ids = tuple(dict.fromkeys(query.run_ids))
    statement = (
        select(SegmentationRun)
        .join(AnalysisJob, AnalysisJob.job_id == SegmentationRun.job_id)
        .where(AnalysisJob.tenant_id == query.tenant_id)
        .options(
            joinedload(SegmentationRun.image),
            joinedload(SegmentationRun.summary),
        )
        .order_by(SegmentationRun.created_at, SegmentationRun.run_id)
    )
    if normalized_run_ids:
        records = session.scalars(
            statement.where(
                SegmentationRun.job_id == query.job_id,
                SegmentationRun.run_id.in_(normalized_run_ids),
            )
        ).all()
        by_id = {record.run_id: record for record in records}
        missing = [run_id for run_id in normalized_run_ids if run_id not in by_id]
        if missing:
            raise _ScopeError("run_ids 包含不存在或不属于指定任务的运行")
        records = [by_id[run_id] for run_id in normalized_run_ids]
    else:
        filters = [SegmentationRun.job_id == query.job_id]
        if query.image_id is not None:
            filters.append(SegmentationRun.image_id == query.image_id)
        records = session.scalars(statement.where(*filters)).all()

    if query.image_id is not None and any(record.image_id != query.image_id for record in records):
        raise _ScopeError("run_ids 与 image_id 筛选条件冲突")

    status_counts = dict(sorted(Counter(record.status for record in records).items()))
    complete_records = [
        record
        for record in records
        if record.status in _COMPLETED_STATUSES and record.summary is not None
    ]
    particle_stats = _particle_stats(session, [record.run_id for record in complete_records])
    warnings: list[str] = []
    data: list[_RunDatum] = []
    for record in records:
        if record.status not in _COMPLETED_STATUSES:
            warnings.append(f"{record.run_id}: run_status={record.status}; completed data excluded")
            continue
        summary = record.summary
        if summary is None:
            warnings.append(f"{record.run_id}: completed run has no persisted summary")
            continue
        (
            row_count,
            nm_row_count,
            particle_mean_px,
            particle_mean_nm,
            perimeter_total_px,
        ) = particle_stats.get(record.run_id, (0, 0, None, None, 0.0))
        image = record.image
        effective_scale, scale_warning = _run_scale_nm_per_pixel(record)
        if scale_warning is not None:
            warnings.append(scale_warning)
        if row_count != summary.particle_count:
            warnings.append(
                f"{record.run_id}: summary particle_count={summary.particle_count} "
                f"but stored particle rows={row_count}"
            )
        if 0 < nm_row_count < row_count:
            warnings.append(
                f"{record.run_id}: physical diameter exists for "
                f"{nm_row_count}/{row_count} particles"
            )
            if effective_scale is not None:
                warnings.append(
                    f"{record.run_id}: missing physical diameters recomputed from "
                    f"pixel diameter and scale={effective_scale:g} nm/px"
                )
        if summary.quality_status != QualityStatus.PASS.value:
            warnings.append(f"{record.run_id}: quality_status={summary.quality_status}")
        reasons = _string_tuple(summary.quality_json.get("reasons"))
        recommendations = _string_tuple(summary.quality_json.get("recommendations"))
        warnings.extend(f"{record.run_id}: {reason}" for reason in reasons)
        effective_particle_mean_nm: float | None
        if row_count > 0 and particle_mean_px is not None:
            if effective_scale is not None:
                effective_particle_mean_nm = particle_mean_px * effective_scale
            elif nm_row_count == row_count:
                effective_particle_mean_nm = particle_mean_nm
            else:
                effective_particle_mean_nm = None
        else:
            effective_particle_mean_nm = None
        if summary.roi_area_px > 0:
            effective_density_px2 = summary.particle_count / summary.roi_area_px
        else:
            effective_density_px2 = summary.number_density_px2
        if not isclose(
            summary.number_density_px2,
            effective_density_px2,
            rel_tol=1e-9,
            abs_tol=1e-12,
        ):
            warnings.append(
                f"{record.run_id}: persisted number_density_px2="
                f"{summary.number_density_px2:g} differs from particle_count/roi_area_px="
                f"{effective_density_px2:g}; using the recomputed value"
            )
        effective_density_um2: float | None = None
        if effective_scale is not None and summary.roi_area_px > 0:
            physical_area_um2 = summary.roi_area_px * (effective_scale / 1000.0) ** 2
            effective_density_um2 = summary.particle_count / physical_area_um2
            if summary.number_density_um2 is not None and not isclose(
                summary.number_density_um2,
                effective_density_um2,
                rel_tol=1e-9,
                abs_tol=1e-9,
            ):
                warnings.append(
                    f"{record.run_id}: persisted number_density_um2="
                    f"{summary.number_density_um2:g} differs from the effective "
                    f"scale-derived value={effective_density_um2:g}; "
                    "using the recomputed value"
                )
        elif summary.number_density_um2 is not None:
            warnings.append(
                f"{record.run_id}: persisted number_density_um2 was ignored because "
                "no valid immutable/effective physical scale is available"
            )
        effective_perimeter_density_px = (
            perimeter_total_px / summary.roi_area_px if summary.roi_area_px > 0 else 0.0
        )
        if not isclose(
            summary.perimeter_density_px,
            effective_perimeter_density_px,
            rel_tol=1e-9,
            abs_tol=1e-12,
        ):
            warnings.append(
                f"{record.run_id}: persisted perimeter_density_px="
                f"{summary.perimeter_density_px:g} differs from "
                f"sum(perimeter_px)/roi_area_px={effective_perimeter_density_px:g}; "
                "using the recomputed value"
            )
        effective_perimeter_density_um: float | None = None
        if effective_scale is not None:
            effective_perimeter_density_um = (
                effective_perimeter_density_px * 1000.0 / effective_scale
            )
            if summary.perimeter_density_um is not None and not isclose(
                summary.perimeter_density_um,
                effective_perimeter_density_um,
                rel_tol=1e-9,
                abs_tol=1e-12,
            ):
                warnings.append(
                    f"{record.run_id}: persisted perimeter_density_um="
                    f"{summary.perimeter_density_um:g} differs from the effective "
                    f"scale-derived value={effective_perimeter_density_um:g}; "
                    "using the recomputed value"
                )
        elif summary.perimeter_density_um is not None:
            warnings.append(
                f"{record.run_id}: persisted perimeter_density_um was ignored because "
                "no valid immutable/effective physical scale is available"
            )
        data.append(
            _RunDatum(
                run_id=record.run_id,
                image_id=record.image_id,
                filename=image.filename,
                sample_id=image.sample_id,
                material_name=image.material_name,
                material_formula=image.material_formula,
                scale_nm_per_pixel=effective_scale,
                model_id=record.model_id,
                status=record.status,
                particle_count=summary.particle_count,
                stored_particle_rows=row_count,
                stored_particle_nm_rows=nm_row_count,
                roi_area_px=summary.roi_area_px,
                number_density_px2=effective_density_px2,
                number_density_um2=effective_density_um2,
                mean_diameter_px=(particle_mean_px if row_count > 0 else None),
                mean_diameter_nm=(effective_particle_mean_nm),
                coverage_ratio=summary.coverage_ratio,
                perimeter_density_px=effective_perimeter_density_px,
                perimeter_density_um=effective_perimeter_density_um,
                quality_status=summary.quality_status,
                quality_reasons=reasons,
                recommendations=recommendations,
            )
        )

    completed_by_image = Counter(run.image_id for run in data)
    for image_id, count in sorted(completed_by_image.items()):
        if count > 1:
            warnings.append(
                f"{image_id}: {count} completed model runs are treated as separate results"
            )

    return _Scope(
        runs=tuple(data),
        selected_run_count=len(records),
        status_counts=status_counts,
        warnings=tuple(dict.fromkeys(warnings)),
        normalized_run_ids=normalized_run_ids,
    )


def _run_scale_nm_per_pixel(record: SegmentationRun) -> tuple[float | None, str | None]:
    """Resolve physical scale from the immutable run contract when available."""

    payload = record.run_config_json
    if payload.get("provenance_status") == "complete":
        if "scale_nm_per_pixel" not in payload:
            return (
                None,
                f"{record.run_id}: complete run configuration lacks frozen physical scale; "
                "live image metadata was not used",
            )
        value = payload["scale_nm_per_pixel"]
        if value is None:
            return None, None
        if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
            return (
                None,
                f"{record.run_id}: frozen physical scale is invalid; "
                "live image metadata was not used",
            )
        return float(value), None

    live_scale = record.image.scale_nm_per_pixel
    rendered = "none" if live_scale is None else f"{live_scale:g} nm/px"
    return (
        live_scale,
        f"{record.run_id}: legacy run configuration has no complete frozen scale; "
        f"using live image metadata fallback={rendered}",
    )


def _particle_stats(
    session: Session,
    run_ids: list[str],
) -> dict[str, tuple[int, int, float | None, float | None, float]]:
    if not run_ids:
        return {}
    rows = session.execute(
        select(
            ParticleRecord.run_id,
            func.count(ParticleRecord.particle_id),
            func.count(ParticleRecord.equivalent_diameter_nm),
            func.avg(ParticleRecord.equivalent_diameter_px),
            func.avg(ParticleRecord.equivalent_diameter_nm),
            func.sum(ParticleRecord.perimeter_px),
        )
        .where(ParticleRecord.run_id.in_(run_ids))
        .group_by(ParticleRecord.run_id)
    ).all()
    return {
        str(run_id): (
            int(count),
            int(nm_count),
            float(mean_px) if mean_px is not None else None,
            float(mean_nm) if mean_nm is not None else None,
            float(perimeter_total_px or 0.0),
        )
        for run_id, count, nm_count, mean_px, mean_nm, perimeter_total_px in rows
    }


def _string_tuple(value: object) -> tuple[str, ...]:
    if not isinstance(value, list):
        return ()
    return tuple(item for item in value if isinstance(item, str) and item)


def _overview_result(scope: _Scope, arguments: dict[str, Any]) -> DataQueryResult:
    if not scope.runs:
        return _no_completed_data("get_job_overview", arguments, scope)
    sources = _source_ids(scope.runs)
    review_count = sum(
        run.quality_status == QualityStatus.REVIEW_REQUIRED.value for run in scope.runs
    )
    total_particles = sum(run.particle_count for run in scope.runs)
    aggregates = {
        "selected_run_count": scope.selected_run_count,
        "completed_run_count": len(scope.runs),
        "particle_count_total": total_particles,
        "review_required_count": review_count,
        "status_counts": scope.status_counts,
    }
    return _success_result(
        tool_name="get_job_overview",
        arguments=arguments,
        rows=[_overview_row(run) for run in scope.runs],
        aggregates=aggregates,
        units={
            "particle_count": "count",
            "roi_area_px": "px^2",
            "number_density_px2": "px^-2",
            "number_density_um2": "um^-2",
            "mean_equivalent_diameter_px": "px",
            "mean_equivalent_diameter_nm": "nm",
            "coverage_ratio": "ratio",
            "perimeter_density_px": "px^-1",
            "perimeter_density_um": "um^-1",
        },
        sources=sources,
        warnings=scope.warnings,
        answer=(
            f"任务筛选到 {scope.selected_run_count} 个运行，其中 {len(scope.runs)} 个已有完成结果；"
            f"完成结果共记录 {total_particles} 个颗粒，{review_count} 个运行需要复核。"
        ),
    )


def _particle_count_result(scope: _Scope, arguments: dict[str, Any]) -> DataQueryResult:
    if not scope.runs:
        return _no_completed_data("get_metric", arguments, scope)
    total = sum(run.particle_count for run in scope.runs)
    rows = [
        {
            **_identity_row(run),
            "particle_count": run.particle_count,
            "stored_particle_rows": run.stored_particle_rows,
        }
        for run in scope.runs
    ]
    return _success_result(
        tool_name="get_metric",
        arguments=arguments,
        rows=rows,
        aggregates={"particle_count": total, "completed_run_count": len(scope.runs)},
        units={"particle_count": "count", "stored_particle_rows": "count"},
        sources=_source_ids(scope.runs),
        warnings=scope.warnings,
        answer=f"所选 {len(scope.runs)} 个完成运行共记录 {total} 个颗粒。",
    )


def _number_density_result(
    scope: _Scope,
    arguments: dict[str, Any],
) -> DataQueryResult:
    if not scope.runs:
        return _no_completed_data("get_metric", arguments, scope)
    comparison_unit = _metric_comparison_unit(
        scope.runs,
        _Metric.NUMBER_DENSITY_PX2,
    )
    if comparison_unit is None:
        return _incomparable_density_result("get_metric", arguments, scope)

    metric_name = _metric_name(_Metric.NUMBER_DENSITY_PX2, comparison_unit)
    arguments["metric"] = metric_name
    density = _group_metric(
        list(scope.runs),
        _Metric.NUMBER_DENSITY_PX2,
        comparison_unit=comparison_unit,
    )
    if density is None:
        return _insufficient_result(
            tool_name="get_metric",
            arguments=arguments,
            reason="完成结果缺少有效 ROI 面积或颗粒数密度，无法汇总",
            outcome="insufficient_data",
            warnings=scope.warnings,
            sources=_source_ids(scope.runs),
        )

    warnings = list(scope.warnings)
    for run in scope.runs:
        if run.number_density_um2 is None:
            warnings.append(
                f"{run.run_id}: physical scale unavailable; um^-2 number density omitted"
            )
    rows = [
        {
            **_identity_row(run),
            "particle_count": run.particle_count,
            "roi_area_px": run.roi_area_px,
            "number_density_px2": run.number_density_px2,
            "number_density_um2": run.number_density_um2,
        }
        for run in scope.runs
    ]
    total_roi_area_px = sum(run.roi_area_px for run in scope.runs)
    aggregates: dict[str, Any] = {
        metric_name: density,
        "particle_count": sum(run.particle_count for run in scope.runs),
        "roi_area_px": total_roi_area_px,
        "completed_run_count": len(scope.runs),
    }
    units = {
        "number_density_px2": "px^-2",
        "number_density_um2": "um^-2",
        "particle_count": "count",
        "roi_area_px": "px^2",
    }
    if comparison_unit == "um^-2":
        physical_area = sum(
            area for run in scope.runs if (area := _physical_roi_area_um2(run)) is not None
        )
        aggregates["roi_area_um2"] = physical_area
        units["roi_area_um2"] = "um^2"
    return _success_result(
        tool_name="get_metric",
        arguments=arguments,
        rows=rows,
        aggregates=aggregates,
        units=units,
        sources=_source_ids(scope.runs),
        warnings=tuple(dict.fromkeys(warnings)),
        answer=(
            f"所选 {len(scope.runs)} 个完成运行按总颗粒数/总 ROI 面积合并的"
            f"颗粒数密度为 {density:.4g} {comparison_unit}。"
            "“较高”需要与相同物理尺度、ROI 和分割设置下的对照结果比较。"
        ),
    )


def _mean_diameter_result(scope: _Scope, arguments: dict[str, Any]) -> DataQueryResult:
    if not scope.runs:
        return _no_completed_data("get_metric", arguments, scope)
    weighted_px = _weighted_mean(
        (run.mean_diameter_px, _diameter_px_weight(run)) for run in scope.runs
    )
    physical_complete = all(
        _physical_diameter_complete(run) for run in scope.runs if _diameter_px_weight(run) > 0
    )
    weighted_nm = (
        _weighted_mean((run.mean_diameter_nm, _diameter_nm_weight(run)) for run in scope.runs)
        if physical_complete
        else None
    )
    if len({run.image_id for run in scope.runs}) > 1 and weighted_nm is None:
        return _insufficient_result(
            tool_name="get_metric",
            arguments=arguments,
            reason=("跨图像平均粒径缺少统一物理尺度；请限定 image_id/run_ids 或补充比例尺"),
            outcome="insufficient_data",
            warnings=scope.warnings,
            sources=_source_ids(scope.runs),
            needs_clarification=True,
        )
    if weighted_px is None and weighted_nm is None:
        return _insufficient_result(
            tool_name="get_metric",
            arguments=arguments,
            reason="完成结果中没有可用的平均粒径数据",
            outcome="insufficient_data",
            warnings=scope.warnings,
            sources=_source_ids(scope.runs),
        )
    warnings = list(scope.warnings)
    for run in scope.runs:
        if run.mean_diameter_nm is None:
            warnings.append(f"{run.run_id}: physical scale unavailable; nm diameter omitted")
    rows = [
        {
            **_identity_row(run),
            "particle_count": run.particle_count,
            "mean_equivalent_diameter_px": run.mean_diameter_px,
            "mean_equivalent_diameter_nm": run.mean_diameter_nm,
        }
        for run in scope.runs
        if run.mean_diameter_px is not None or run.mean_diameter_nm is not None
    ]
    aggregates = {
        "mean_equivalent_diameter_px": weighted_px,
        "mean_equivalent_diameter_nm": weighted_nm,
        "completed_run_count": len(scope.runs),
    }
    display = f"{weighted_nm:.4g} nm" if weighted_nm is not None else f"{weighted_px:.4g} px"
    return _success_result(
        tool_name="get_metric",
        arguments=arguments,
        rows=rows,
        aggregates=aggregates,
        units={
            "particle_count": "count",
            "mean_equivalent_diameter_px": "px",
            "mean_equivalent_diameter_nm": "nm",
        },
        sources=_source_ids(scope.runs),
        warnings=tuple(dict.fromkeys(warnings)),
        answer=f"按颗粒数加权的平均等效粒径为 {display}。",
    )


def _coverage_result(scope: _Scope, arguments: dict[str, Any]) -> DataQueryResult:
    if not scope.runs:
        return _no_completed_data("get_metric", arguments, scope)
    total_area = sum(run.roi_area_px for run in scope.runs)
    if total_area <= 0:
        return _insufficient_result(
            tool_name="get_metric",
            arguments=arguments,
            reason="完成结果缺少有效 ROI 面积，无法汇总覆盖率",
            outcome="insufficient_data",
            warnings=scope.warnings,
            sources=_source_ids(scope.runs),
        )
    coverage = sum(run.coverage_ratio * run.roi_area_px for run in scope.runs) / total_area
    rows = [
        {
            **_identity_row(run),
            "roi_area_px": run.roi_area_px,
            "coverage_ratio": run.coverage_ratio,
            "coverage_percent": run.coverage_ratio * 100,
        }
        for run in scope.runs
    ]
    return _success_result(
        tool_name="get_metric",
        arguments=arguments,
        rows=rows,
        aggregates={
            "coverage_ratio": coverage,
            "coverage_percent": coverage * 100,
            "roi_area_px": total_area,
        },
        units={"coverage_ratio": "ratio", "coverage_percent": "%", "roi_area_px": "px^2"},
        sources=_source_ids(scope.runs),
        warnings=scope.warnings,
        answer=f"按 ROI 面积加权的覆盖率为 {coverage:.4f}（{coverage * 100:.2f}%）。",
    )


def _perimeter_density_result(
    scope: _Scope,
    arguments: dict[str, Any],
) -> DataQueryResult:
    if not scope.runs:
        return _no_completed_data("get_metric", arguments, scope)
    comparison_unit = _metric_comparison_unit(
        scope.runs,
        _Metric.PERIMETER_DENSITY_PX,
    )
    if comparison_unit is None:
        return _incomparable_perimeter_density_result("get_metric", arguments, scope)

    metric_name = _metric_name(_Metric.PERIMETER_DENSITY_PX, comparison_unit)
    arguments["metric"] = metric_name
    density = _group_metric(
        list(scope.runs),
        _Metric.PERIMETER_DENSITY_PX,
        comparison_unit=comparison_unit,
    )
    if density is None:
        return _insufficient_result(
            tool_name="get_metric",
            arguments=arguments,
            reason="完成结果缺少有效 ROI 面积或周长密度，无法汇总",
            outcome="insufficient_data",
            warnings=scope.warnings,
            sources=_source_ids(scope.runs),
        )

    warnings = list(scope.warnings)
    for run in scope.runs:
        if run.perimeter_density_um is None:
            warnings.append(
                f"{run.run_id}: physical scale unavailable; um^-1 perimeter density omitted"
            )
    rows = [
        {
            **_identity_row(run),
            "roi_area_px": run.roi_area_px,
            "perimeter_density_px": run.perimeter_density_px,
            "perimeter_density_um": run.perimeter_density_um,
        }
        for run in scope.runs
    ]
    aggregates: dict[str, Any] = {
        metric_name: density,
        "roi_area_px": sum(run.roi_area_px for run in scope.runs),
        "completed_run_count": len(scope.runs),
    }
    units = {
        "perimeter_density_px": "px^-1",
        "perimeter_density_um": "um^-1",
        "roi_area_px": "px^2",
    }
    if comparison_unit == "um^-1":
        physical_area = sum(
            area for run in scope.runs if (area := _physical_roi_area_um2(run)) is not None
        )
        aggregates["roi_area_um2"] = physical_area
        units["roi_area_um2"] = "um^2"
    return _success_result(
        tool_name="get_metric",
        arguments=arguments,
        rows=rows,
        aggregates=aggregates,
        units=units,
        sources=_source_ids(scope.runs),
        warnings=tuple(dict.fromkeys(warnings)),
        answer=(
            f"所选 {len(scope.runs)} 个完成运行按总颗粒周长/总 ROI 面积合并的"
            f"周长密度为 {density:.4g} {comparison_unit}。"
            "跨样品解释需要保持比例尺、ROI 和分割设置可比。"
        ),
    )


def _review_result(scope: _Scope, arguments: dict[str, Any]) -> DataQueryResult:
    if not scope.runs:
        return _no_completed_data("find_review", arguments, scope)
    flagged = [
        run for run in scope.runs if run.quality_status == QualityStatus.REVIEW_REQUIRED.value
    ]
    rows = [
        {
            **_identity_row(run),
            "quality_status": run.quality_status,
            "reasons": list(run.quality_reasons),
            "recommendations": list(run.recommendations),
        }
        for run in flagged
    ]
    return _success_result(
        tool_name="find_review",
        arguments=arguments,
        rows=rows,
        aggregates={
            "review_required_count": len(flagged),
            "completed_run_count": len(scope.runs),
        },
        units={"review_required_count": "count"},
        sources=_source_ids(scope.runs),
        warnings=scope.warnings,
        answer=(
            f"共有 {len(flagged)} 个完成运行需要复核。"
            if flagged
            else "所选完成运行中没有标记为需要复核的结果。"
        ),
    )


def _anomaly_result(scope: _Scope, arguments: dict[str, Any]) -> DataQueryResult:
    if not scope.runs:
        return _no_completed_data("find_anomalies", arguments, scope)
    flagged = [
        run
        for run in scope.runs
        if run.quality_status != QualityStatus.PASS.value
        or run.stored_particle_rows != run.particle_count
    ]
    rows = [
        {
            **_identity_row(run),
            "summary_particle_count": run.particle_count,
            "stored_particle_rows": run.stored_particle_rows,
            "reasons": list(run.quality_reasons),
            "recommendations": list(run.recommendations),
        }
        for run in flagged
    ]
    return _success_result(
        tool_name="find_anomalies",
        arguments=arguments,
        rows=rows,
        aggregates={
            "anomaly_count": len(flagged),
            "completed_run_count": len(scope.runs),
        },
        units={"anomaly_count": "count"},
        sources=_source_ids(scope.runs),
        warnings=scope.warnings,
        answer=(
            f"发现 {len(flagged)} 个带质量或持久化一致性异常的完成运行。"
            if flagged
            else "所选完成运行未发现质量或持久化一致性异常。"
        ),
    )


def _ranking_result(
    scope: _Scope,
    arguments: dict[str, Any],
    intent: _Intent,
) -> DataQueryResult:
    if not scope.runs:
        return _no_completed_data("rank_samples", arguments, scope)
    if intent.metric is None or intent.group_by is None:
        raise AssertionError("resolved ranking intent lacks metric or group_by")
    if len({run.model_id for run in scope.runs}) > 1:
        return _insufficient_result(
            tool_name="rank_samples",
            arguments=arguments,
            reason="排名混入多个模型；请用 run_ids 为每张图像选择同一模型结果",
            outcome="insufficient_data",
            warnings=scope.warnings,
            sources=_source_ids(scope.runs),
            needs_clarification=True,
        )
    comparison_unit = _metric_comparison_unit(scope.runs, intent.metric)
    if comparison_unit is None:
        return _incomparable_metric_result(
            "rank_samples",
            arguments,
            scope,
            intent.metric,
        )
    metric_name = _metric_name(intent.metric, comparison_unit)
    arguments["metric"] = metric_name

    grouped: dict[str, list[_RunDatum]] = {}
    warnings = list(scope.warnings)
    for run in scope.runs:
        group = _group_key(run, intent.group_by)
        if group is None:
            warnings.append(f"{run.run_id}: missing {intent.group_by.value} metadata; excluded")
            continue
        grouped.setdefault(group, []).append(run)

    ranked_rows: list[dict[str, Any]] = []
    used_runs: list[_RunDatum] = []
    for group, runs in grouped.items():
        value = _group_metric(
            runs,
            intent.metric,
            comparison_unit=comparison_unit,
        )
        if value is None:
            warnings.append(f"{group}: metric {intent.metric.value} unavailable; excluded")
            continue
        used_runs.extend(runs)
        ranked_rows.append(
            {
                "group": group,
                "group_by": intent.group_by.value,
                "metric": metric_name,
                "value": value,
                "aggregation": _ranking_aggregation(intent.metric),
                "observation_unit": "run",
                "observation_count": len(runs),
                "weight_total": _ranking_weight_total(
                    runs,
                    intent.metric,
                    comparison_unit=comparison_unit,
                ),
                "weight_unit": _ranking_weight_unit(intent.metric, comparison_unit),
                "image_ids": sorted({run.image_id for run in runs}),
                "run_ids": sorted(run.run_id for run in runs),
            }
        )
    if not ranked_rows:
        return _insufficient_result(
            tool_name="rank_samples",
            arguments=arguments,
            reason="没有具备分组元数据和目标指标的完成结果",
            outcome="insufficient_data",
            warnings=tuple(dict.fromkeys(warnings)),
        )
    if intent.order == "desc":
        ranked_rows.sort(key=lambda row: (-float(row["value"]), str(row["group"])))
    else:
        ranked_rows.sort(key=lambda row: (float(row["value"]), str(row["group"])))
    ranked_rows = ranked_rows[: intent.top_k]
    included_run_ids = {
        run_id for row in ranked_rows for run_id in row["run_ids"] if isinstance(run_id, str)
    }
    sources = sorted(run.run_id for run in used_runs if run.run_id in included_run_ids)
    first = ranked_rows[0]
    unit = _comparison_metric_unit(intent.metric, comparison_unit)
    return _success_result(
        tool_name="rank_samples",
        arguments=arguments,
        rows=ranked_rows,
        aggregates={"group_count": len(ranked_rows)},
        units={
            str(arguments["metric"]): unit,
            "value": unit,
        },
        sources=sources,
        warnings=tuple(dict.fromkeys(warnings)),
        answer=(
            f"按 {metric_name} {intent.order} 排名，首位是 {first['group']}，"
            f"值为 {float(first['value']):.4g}。"
        ),
    )


def _compare_groups_result(
    scope: _Scope,
    arguments: dict[str, Any],
    intent: _Intent,
) -> DataQueryResult:
    if not scope.runs:
        return _no_completed_data("compare_groups", arguments, scope)
    if intent.metric is None or intent.group_by not in {
        _GroupBy.SAMPLE,
        _GroupBy.MATERIAL,
    }:
        raise AssertionError("resolved group comparison lacks metric or group_by")
    if len({run.model_id for run in scope.runs}) > 1:
        return _insufficient_result(
            tool_name="compare_groups",
            arguments=arguments,
            reason="分组比较混入多个模型；请用 run_ids 为每张图像选择同一模型结果",
            outcome="insufficient_data",
            warnings=scope.warnings,
            sources=_source_ids(scope.runs),
            needs_clarification=True,
        )

    comparison_unit = _metric_comparison_unit(scope.runs, intent.metric)
    if comparison_unit is None:
        return _incomparable_metric_result(
            "compare_groups",
            arguments,
            scope,
            intent.metric,
        )
    metric_name = _metric_name(intent.metric, comparison_unit)
    arguments["metric"] = metric_name

    grouped: dict[str, list[_RunDatum]] = {}
    warnings = list(scope.warnings)
    for run in scope.runs:
        group = _group_key(run, intent.group_by)
        if group is None:
            warnings.append(f"{run.run_id}: missing {intent.group_by.value} metadata; excluded")
            continue
        grouped.setdefault(group, []).append(run)
    if len(grouped) < 2:
        return _insufficient_result(
            tool_name="compare_groups",
            arguments=arguments,
            reason="至少需要两个具有目标指标的样品或材料组",
            outcome="insufficient_data",
            warnings=tuple(dict.fromkeys(warnings)),
            sources=_source_ids(scope.runs),
            needs_clarification=True,
        )

    rows: list[dict[str, Any]] = []
    used: list[_RunDatum] = []
    for group, runs in sorted(grouped.items()):
        values: list[float] = []
        for run in runs:
            value = _run_metric(
                run,
                intent.metric,
                comparison_unit=comparison_unit,
            )
            if value is not None:
                values.append(value)
        if not values:
            warnings.append(f"{group}: metric {intent.metric.value} unavailable; excluded")
            continue
        used.extend(runs)
        value = _statistic(values, intent.statistic)
        weighting = "unweighted_run_values"
        if intent.metric in {
            _Metric.NUMBER_DENSITY_PX2,
            _Metric.PERIMETER_DENSITY_PX,
        } and intent.statistic == "mean":
            grouped_density = _group_metric(
                runs,
                intent.metric,
                comparison_unit=comparison_unit,
            )
            if grouped_density is None:
                warnings.append(
                    f"{group}: density metric lacks a valid ROI area; excluded"
                )
                continue
            value = grouped_density
            weighting = _ranking_aggregation(intent.metric)
        rows.append(
            {
                "group": group,
                "group_by": intent.group_by.value,
                "metric": metric_name,
                "statistic": intent.statistic,
                "value": value,
                "count": len(values),
                "observation_unit": "run",
                "weighting": weighting,
                "min": min(values),
                "max": max(values),
                "run_ids": sorted(run.run_id for run in runs),
            }
        )
    if len(rows) < 2:
        return _insufficient_result(
            tool_name="compare_groups",
            arguments=arguments,
            reason="至少两个组需要具备可比较的目标指标",
            outcome="insufficient_data",
            warnings=tuple(dict.fromkeys(warnings)),
        )
    rows.sort(key=lambda row: (-float(row["value"]), str(row["group"])))
    arguments["groups"] = [str(row["group"]) for row in rows]
    unit = _comparison_metric_unit(intent.metric, comparison_unit)
    comparison_method = (
        "按总量/总 ROI 面积"
        if intent.metric
        in {_Metric.NUMBER_DENSITY_PX2, _Metric.PERIMETER_DENSITY_PX}
        and intent.statistic == "mean"
        else "按等权运行观测"
    )
    return _success_result(
        tool_name="compare_groups",
        arguments=arguments,
        rows=rows,
        aggregates={"group_count": len(rows)},
        units={
            str(arguments["metric"]): unit,
            "value": unit,
        },
        sources=_source_ids(used),
        warnings=tuple(dict.fromkeys(warnings)),
        answer=(
            f"已{comparison_method}，按 {intent.group_by.value} 比较 {len(rows)} 组的 "
            f"{metric_name}；{intent.statistic} 最高的是 {rows[0]['group']}。"
        ),
    )


def _distribution_result(
    session: Session,
    scope: _Scope,
    arguments: dict[str, Any],
    intent: _Intent,
    *,
    evidence_limit: int,
) -> DataQueryResult:
    if not scope.runs:
        return _no_completed_data("describe_distribution", arguments, scope)
    if intent.metric is None:
        raise AssertionError("resolved distribution intent lacks metric")
    comparison_unit = _metric_comparison_unit(scope.runs, intent.metric)
    if comparison_unit is None:
        return _incomparable_metric_result(
            "describe_distribution",
            arguments,
            scope,
            intent.metric,
        )
    if intent.metric == _Metric.MEAN_DIAMETER_PX:
        if comparison_unit == "nm":
            arguments["metric"] = "mean_equivalent_diameter_nm"
        return _particle_distribution_result(
            session,
            scope,
            arguments,
            intent,
            diameter_unit=comparison_unit,
            evidence_limit=evidence_limit,
        )

    metric_key = _metric_name(intent.metric, comparison_unit)
    arguments["metric"] = metric_key
    pairs = [
        (run, value)
        for run in scope.runs
        if (
            value := _run_metric(
                run,
                intent.metric,
                comparison_unit=comparison_unit,
            )
        )
        is not None
    ]
    values = [float(value) for _, value in pairs]
    rows = [{**_identity_row(run), metric_key: value} for run, value in pairs]
    if not values:
        return _insufficient_result(
            tool_name="describe_distribution",
            arguments=arguments,
            reason="目标指标没有可用观测值",
            outcome="insufficient_data",
            warnings=scope.warnings,
        )
    ordered = sorted(values)
    median_value = float(median(ordered))
    aggregates = {
        "count": len(ordered),
        "min": ordered[0],
        "q1": _percentile(ordered, 0.25),
        "median": median_value,
        "mean": fmean(ordered),
        "q3": _percentile(ordered, 0.75),
        "max": ordered[-1],
        "histogram": _histogram(ordered, intent.bins),
    }
    unit = _comparison_metric_unit(intent.metric, comparison_unit)
    return _success_result(
        tool_name="describe_distribution",
        arguments=arguments,
        rows=rows,
        aggregates=aggregates,
        units={
            str(arguments["metric"]): unit,
            "min": unit,
            "q1": unit,
            "median": unit,
            "mean": unit,
            "q3": unit,
            "max": unit,
        },
        sources=_source_ids(scope.runs),
        warnings=scope.warnings,
        answer=(
            f"{intent.metric.value} 共 {len(ordered)} 个观测；中位数 "
            f"{median_value:.4g} {unit}，范围 "
            f"{ordered[0]:.4g}–{ordered[-1]:.4g} {unit}。"
        ),
    )


def _particle_distribution_result(
    session: Session,
    scope: _Scope,
    arguments: dict[str, Any],
    intent: _Intent,
    *,
    diameter_unit: str,
    evidence_limit: int,
) -> DataQueryResult:
    """Aggregate all particles in SQL and return only bounded, deterministic evidence."""

    run_ids = _source_ids(scope.runs)
    scales = {run.run_id: run.scale_nm_per_pixel for run in scope.runs}
    diameter = _particle_diameter_expression(scope.runs, diameter_unit)
    predicate = (
        ParticleRecord.run_id.in_(run_ids),
        diameter.is_not(None),
    )
    count_value, minimum_value, mean_value, maximum_value = session.execute(
        select(
            func.count(diameter),
            func.min(diameter),
            func.avg(diameter),
            func.max(diameter),
        ).where(*predicate)
    ).one()
    observation_count = int(count_value)
    if (
        observation_count <= 0
        or minimum_value is None
        or mean_value is None
        or maximum_value is None
    ):
        return _insufficient_result(
            tool_name="describe_distribution",
            arguments=arguments,
            reason="目标指标没有可用观测值",
            outcome="insufficient_data",
            warnings=scope.warnings,
        )

    minimum = float(minimum_value)
    maximum = float(maximum_value)
    median_value = _sql_percentile(
        session,
        diameter,
        predicate,
        count=observation_count,
        fraction=0.5,
    )
    aggregates: dict[str, Any] = {
        "count": observation_count,
        "min": minimum,
        "q1": _sql_percentile(
            session,
            diameter,
            predicate,
            count=observation_count,
            fraction=0.25,
        ),
        "median": median_value,
        "mean": float(mean_value),
        "q3": _sql_percentile(
            session,
            diameter,
            predicate,
            count=observation_count,
            fraction=0.75,
        ),
        "max": maximum,
        "histogram": _sql_histogram(
            session,
            diameter,
            predicate,
            low=minimum,
            high=maximum,
            bins=intent.bins,
            count=observation_count,
        ),
    }
    rows = _sample_particle_evidence(
        session,
        diameter,
        predicate,
        scales=scales,
        diameter_unit=diameter_unit,
        total=observation_count,
        limit=evidence_limit,
    )
    arguments["evidence_row_limit"] = evidence_limit
    aggregates["evidence_rows_returned"] = len(rows)
    aggregates["evidence_rows_total"] = observation_count
    aggregates["evidence_sampling"] = "even_stride_by_run_instance"
    warnings = list(scope.warnings)
    if len(rows) < observation_count:
        warnings.append(
            "particle evidence rows were deterministically sampled: "
            f"returned {len(rows)} of {observation_count} observations"
        )
    unit = _comparison_metric_unit(intent.metric or _Metric.MEAN_DIAMETER_PX, diameter_unit)
    return _success_result(
        tool_name="describe_distribution",
        arguments=arguments,
        rows=rows,
        aggregates=aggregates,
        units={
            str(arguments["metric"]): unit,
            "min": unit,
            "q1": unit,
            "median": unit,
            "mean": unit,
            "q3": unit,
            "max": unit,
        },
        sources=run_ids,
        warnings=tuple(dict.fromkeys(warnings)),
        answer=(
            f"{_Metric.MEAN_DIAMETER_PX.value} 共 {observation_count} 个观测；中位数 "
            f"{median_value:.4g} {unit}，范围 {minimum:.4g}–{maximum:.4g} {unit}。"
        ),
    )


def _particle_diameter_expression(
    runs: tuple[_RunDatum, ...],
    diameter_unit: str,
) -> SQLColumnExpression[Any]:
    if diameter_unit == "px":
        return ParticleRecord.equivalent_diameter_px
    scaled_cases = [
        (
            ParticleRecord.run_id == run.run_id,
            ParticleRecord.equivalent_diameter_px * run.scale_nm_per_pixel,
        )
        for run in runs
        if run.scale_nm_per_pixel is not None
    ]
    return case(*scaled_cases, else_=ParticleRecord.equivalent_diameter_nm)


def _sql_percentile(
    session: Session,
    expression: SQLColumnExpression[Any],
    predicate: tuple[SQLColumnExpression[bool], ...],
    *,
    count: int,
    fraction: float,
) -> float:
    position = (count - 1) * fraction
    lower_index = int(position)
    upper_index = min(lower_index + 1, count - 1)
    values = [
        float(value)
        for value in session.scalars(
            select(expression)
            .where(*predicate)
            .order_by(
                expression,
                ParticleRecord.run_id,
                ParticleRecord.instance_index,
                ParticleRecord.particle_id,
            )
            .offset(lower_index)
            .limit(upper_index - lower_index + 1)
        ).all()
    ]
    if not values:
        raise ValueError("percentile query returned no observations")
    weight = position - lower_index
    upper = values[-1]
    return values[0] * (1 - weight) + upper * weight


def _sql_histogram(
    session: Session,
    expression: SQLColumnExpression[Any],
    predicate: tuple[SQLColumnExpression[bool], ...],
    *,
    low: float,
    high: float,
    bins: int,
    count: int,
) -> list[dict[str, float | int]]:
    if low == high:
        return [{"lower": low, "upper": high, "count": count}]
    width = (high - low) / bins
    raw_bucket = cast((expression - low) / width, Integer)
    bucket = case(
        (raw_bucket < 0, 0),
        (raw_bucket >= bins, bins - 1),
        else_=raw_bucket,
    )
    grouped = {
        int(index): int(bucket_count)
        for index, bucket_count in session.execute(
            select(bucket, func.count()).where(*predicate).group_by(bucket)
        ).all()
    }
    return [
        {
            "lower": low + index * width,
            "upper": low + (index + 1) * width,
            "count": grouped.get(index, 0),
        }
        for index in range(bins)
    ]


def _sample_particle_evidence(
    session: Session,
    expression: SQLColumnExpression[Any],
    predicate: tuple[SQLColumnExpression[bool], ...],
    *,
    scales: dict[str, float | None],
    diameter_unit: str,
    total: int,
    limit: int,
) -> list[dict[str, Any]]:
    stride = max(1, ceil(total / limit))
    ordinal = (
        func.row_number().over(
            order_by=(
                ParticleRecord.run_id,
                ParticleRecord.instance_index,
                ParticleRecord.particle_id,
            )
        )
        - 1
    ).label("sample_ordinal")
    candidates = (
        select(
            ParticleRecord.run_id.label("run_id"),
            ParticleRecord.particle_id.label("particle_id"),
            ParticleRecord.equivalent_diameter_px.label("diameter_px"),
            ParticleRecord.equivalent_diameter_nm.label("diameter_nm"),
            expression.label("distribution_value"),
            ordinal,
        )
        .where(*predicate)
        .subquery()
    )
    sampled = session.execute(
        select(candidates)
        .where(candidates.c.sample_ordinal % stride == 0)
        .order_by(candidates.c.sample_ordinal)
        .limit(limit)
    ).all()
    rows: list[dict[str, Any]] = []
    for row in sampled:
        run_id = str(row.run_id)
        diameter_px = float(row.diameter_px)
        stored_nm = float(row.diameter_nm) if row.diameter_nm is not None else None
        physical_nm = (
            float(row.distribution_value)
            if diameter_unit == "nm"
            else _particle_diameter_nm(
                diameter_px=diameter_px,
                diameter_nm=stored_nm,
                scale_nm_per_pixel=scales.get(run_id),
            )
        )
        rows.append(
            {
                "run_id": run_id,
                "particle_id": str(row.particle_id),
                "equivalent_diameter_px": diameter_px,
                "equivalent_diameter_nm": physical_nm,
            }
        )
    return rows


def _compare_models_result(
    scope: _Scope,
    arguments: dict[str, Any],
    intent: _Intent,
) -> DataQueryResult:
    if not scope.runs:
        return _no_completed_data("compare_models", arguments, scope)
    if intent.metric is None:
        raise AssertionError("resolved model comparison lacks metric")
    comparison_unit = _metric_comparison_unit(scope.runs, intent.metric)
    if comparison_unit is None:
        return _incomparable_metric_result(
            "compare_models",
            arguments,
            scope,
            intent.metric,
        )
    image_ids = {run.image_id for run in scope.runs}
    if len(image_ids) != 1:
        return _insufficient_result(
            tool_name="compare_models",
            arguments=arguments,
            reason="模型对比必须限定同一 image_id，并选择 2–3 个完成运行",
            outcome="insufficient_data",
            warnings=scope.warnings,
            sources=_source_ids(scope.runs),
            needs_clarification=True,
        )
    by_model: dict[str, _RunDatum] = {}
    for run in scope.runs:
        if run.model_id in by_model:
            return _insufficient_result(
                tool_name="compare_models",
                arguments=arguments,
                reason="同一模型存在多个运行；请用 run_ids 为每个模型选择一个运行",
                outcome="insufficient_data",
                warnings=scope.warnings,
                sources=_source_ids(scope.runs),
                needs_clarification=True,
            )
        by_model[run.model_id] = run
    if not 2 <= len(by_model) <= 3:
        return _insufficient_result(
            tool_name="compare_models",
            arguments=arguments,
            reason="模型对比需要同一图像的 2–3 个不同模型运行",
            outcome="insufficient_data",
            warnings=scope.warnings,
            sources=_source_ids(scope.runs),
            needs_clarification=True,
        )
    rows = []
    for _model_id, run in sorted(by_model.items()):
        value = _run_metric(
            run,
            intent.metric,
            comparison_unit=comparison_unit,
        )
        if value is None:
            continue
        rows.append(
            {
                **_identity_row(run),
                "metric": intent.metric.value,
                "value": value,
                "mean_equivalent_diameter_nm": run.mean_diameter_nm,
            }
        )
    if len(rows) < 2:
        return _insufficient_result(
            tool_name="compare_models",
            arguments=arguments,
            reason="至少两个模型运行需要具备目标指标",
            outcome="insufficient_data",
            warnings=scope.warnings,
        )
    rows.sort(key=lambda row: (-float(row["value"]), str(row["model_id"])))
    metric_name = _metric_name(intent.metric, comparison_unit)
    for row in rows:
        row["metric"] = metric_name
    arguments.update(
        {
            "image_id": next(iter(image_ids)),
            "run_ids": [str(row["run_id"]) for row in rows],
            "metric": metric_name,
        }
    )
    unit = _comparison_metric_unit(intent.metric, comparison_unit)
    return _success_result(
        tool_name="compare_models",
        arguments=arguments,
        rows=rows,
        aggregates={"model_count": len(rows)},
        units={
            metric_name: unit,
            "value": unit,
        },
        sources=[str(row["run_id"]) for row in rows],
        warnings=scope.warnings,
        answer=(
            f"已比较同一图像的 {len(rows)} 个模型运行；按 {metric_name}，"
            f"最高的是 {rows[0]['model_id']}。"
        ),
    )


def _group_key(run: _RunDatum, group_by: _GroupBy) -> str | None:
    if group_by == _GroupBy.IMAGE:
        return run.image_id
    if group_by == _GroupBy.SAMPLE:
        return run.sample_id or None
    return run.material_formula or run.material_name


def _group_metric(
    runs: list[_RunDatum],
    metric: _Metric,
    *,
    comparison_unit: str | None = None,
) -> float | None:
    if metric == _Metric.PARTICLE_COUNT:
        return float(sum(run.particle_count for run in runs))
    if metric == _Metric.NUMBER_DENSITY_PX2:
        if comparison_unit == "um^-2":
            physical_areas = [_physical_roi_area_um2(run) for run in runs]
            if any(area is None for area in physical_areas):
                return None
            total_area_um2 = sum(area for area in physical_areas if area is not None)
            if total_area_um2 <= 0:
                return None
            return sum(run.particle_count for run in runs) / total_area_um2
        total_area_px = sum(run.roi_area_px for run in runs)
        if total_area_px <= 0:
            return None
        return sum(run.particle_count for run in runs) / total_area_px
    if metric == _Metric.MEAN_DIAMETER_PX:
        if comparison_unit == "nm":
            return _weighted_mean((run.mean_diameter_nm, _diameter_nm_weight(run)) for run in runs)
        return _weighted_mean((run.mean_diameter_px, _diameter_px_weight(run)) for run in runs)
    if metric == _Metric.PERIMETER_DENSITY_PX:
        if comparison_unit == "um^-1":
            physical_areas = [_physical_roi_area_um2(run) for run in runs]
            if any(area is None for area in physical_areas):
                return None
            total_area_um2 = sum(area for area in physical_areas if area is not None)
            if total_area_um2 <= 0:
                return None
            total_perimeter_um = sum(
                _physical_perimeter_total_um(run) or 0.0 for run in runs
            )
            return total_perimeter_um / total_area_um2
        total_area_px = sum(run.roi_area_px for run in runs)
        if total_area_px <= 0:
            return None
        return (
            sum(run.perimeter_density_px * run.roi_area_px for run in runs)
            / total_area_px
        )
    total_area = sum(run.roi_area_px for run in runs)
    if total_area <= 0:
        return None
    return sum(run.coverage_ratio * run.roi_area_px for run in runs) / total_area


def _ranking_aggregation(metric: _Metric) -> str:
    if metric == _Metric.PARTICLE_COUNT:
        return "sum_across_runs"
    if metric == _Metric.NUMBER_DENSITY_PX2:
        return "total_particle_count_over_total_roi_area"
    if metric == _Metric.MEAN_DIAMETER_PX:
        return "particle_weighted_mean"
    if metric == _Metric.PERIMETER_DENSITY_PX:
        return "total_perimeter_over_total_roi_area"
    return "roi_area_weighted_mean"


def _ranking_weight_total(
    runs: list[_RunDatum],
    metric: _Metric,
    *,
    comparison_unit: str | None,
) -> float | int:
    if metric == _Metric.PARTICLE_COUNT:
        return len(runs)
    if metric == _Metric.NUMBER_DENSITY_PX2:
        if comparison_unit == "um^-2":
            return sum(_physical_roi_area_um2(run) or 0.0 for run in runs)
        return sum(run.roi_area_px for run in runs)
    if metric == _Metric.MEAN_DIAMETER_PX:
        if comparison_unit == "nm":
            return sum(_diameter_nm_weight(run) for run in runs)
        return sum(_diameter_px_weight(run) for run in runs)
    if metric == _Metric.PERIMETER_DENSITY_PX:
        if comparison_unit == "um^-1":
            return sum(_physical_roi_area_um2(run) or 0.0 for run in runs)
        return sum(run.roi_area_px for run in runs)
    return sum(run.roi_area_px for run in runs)


def _ranking_weight_unit(metric: _Metric, comparison_unit: str | None) -> str:
    if metric == _Metric.NUMBER_DENSITY_PX2:
        return "um^2" if comparison_unit == "um^-2" else "px^2"
    if metric == _Metric.MEAN_DIAMETER_PX:
        return "particle_count"
    if metric in {_Metric.COVERAGE_RATIO, _Metric.PERIMETER_DENSITY_PX}:
        if metric == _Metric.PERIMETER_DENSITY_PX and comparison_unit == "um^-1":
            return "um^2"
        return "px^2"
    return "run_count"


def _run_metric(
    run: _RunDatum,
    metric: _Metric,
    *,
    comparison_unit: str | None = None,
) -> float | None:
    if metric == _Metric.PARTICLE_COUNT:
        return float(run.particle_count)
    if metric == _Metric.NUMBER_DENSITY_PX2:
        if comparison_unit == "um^-2":
            return run.number_density_um2
        return run.number_density_px2
    if metric == _Metric.MEAN_DIAMETER_PX:
        if comparison_unit == "nm":
            return run.mean_diameter_nm
        return run.mean_diameter_px
    if metric == _Metric.PERIMETER_DENSITY_PX:
        if comparison_unit == "um^-1":
            return run.perimeter_density_um
        return run.perimeter_density_px
    return run.coverage_ratio


def _statistic(values: list[float], statistic: str) -> float:
    if statistic == "median":
        return float(median(values))
    if statistic == "min":
        return min(values)
    if statistic == "max":
        return max(values)
    return float(fmean(values))


def _percentile(ordered: list[float], fraction: float) -> float:
    if not ordered:
        raise ValueError("percentile requires at least one value")
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def _histogram(values: list[float], bins: int) -> list[dict[str, float | int]]:
    low, high = values[0], values[-1]
    if low == high:
        return [{"lower": low, "upper": high, "count": len(values)}]
    width = (high - low) / bins
    counts = [0] * bins
    for value in values:
        index = min(int((value - low) / width), bins - 1)
        counts[index] += 1
    return [
        {
            "lower": low + index * width,
            "upper": low + (index + 1) * width,
            "count": count,
        }
        for index, count in enumerate(counts)
    ]


def _metric_unit(metric: _Metric) -> str:
    if metric == _Metric.PARTICLE_COUNT:
        return "count"
    if metric == _Metric.NUMBER_DENSITY_PX2:
        return "px^-2"
    if metric == _Metric.MEAN_DIAMETER_PX:
        return "px"
    if metric == _Metric.PERIMETER_DENSITY_PX:
        return "px^-1"
    return "ratio"


def _comparison_metric_unit(metric: _Metric, comparison_unit: str | None) -> str:
    if metric in {
        _Metric.MEAN_DIAMETER_PX,
        _Metric.NUMBER_DENSITY_PX2,
        _Metric.PERIMETER_DENSITY_PX,
    }:
        return comparison_unit or _metric_unit(metric)
    return _metric_unit(metric)


def _metric_comparison_unit(
    runs: tuple[_RunDatum, ...],
    metric: _Metric,
) -> str | None:
    if metric == _Metric.NUMBER_DENSITY_PX2:
        if runs and all(_physical_density_complete(run) for run in runs):
            return "um^-2"
        if len({run.image_id for run in runs}) == 1 or _has_one_known_scale(runs):
            return "px^-2"
        return None
    if metric == _Metric.PERIMETER_DENSITY_PX:
        if runs and all(_physical_perimeter_density_complete(run) for run in runs):
            return "um^-1"
        if len({run.image_id for run in runs}) == 1 or _has_one_known_scale(runs):
            return "px^-1"
        return None
    if metric != _Metric.MEAN_DIAMETER_PX:
        return _metric_unit(metric)
    if all(_physical_diameter_complete(run) for run in runs if _diameter_px_weight(run) > 0):
        return "nm"
    if len({run.image_id for run in runs}) == 1 or _has_one_known_scale(runs):
        return "px"
    return None


def _metric_name(metric: _Metric, comparison_unit: str | None) -> str:
    if metric == _Metric.MEAN_DIAMETER_PX and comparison_unit == "nm":
        return "mean_equivalent_diameter_nm"
    if metric == _Metric.NUMBER_DENSITY_PX2 and comparison_unit == "um^-2":
        return "number_density_um2"
    if metric == _Metric.PERIMETER_DENSITY_PX and comparison_unit == "um^-1":
        return "perimeter_density_um"
    return metric.value


def _has_one_known_scale(runs: Iterable[_RunDatum]) -> bool:
    values = [run.scale_nm_per_pixel for run in runs]
    return bool(values) and all(value is not None for value in values) and len(set(values)) == 1


def _ambiguous_images(scope: _Scope) -> dict[str, list[str]]:
    grouped: dict[str, list[str]] = {}
    for run in scope.runs:
        grouped.setdefault(run.image_id, []).append(run.run_id)
    return {
        image_id: sorted(run_ids)
        for image_id, run_ids in sorted(grouped.items())
        if len(run_ids) > 1
    }


def _incomparable_metric_result(
    tool_name: str,
    arguments: dict[str, Any],
    scope: _Scope,
    metric: _Metric,
) -> DataQueryResult:
    if metric == _Metric.NUMBER_DENSITY_PX2:
        return _incomparable_density_result(tool_name, arguments, scope)
    if metric == _Metric.PERIMETER_DENSITY_PX:
        return _incomparable_perimeter_density_result(tool_name, arguments, scope)
    return _incomparable_diameter_result(tool_name, arguments, scope)


def _incomparable_diameter_result(
    tool_name: str,
    arguments: dict[str, Any],
    scope: _Scope,
) -> DataQueryResult:
    return _insufficient_result(
        tool_name=tool_name,
        arguments=arguments,
        reason=("跨图像粒径比较缺少统一物理尺度；请补充比例尺，或限定同一图像的 run_ids"),
        outcome="insufficient_data",
        warnings=scope.warnings,
        sources=_source_ids(scope.runs),
        needs_clarification=True,
    )


def _incomparable_density_result(
    tool_name: str,
    arguments: dict[str, Any],
    scope: _Scope,
) -> DataQueryResult:
    return _insufficient_result(
        tool_name=tool_name,
        arguments=arguments,
        reason=(
            "跨图像颗粒数密度比较缺少可统一的物理面积尺度；请补充比例尺，"
            "或限定同一图像/相同像素尺度的 run_ids"
        ),
        outcome="insufficient_data",
        warnings=scope.warnings,
        sources=_source_ids(scope.runs),
        needs_clarification=True,
    )


def _incomparable_perimeter_density_result(
    tool_name: str,
    arguments: dict[str, Any],
    scope: _Scope,
) -> DataQueryResult:
    return _insufficient_result(
        tool_name=tool_name,
        arguments=arguments,
        reason=(
            "跨图像周长密度比较缺少可统一的物理长度尺度；请补充比例尺，"
            "或限定同一图像/相同像素尺度的 run_ids"
        ),
        outcome="insufficient_data",
        warnings=scope.warnings,
        sources=_source_ids(scope.runs),
        needs_clarification=True,
    )


def _physical_roi_area_um2(run: _RunDatum) -> float | None:
    if run.scale_nm_per_pixel is None or run.roi_area_px <= 0:
        return None
    return run.roi_area_px * (run.scale_nm_per_pixel / 1000.0) ** 2


def _physical_density_complete(run: _RunDatum) -> bool:
    return run.number_density_um2 is not None and _physical_roi_area_um2(run) is not None


def _physical_perimeter_density_complete(run: _RunDatum) -> bool:
    return (
        run.perimeter_density_um is not None
        and _physical_roi_area_um2(run) is not None
    )


def _physical_perimeter_total_um(run: _RunDatum) -> float | None:
    area_um2 = _physical_roi_area_um2(run)
    if run.perimeter_density_um is None or area_um2 is None:
        return None
    return run.perimeter_density_um * area_um2


def _diameter_px_weight(run: _RunDatum) -> int:
    if run.stored_particle_rows != run.particle_count:
        return 0
    return run.stored_particle_rows


def _diameter_nm_weight(run: _RunDatum) -> int:
    if run.stored_particle_rows != run.particle_count:
        return 0
    if run.stored_particle_rows:
        if run.scale_nm_per_pixel is not None:
            return run.stored_particle_rows
        if run.stored_particle_nm_rows == run.stored_particle_rows:
            return run.stored_particle_rows
        return 0
    return 0


def _physical_diameter_complete(run: _RunDatum) -> bool:
    if run.stored_particle_rows != run.particle_count:
        return False
    if run.particle_count <= 0:
        return True
    if run.mean_diameter_nm is None:
        return False
    return (
        run.scale_nm_per_pixel is not None
        or run.stored_particle_nm_rows == run.stored_particle_rows
    )


def _particle_diameter_nm(
    *,
    diameter_px: float,
    diameter_nm: float | None,
    scale_nm_per_pixel: float | None,
) -> float | None:
    if scale_nm_per_pixel is not None:
        return diameter_px * scale_nm_per_pixel
    return diameter_nm


def _weighted_mean(values: Iterable[tuple[float | None, int]]) -> float | None:
    numerator = 0.0
    denominator = 0
    for value, weight in values:
        if value is None or weight <= 0:
            continue
        numerator += value * weight
        denominator += weight
    return numerator / denominator if denominator else None


def _identity_row(run: _RunDatum) -> dict[str, Any]:
    return {
        "run_id": run.run_id,
        "image_id": run.image_id,
        "filename": run.filename,
        "sample_id": run.sample_id,
        "material_name": run.material_name,
        "material_formula": run.material_formula,
        "model_id": run.model_id,
        "status": run.status,
        "quality_status": run.quality_status,
    }


def _overview_row(run: _RunDatum) -> dict[str, Any]:
    return {
        **_identity_row(run),
        "particle_count": run.particle_count,
        "roi_area_px": run.roi_area_px,
        "number_density_px2": run.number_density_px2,
        "number_density_um2": run.number_density_um2,
        "mean_equivalent_diameter_px": run.mean_diameter_px,
        "mean_equivalent_diameter_nm": run.mean_diameter_nm,
        "coverage_ratio": run.coverage_ratio,
        "perimeter_density_px": run.perimeter_density_px,
        "perimeter_density_um": run.perimeter_density_um,
    }


def _source_ids(runs: Iterable[_RunDatum]) -> list[str]:
    return sorted({run.run_id for run in runs})


def _tool_name(intent: _Intent) -> str:
    if intent.kind == _IntentKind.OVERVIEW:
        return "get_job_overview"
    if intent.kind in {
        _IntentKind.PARTICLE_COUNT,
        _IntentKind.NUMBER_DENSITY,
        _IntentKind.MEAN_DIAMETER,
        _IntentKind.COVERAGE,
        _IntentKind.PERIMETER_DENSITY,
    }:
        return "get_metric"
    if intent.kind == _IntentKind.REVIEW:
        return "find_review"
    if intent.kind == _IntentKind.ANOMALIES:
        return "find_anomalies"
    if intent.kind == _IntentKind.RANK:
        return "rank_samples"
    if intent.kind == _IntentKind.COMPARE_GROUPS:
        return "compare_groups"
    if intent.kind == _IntentKind.DISTRIBUTION:
        return "describe_distribution"
    if intent.kind == _IntentKind.COMPARE_MODELS:
        return "compare_models"
    return "resolve_data_intent"


def _no_completed_data(
    tool_name: str,
    arguments: dict[str, Any],
    scope: _Scope,
) -> DataQueryResult:
    return _insufficient_result(
        tool_name=tool_name,
        arguments=arguments,
        reason="筛选范围内没有带持久化结果的完成运行",
        outcome="insufficient_data",
        warnings=scope.warnings,
    )


def _success_result(
    *,
    tool_name: str,
    arguments: dict[str, Any],
    rows: list[dict[str, Any]],
    aggregates: dict[str, Any],
    units: dict[str, str],
    sources: list[str],
    warnings: tuple[str, ...],
    answer: str,
) -> DataQueryResult:
    evidence = ToolEvidence(
        tool_name=tool_name,
        validated_arguments=arguments,
        rows=rows,
        aggregates=aggregates,
        units=units,
        source_run_ids=sources,
        quality_warnings=list(warnings),
    )
    call = ToolCallLog(
        tool_name=tool_name,
        arguments=arguments,
        outcome="success",
        source_run_ids=sources,
    )
    return DataQueryResult(
        answer=answer,
        evidence=(evidence,),
        tool_calls=(call,),
        confidence="high" if not warnings else "medium",
    )


def _insufficient_result(
    *,
    tool_name: str,
    arguments: dict[str, Any],
    reason: str,
    outcome: str,
    warnings: tuple[str, ...] = (),
    sources: list[str] | None = None,
    needs_clarification: bool = False,
) -> DataQueryResult:
    source_ids = sources or []
    evidence = ToolEvidence(
        tool_name=tool_name,
        validated_arguments=arguments,
        rows=[],
        aggregates={},
        units={},
        source_run_ids=source_ids,
        quality_warnings=list(warnings),
    )
    call = ToolCallLog(
        tool_name=tool_name,
        arguments=arguments,
        outcome="error" if outcome == "error" else "insufficient_data",
        source_run_ids=source_ids,
    )
    return DataQueryResult(
        answer=reason,
        evidence=(evidence,),
        tool_calls=(call,),
        confidence="low",
        limitations=(reason,),
        needs_clarification=needs_clarification,
        outcome_code="INSUFFICIENT_EVIDENCE",
    )
