"""Explainable rule-based quality gate for heterogeneous model outputs."""

from dataclasses import dataclass

from app.analysis.config import QualityGateConfig
from app.analysis.postprocessing import NormalizedInstance
from app.contracts.analyses import QualityReportDTO
from app.contracts.enums import QualityStatus


@dataclass(frozen=True, slots=True)
class QualityInputs:
    roi_area_px: int
    foreground_area_px: int
    instances: list[NormalizedInstance]
    minimum_area_px: int
    validation_warnings: list[str]
    candidate_instance_count: int | None = None
    boundary_instance_count: int | None = None


def evaluate(inputs: QualityInputs, config: QualityGateConfig) -> QualityReportDTO:
    if inputs.roi_area_px <= 0:
        raise ValueError("roi_area_px must be positive")
    foreground_ratio = inputs.foreground_area_px / inputs.roi_area_px
    confidences = [item.confidence for item in inputs.instances if item.confidence is not None]
    mean_confidence = sum(confidences) / len(confidences) if confidences else None
    small_cutoff = max(inputs.minimum_area_px * 2, 1)
    fragment_ratio = (
        sum(item.area_px < small_cutoff for item in inputs.instances) / len(inputs.instances)
        if inputs.instances
        else 0.0
    )
    candidate_count = (
        inputs.candidate_instance_count
        if inputs.candidate_instance_count is not None
        else len(inputs.instances)
    )
    boundary_count = (
        inputs.boundary_instance_count
        if inputs.boundary_instance_count is not None
        else sum(item.touches_roi_boundary for item in inputs.instances)
    )
    if candidate_count < 0 or boundary_count < 0 or boundary_count > candidate_count:
        raise ValueError("invalid prefilter instance diagnostics")
    edge_touch_ratio = boundary_count / candidate_count if candidate_count else 0.0

    review: list[str] = []
    warnings = list(inputs.validation_warnings)
    recommendations: list[str] = []
    if foreground_ratio <= config.foreground_ratio_review_low:
        review.append("foreground_ratio_too_low")
        recommendations.append("检查模型、阈值与分析区域")
    elif foreground_ratio >= config.foreground_ratio_review_high:
        review.append("foreground_ratio_too_high")
        recommendations.append("检查阈值或选择更适合的模型")
    elif foreground_ratio >= config.foreground_ratio_warn_high:
        warnings.append("foreground_ratio_high")
    if mean_confidence is not None and mean_confidence < config.confidence_warn_below:
        warnings.append("model_confidence_low")
        recommendations.append("复核分割叠加图")
    if fragment_ratio > config.fragment_ratio_warn_above:
        warnings.append("small_fragment_ratio_high")
        recommendations.append("提高 min_area_px 或选择小颗粒专用模型")
    if edge_touch_ratio > config.edge_touch_ratio_warn_above:
        warnings.append("roi_edge_truncation")
        recommendations.append("扩大选框或使用全图模式")

    if review:
        status = QualityStatus.REVIEW_REQUIRED
    elif warnings:
        status = QualityStatus.WARN
    else:
        status = QualityStatus.PASS
    return QualityReportDTO(
        status=status,
        reasons=[*review, *warnings],
        metrics={
            "foreground_ratio": foreground_ratio,
            "mean_confidence": mean_confidence,
            "small_fragment_ratio": fragment_ratio,
            "edge_touch_ratio": edge_touch_ratio,
            "candidate_instance_count": candidate_count,
            "boundary_instance_count": boundary_count,
            "excluded_border_instance_count": candidate_count - len(inputs.instances),
        },
        recommendations=list(dict.fromkeys(recommendations)),
    )
