"""Pure model-catalog helpers shared by the Streamlit UI and tests."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal


@dataclass(frozen=True, slots=True)
class ModelAvailability:
    """Fail-closed interpretation of one model registry health record."""

    runnable: bool
    severity: Literal["success", "warning", "error"]
    message: str


def model_filter_query(
    *,
    family: str | None,
    variant: str | None,
    quality_tier: str | None,
    status: str | None,
    material: str | None,
) -> dict[str, str | None]:
    """Normalize optional UI filters for the existing ``GET /models`` query."""

    normalized_material = material.strip() if material else None
    return {
        "status": status,
        "family": family,
        "variant": variant,
        "quality_tier": quality_tier,
        "material": normalized_material or None,
    }


def model_availability(model: Mapping[str, Any]) -> ModelAvailability:
    """Describe registry availability without turning missing evidence into readiness."""

    status = str(model.get("status") or "unknown").strip().casefold()
    raw_error = model.get("health_error")
    health_error = str(raw_error).strip() if raw_error is not None else ""

    if status == "ready" and not health_error:
        return ModelAvailability(
            runnable=True,
            severity="success",
            message="模型已就绪，可在人工确认后用于新运行。",
        )
    if status == "ready":
        return ModelAvailability(
            runnable=False,
            severity="warning",
            message=(
                "后端将模型标记为就绪，但同时返回健康错误；工作台将按不可运行处理。"
                f"原因：{health_error}"
            ),
        )
    if status == "loading":
        reason = health_error or "模型仍在加载，后端未提供进一步原因。"
        return ModelAvailability(
            runnable=False,
            severity="warning",
            message=f"模型尚不可运行。原因：{reason}",
        )

    reason = health_error or "后端未提供具体健康原因。"
    return ModelAvailability(
        runnable=False,
        severity="error",
        message=f"模型当前不可运行。原因：{reason}",
    )


def model_is_runnable(model: Mapping[str, Any]) -> bool:
    """Return whether a model may be exposed as a run candidate."""

    return model_availability(model).runnable


__all__ = [
    "ModelAvailability",
    "model_availability",
    "model_filter_query",
    "model_is_runnable",
]
