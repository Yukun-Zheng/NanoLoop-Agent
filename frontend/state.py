"""Pure state and validation helpers for the Streamlit workbench."""

from __future__ import annotations

import json
import os
import re
from collections.abc import Iterable, Mapping, MutableMapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urljoin, urlparse

DEFAULT_API_BASE_URL = "http://127.0.0.1:8000"
TERMINAL_RUN_STATUSES = frozenset({"COMPLETED", "COMPLETED_WITH_WARNINGS", "FAILED"})
EXPORTABLE_RUN_STATUSES = frozenset({"COMPLETED", "COMPLETED_WITH_WARNINGS"})
RUNNING_RUN_STATUSES = frozenset(
    {
        "CREATED",
        "VALIDATING",
        "READY_FOR_CONFIGURATION",
        "QUEUED",
        "PREPROCESSING",
        "SEGMENTING",
        "POSTPROCESSING",
        "QUALITY_CHECKING",
        "ANALYZING",
        "AGGREGATING",
    }
)


@dataclass(frozen=True, slots=True)
class HealthRollup:
    status: str
    label: str
    detail: str
    unhealthy_components: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class BoxValidation:
    boxes: tuple[dict[str, Any], ...]
    errors: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class PreviewArtifact:
    key: str
    download_url: str


@dataclass(frozen=True, slots=True)
class KnowledgeDocumentToggle:
    enabled: bool
    button_label: str
    completed_label: str


def default_state() -> dict[str, Any]:
    """Return fresh session defaults; mutable values are never shared."""

    return {
        "api_base_url": os.getenv("NANOLOOP_API_BASE_URL", DEFAULT_API_BASE_URL),
        "api_timeout_seconds": 30.0,
        "health": None,
        "health_checked_at": None,
        "active_job_id": "",
        "job_detail": None,
        "box_sets": {},
        "models": [],
        "models_loaded": False,
        "model_catalog_filters": {},
        "model_catalog_selected_id": "",
        "recommendations": [],
        "selected_image_ids": [],
        "selected_model_ids": [],
        "run_ids": [],
        "runs": {},
        "query_history": [],
        "knowledge_documents": [],
        "knowledge_status_notice": None,
        "last_ingest_report": None,
        "last_reindex_report": None,
        "last_export": None,
        "last_error": None,
        "image_preview": None,
        "roi_drafts": {},
        "roi_canvas_events": {},
        "comparison_previews": {},
        "comparison_downloads": {},
        "result_layer_previews": {},
        "navigation": "Connection",
    }


def ensure_session_state(state: MutableMapping[str, Any]) -> MutableMapping[str, Any]:
    """Add missing workbench keys without overwriting Streamlit session state."""

    for key, value in default_state().items():
        if key not in state:
            if isinstance(value, dict):
                state[key] = dict(value)
            elif isinstance(value, list):
                state[key] = list(value)
            else:
                state[key] = value
    return state


def knowledge_document_toggle(status: object) -> KnowledgeDocumentToggle | None:
    """Return the only valid enable/disable transition exposed by the catalogue UI."""

    normalized = str(status).strip().casefold()
    if normalized == "ready":
        return KnowledgeDocumentToggle(
            enabled=False,
            button_label="禁用文档",
            completed_label="已禁用",
        )
    if normalized == "disabled":
        return KnowledgeDocumentToggle(
            enabled=True,
            button_label="启用文档",
            completed_label="已启用",
        )
    return None


def normalize_base_url(value: str) -> str:
    """Validate and normalize an HTTP(S) API prefix."""

    normalized = value.strip().rstrip("/")
    parsed = urlparse(normalized)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError("API base URL must be an absolute http:// or https:// URL")
    if parsed.query or parsed.fragment:
        raise ValueError("API base URL cannot contain a query string or fragment")
    return normalized


def absolute_api_url(base_url: str, path_or_url: str | None) -> str | None:
    if not path_or_url:
        return None
    parsed = urlparse(path_or_url)
    if parsed.scheme in {"http", "https"}:
        return path_or_url
    normalized_base = normalize_base_url(base_url)
    if path_or_url.startswith("/"):
        parsed_base = urlparse(normalized_base)
        origin = f"{parsed_base.scheme}://{parsed_base.netloc}/"
        return urljoin(origin, path_or_url.lstrip("/"))
    return urljoin(normalized_base + "/", path_or_url)


def to_plain(value: Any) -> Any:
    """Convert API result/Pydantic objects into JSON-like values."""

    if hasattr(value, "data") and hasattr(value, "request_id"):
        return to_plain(value.data)
    if hasattr(value, "model_dump"):
        return to_plain(value.model_dump(mode="json"))
    if isinstance(value, Mapping):
        return {str(key): to_plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_plain(item) for item in value]
    return value


def as_dict(value: Any) -> dict[str, Any]:
    plain = to_plain(value)
    if not isinstance(plain, dict):
        raise TypeError("API response data must be an object")
    return plain


def health_rollup(health: Mapping[str, Any] | None) -> HealthRollup:
    """Summarize health without masking individual degraded components."""

    if not health:
        return HealthRollup(
            status="unknown",
            label="Not checked",
            detail="Run a connection check before starting a workflow.",
            unhealthy_components=(),
        )
    component_names = ("service", "database", "model_registry", "rag_index")
    statuses: dict[str, str] = {}
    details: list[str] = []
    for name in component_names:
        component = health.get(name)
        if not isinstance(component, Mapping):
            statuses[name] = "unavailable"
            details.append(f"{name}: no health record")
            continue
        status = str(component.get("status", "unavailable")).casefold()
        statuses[name] = status
        detail = component.get("detail")
        if status != "healthy" and detail:
            details.append(f"{name}: {detail}")

    unavailable_core = any(statuses.get(name) == "unavailable" for name in ("service", "database"))
    unhealthy = tuple(name for name, status in statuses.items() if status != "healthy")
    if unavailable_core:
        status, label = "unavailable", "Core unavailable"
    elif unhealthy:
        status, label = "degraded", "Connected with limitations"
    else:
        status, label = "healthy", "Connected"
    return HealthRollup(
        status=status,
        label=label,
        detail="; ".join(details) or "All reported components are healthy.",
        unhealthy_components=unhealthy,
    )


def status_tone(status: str | None) -> str:
    normalized = (status or "unknown").casefold()
    if normalized in {"healthy", "ready", "pass", "completed", "success"}:
        return "good"
    if normalized in {
        "degraded",
        "warn",
        "review_required",
        "completed_with_warnings",
        "accepted",
        "indexing",
        "loading",
    }:
        return "warn"
    if normalized in {"unavailable", "failed", "error", "disabled"}:
        return "bad"
    if status and status.upper() in RUNNING_RUN_STATUSES:
        return "live"
    return "neutral"


def is_terminal_run(status: str | None) -> bool:
    return bool(status and status.upper() in TERMINAL_RUN_STATUSES)


def pollable_run_ids(runs: Mapping[str, Mapping[str, Any]]) -> list[str]:
    return [
        run_id for run_id, run in runs.items() if not is_terminal_run(str(run.get("status", "")))
    ]


def exportable_run_ids(runs: Iterable[Mapping[str, Any]]) -> list[str]:
    return [
        str(run["run_id"])
        for run in runs
        if run.get("run_id") and str(run.get("status", "")).upper() in EXPORTABLE_RUN_STATUSES
    ]


def comparable_runs_by_image(
    runs: Sequence[Mapping[str, Any]],
    *,
    minimum_runs: int = 2,
) -> dict[str, list[dict[str, Any]]]:
    """Group unique, successfully completed runs by image for fair comparison."""

    if minimum_runs < 1:
        raise ValueError("minimum_runs must be positive")
    grouped: dict[str, list[dict[str, Any]]] = {}
    seen_run_ids: set[str] = set()
    for run in runs:
        run_id = str(run.get("run_id") or "").strip()
        image_id = str(run.get("image_id") or "").strip()
        status = str(run.get("status") or "").upper()
        if (
            not run_id
            or not image_id
            or run_id in seen_run_ids
            or status not in EXPORTABLE_RUN_STATUSES
        ):
            continue
        seen_run_ids.add(run_id)
        grouped.setdefault(image_id, []).append(dict(run))
    return {
        image_id: image_runs
        for image_id, image_runs in grouped.items()
        if len(image_runs) >= minimum_runs
    }


def select_comparison_runs(
    runs: Sequence[Mapping[str, Any]],
    *,
    image_id: str,
    run_ids: Sequence[str],
) -> tuple[dict[str, Any], ...]:
    """Validate and resolve a 2-3 run comparison without silently changing scope."""

    selected_ids = [str(run_id).strip() for run_id in run_ids]
    if not 2 <= len(selected_ids) <= 3:
        raise ValueError("Select two or three runs for comparison")
    if len(set(selected_ids)) != len(selected_ids) or any(not value for value in selected_ids):
        raise ValueError("Comparison run IDs must be unique and non-empty")
    target_image_id = image_id.strip()
    if not target_image_id:
        raise ValueError("Select an image for comparison")

    candidates = comparable_runs_by_image(runs, minimum_runs=1).get(target_image_id, [])
    by_id = {str(run["run_id"]): run for run in candidates}
    missing = [run_id for run_id in selected_ids if run_id not in by_id]
    if missing:
        raise ValueError(
            "Comparison runs must be completed runs from the selected image: " + ", ".join(missing)
        )
    return tuple(dict(by_id[run_id]) for run_id in selected_ids)


def preferred_preview_artifact(run: Mapping[str, Any]) -> PreviewArtifact | None:
    """Choose a visual REST artifact, preferring overlay over derived mask views."""

    artifacts = run.get("artifacts")
    if not isinstance(artifacts, Mapping):
        return None
    for key in ("overlay_url", "labeled_particles_url", "mask_url"):
        value = artifacts.get(key)
        if isinstance(value, str) and value.strip():
            return PreviewArtifact(key=key, download_url=value.strip())
    return None


def parse_aliases(value: str) -> list[str]:
    parts = re.split(r"[,;；，\n]+", value)
    return list(dict.fromkeys(part.strip() for part in parts if part.strip()))


def parse_json_object(value: str, *, field_name: str) -> dict[str, Any]:
    if not value.strip():
        return {}
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as error:
        raise ValueError(
            f"{field_name} must be valid JSON: line {error.lineno}, column {error.colno}"
        ) from error
    if not isinstance(parsed, dict):
        raise ValueError(f"{field_name} must be a JSON object")
    return parsed


def build_analysis_metadata(
    job_name: str,
    filenames: Sequence[str],
    drafts: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Validate project form values and build the frozen REST metadata object."""

    clean_name = job_name.strip()
    if not clean_name:
        raise ValueError("Project name is required")
    if not 1 <= len(filenames) <= 20:
        raise ValueError("Select between 1 and 20 images")
    if len(set(filenames)) != len(filenames):
        raise ValueError("Image filenames must be unique within a project")

    images: list[dict[str, Any]] = []
    for filename in filenames:
        values = drafts.get(filename, {})
        sample_id = str(values.get("sample_id", "")).strip()
        if not sample_id:
            raise ValueError(f"{filename}: sample ID is required")
        scale_mode = str(values.get("scale_mode", "pixel_only"))
        if scale_mode not in {"pixel_only", "nm_per_pixel"}:
            raise ValueError(f"{filename}: unsupported scale mode")
        raw_scale = values.get("scale_value")
        if scale_mode == "nm_per_pixel":
            if isinstance(raw_scale, bool) or not isinstance(raw_scale, (int, float)):
                raise ValueError(f"{filename}: nm/pixel scale is required")
            if float(raw_scale) <= 0:
                raise ValueError(f"{filename}: nm/pixel scale must be positive")
            scale: dict[str, Any] = {"mode": scale_mode, "value": float(raw_scale)}
        else:
            scale = {"mode": scale_mode, "value": None}

        conditions = values.get("experiment_conditions", {})
        if not isinstance(conditions, Mapping):
            raise ValueError(f"{filename}: experiment conditions must be an object")
        images.append(
            {
                "filename": filename,
                "sample_id": sample_id,
                "material_name": _optional_text(values.get("material_name")),
                "material_formula": _optional_text(values.get("material_formula")),
                "experiment_conditions": dict(conditions),
                "scale": scale,
            }
        )
    return {"job_name": clean_name, "images": images}


def rows_from_editor(value: Any) -> list[Mapping[str, Any]]:
    rows = value.to_dict(orient="records") if hasattr(value, "to_dict") else value
    if not isinstance(rows, list):
        raise TypeError("ROI editor must return row records")
    return [row for row in rows if isinstance(row, Mapping)]


def validate_box_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    width: int,
    height: int,
    max_boxes: int = 20,
    minimum_size_px: int = 32,
    valid_rect: Mapping[str, Any] | None = None,
    invalid_rects: Sequence[Mapping[str, Any]] | None = None,
) -> BoxValidation:
    """Validate half-open original-pixel boxes against the analysis ROI."""

    errors: list[str] = []
    boxes: list[dict[str, Any]] = []
    if width <= 0 or height <= 0:
        return BoxValidation((), ("Image dimensions are unavailable.",))
    if minimum_size_px <= 0:
        return BoxValidation((), ("Minimum ROI size must be positive.",))
    effective_valid = _rect_coordinates(valid_rect) if valid_rect is not None else None
    if valid_rect is not None and effective_valid is None:
        return BoxValidation((), ("Analysis valid_rect is invalid.",))
    if effective_valid is None:
        effective_valid = (0, 0, width, height)
    invalid_regions: list[tuple[tuple[int, int, int, int], str]] = []
    for index, invalid in enumerate(invalid_rects or (), start=1):
        invalid_coordinates = _rect_coordinates(invalid)
        if invalid_coordinates is None:
            errors.append(f"Invalid analysis region {index} has malformed coordinates.")
            continue
        invalid_regions.append((invalid_coordinates, str(invalid.get("reason") or "excluded")))
    if len(rows) > max_boxes:
        errors.append(f"At most {max_boxes} ROI rows are allowed.")

    for index, row in enumerate(rows[:max_boxes], start=1):
        coordinate_values = [row.get(name) for name in ("x1", "y1", "x2", "y2")]
        label = str(row.get("label") or "").strip()
        if all(value in {None, ""} for value in coordinate_values) and not label:
            continue
        parsed: list[int] = []
        for name, value in zip(("x1", "y1", "x2", "y2"), coordinate_values, strict=True):
            number = _coerce_int(value)
            if number is None:
                errors.append(f"ROI {index}: {name} must be an integer.")
                break
            parsed.append(number)
        if len(parsed) != 4:
            continue
        x1, y1, x2, y2 = parsed
        if x1 < 0 or y1 < 0 or x2 <= x1 or y2 <= y1:
            errors.append(f"ROI {index}: require 0 ≤ x1 < x2 and 0 ≤ y1 < y2.")
            continue
        if x2 > width or y2 > height:
            errors.append(f"ROI {index}: [{x1}:{x2}, {y1}:{y2}] exceeds {width}×{height}.")
            continue
        if x2 - x1 < minimum_size_px or y2 - y1 < minimum_size_px:
            errors.append(
                f"ROI {index}: width and height must each be at least {minimum_size_px} pixels."
            )
            continue
        valid_x1, valid_y1, valid_x2, valid_y2 = effective_valid
        if x1 < valid_x1 or y1 < valid_y1 or x2 > valid_x2 or y2 > valid_y2:
            errors.append(
                f"ROI {index}: [{x1}:{x2}, {y1}:{y2}] falls outside analysis "
                f"valid_rect [{valid_x1}:{valid_x2}, {valid_y1}:{valid_y2}]."
            )
            continue
        intersection = next(
            (
                (coordinates, reason)
                for coordinates, reason in invalid_regions
                if _rectangles_intersect((x1, y1, x2, y2), coordinates)
            ),
            None,
        )
        if intersection is not None:
            (invalid_x1, invalid_y1, invalid_x2, invalid_y2), reason = intersection
            errors.append(
                f"ROI {index}: intersects invalid region '{reason}' "
                f"[{invalid_x1}:{invalid_x2}, {invalid_y1}:{invalid_y2}]."
            )
            continue
        box: dict[str, Any] = {
            "label": label,
            "x1": x1,
            "y1": y1,
            "x2": x2,
            "y2": y2,
            "active": bool(row.get("active", True)),
        }
        box_id = _optional_text(row.get("box_id"))
        if box_id:
            box["box_id"] = box_id
        boxes.append(box)
    return BoxValidation(tuple(boxes), tuple(errors))


def build_run_payload(
    *,
    image_ids: Sequence[str],
    model_ids: Sequence[str],
    roi_mode: str,
    box_sets: Mapping[str, Mapping[str, Any]],
    threshold: float | None,
    min_area_px: int,
    watershed_enabled: bool,
    exclude_border: bool,
    device: str,
) -> dict[str, Any]:
    unique_images = list(dict.fromkeys(image_ids))
    unique_models = list(dict.fromkeys(model_ids))
    if not 1 <= len(unique_images) <= 20:
        raise ValueError("Select at least one and at most 20 images")
    if not 1 <= len(unique_models) <= 3:
        raise ValueError("Select between one and three models")
    if roi_mode not in {"full_image", "boxes"}:
        raise ValueError("ROI mode must be full_image or boxes")
    if threshold is not None and not 0 <= threshold <= 1:
        raise ValueError("Threshold must be between 0 and 1")
    if min_area_px < 0:
        raise ValueError("Minimum area cannot be negative")

    revisions: dict[str, int] = {}
    if roi_mode == "boxes":
        for image_id in unique_images:
            box_set = box_sets.get(image_id)
            if not box_set:
                raise ValueError(f"Save ROI boxes for {image_id} before submitting")
            boxes = box_set.get("boxes")
            if not isinstance(boxes, list) or not any(
                isinstance(box, Mapping) and box.get("active", True) for box in boxes
            ):
                raise ValueError(f"{image_id} has no active saved ROI boxes")
            revision = box_set.get("revision")
            if isinstance(revision, bool) or not isinstance(revision, int) or revision < 0:
                raise ValueError(f"{image_id} has no valid saved box revision")
            revisions[image_id] = revision

    return {
        "image_ids": unique_images,
        "model_ids": unique_models,
        "roi_mode": roi_mode,
        "box_revisions": revisions,
        "inference": {
            "threshold": threshold,
            "min_area_px": min_area_px,
            "watershed_enabled": watershed_enabled,
            "exclude_border": exclude_border,
            "device": device,
            "seed": 42,
        },
    }


def append_history(
    history: Sequence[Mapping[str, Any]],
    entry: Mapping[str, Any],
    *,
    limit: int = 20,
) -> list[dict[str, Any]]:
    if limit < 1:
        raise ValueError("history limit must be positive")
    values = [dict(item) for item in history]
    values.append(dict(entry))
    return values[-limit:]


def artifact_filename(download_url: str | None, fallback: str) -> str:
    if not download_url:
        return fallback
    name = Path(urlparse(download_url).path).name
    return name or fallback


def _optional_text(value: object) -> str | None:
    if value is None:
        return None
    text_value = str(value).strip()
    return text_value or None


def _coerce_int(value: object) -> int | None:
    if isinstance(value, bool) or value is None or value == "":
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, str):
        try:
            number = float(value)
        except ValueError:
            return None
        if number.is_integer():
            return int(number)
    return None


def _rect_coordinates(value: Mapping[str, Any] | None) -> tuple[int, int, int, int] | None:
    if value is None:
        return None
    coordinates = tuple(_coerce_int(value.get(name)) for name in ("x1", "y1", "x2", "y2"))
    if any(coordinate is None for coordinate in coordinates):
        return None
    x1, y1, x2, y2 = coordinates
    if x1 is None or y1 is None or x2 is None or y2 is None:
        return None
    if x1 < 0 or y1 < 0 or x1 >= x2 or y1 >= y2:
        return None
    return x1, y1, x2, y2


def _rectangles_intersect(
    first: tuple[int, int, int, int],
    second: tuple[int, int, int, int],
) -> bool:
    first_x1, first_y1, first_x2, first_y2 = first
    second_x1, second_y1, second_x2, second_y2 = second
    return (
        first_x1 < second_x2
        and first_x2 > second_x1
        and first_y1 < second_y2
        and first_y2 > second_y1
    )


__all__ = [
    "EXPORTABLE_RUN_STATUSES",
    "RUNNING_RUN_STATUSES",
    "TERMINAL_RUN_STATUSES",
    "BoxValidation",
    "HealthRollup",
    "PreviewArtifact",
    "absolute_api_url",
    "append_history",
    "artifact_filename",
    "as_dict",
    "build_analysis_metadata",
    "build_run_payload",
    "comparable_runs_by_image",
    "default_state",
    "ensure_session_state",
    "exportable_run_ids",
    "health_rollup",
    "is_terminal_run",
    "normalize_base_url",
    "parse_aliases",
    "parse_json_object",
    "pollable_run_ids",
    "preferred_preview_artifact",
    "rows_from_editor",
    "select_comparison_runs",
    "status_tone",
    "to_plain",
    "validate_box_rows",
]
