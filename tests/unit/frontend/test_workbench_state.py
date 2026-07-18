"""Pure frontend state and validation tests; no Streamlit runtime required."""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from frontend.components import display_enum, status_badge
from frontend.state import (
    absolute_api_url,
    append_history,
    as_dict,
    build_analysis_metadata,
    build_run_payload,
    comparable_runs_by_image,
    default_state,
    ensure_session_state,
    exportable_run_ids,
    health_rollup,
    knowledge_document_toggle,
    normalize_base_url,
    parse_aliases,
    parse_json_object,
    pollable_run_ids,
    preferred_preview_artifact,
    select_comparison_runs,
    validate_box_rows,
)


def test_session_defaults_are_fresh_and_preserve_existing_values() -> None:
    first = default_state()
    second = default_state()
    first["runs"]["run_1"] = {"status": "QUEUED"}
    first["roi_drafts"]["img_1"] = {"rows": []}

    state = ensure_session_state({"api_base_url": "https://lab.example"})

    assert second["runs"] == {}
    assert second["roi_drafts"] == {}
    assert second["comparison_previews"] == {}
    assert second["result_layer_previews"] == {}
    assert state["api_base_url"] == "https://lab.example"
    assert state["knowledge_documents"] == []
    assert state["knowledge_status_notice"] is None


def test_knowledge_document_toggle_only_allows_ready_disabled_transitions() -> None:
    disable = knowledge_document_toggle("READY")
    enable = knowledge_document_toggle("disabled")

    assert disable is not None
    assert (disable.enabled, disable.button_label, disable.completed_label) == (
        False,
        "禁用文档",
        "已禁用",
    )
    assert enable is not None
    assert (enable.enabled, enable.button_label, enable.completed_label) == (
        True,
        "启用文档",
        "已启用",
    )
    assert knowledge_document_toggle("indexing") is None
    assert knowledge_document_toggle(None) is None


def test_completed_runs_are_grouped_by_image_for_comparison() -> None:
    runs = [
        {
            "run_id": "run_a",
            "image_id": "img_1",
            "model_id": "model_a",
            "status": "COMPLETED",
        },
        {
            "run_id": "run_b",
            "image_id": "img_1",
            "model_id": "model_b",
            "status": "COMPLETED_WITH_WARNINGS",
        },
        {
            "run_id": "run_failed",
            "image_id": "img_1",
            "model_id": "model_c",
            "status": "FAILED",
        },
        {
            "run_id": "run_other",
            "image_id": "img_2",
            "model_id": "model_a",
            "status": "COMPLETED",
        },
        {
            "run_id": "run_a",
            "image_id": "img_1",
            "model_id": "duplicate",
            "status": "COMPLETED",
        },
    ]

    groups = comparable_runs_by_image(runs)

    assert list(groups) == ["img_1"]
    assert [run["run_id"] for run in groups["img_1"]] == ["run_a", "run_b"]
    assert list(comparable_runs_by_image(runs, minimum_runs=1)) == ["img_1", "img_2"]
    with pytest.raises(ValueError, match="positive"):
        comparable_runs_by_image(runs, minimum_runs=0)


def test_comparison_selection_is_same_image_terminal_and_ordered() -> None:
    runs = [
        {"run_id": "run_a", "image_id": "img_1", "status": "COMPLETED"},
        {
            "run_id": "run_b",
            "image_id": "img_1",
            "status": "COMPLETED_WITH_WARNINGS",
        },
        {"run_id": "run_pending", "image_id": "img_1", "status": "SEGMENTING"},
        {"run_id": "run_other", "image_id": "img_2", "status": "COMPLETED"},
    ]

    selected = select_comparison_runs(
        runs,
        image_id="img_1",
        run_ids=["run_b", "run_a"],
    )

    assert [run["run_id"] for run in selected] == ["run_b", "run_a"]
    with pytest.raises(ValueError, match="two or three"):
        select_comparison_runs(runs, image_id="img_1", run_ids=["run_a"])
    with pytest.raises(ValueError, match="unique"):
        select_comparison_runs(
            runs,
            image_id="img_1",
            run_ids=["run_a", "run_a"],
        )
    with pytest.raises(ValueError, match="selected image"):
        select_comparison_runs(
            runs,
            image_id="img_1",
            run_ids=["run_a", "run_other"],
        )
    with pytest.raises(ValueError, match="selected image"):
        select_comparison_runs(
            runs,
            image_id="img_1",
            run_ids=["run_a", "run_pending"],
        )


def test_comparison_preview_prefers_overlay_then_labeled_image_then_mask() -> None:
    overlay = preferred_preview_artifact(
        {
            "artifacts": {
                "mask_url": "/api/v1/files/mask",
                "labeled_particles_url": "/api/v1/files/labeled",
                "overlay_url": "/api/v1/files/overlay",
            }
        }
    )
    labeled = preferred_preview_artifact(
        {
            "artifacts": {
                "mask_url": "/api/v1/files/mask",
                "labeled_particles_url": "/api/v1/files/labeled",
            }
        }
    )

    assert overlay is not None
    assert (overlay.key, overlay.download_url) == (
        "overlay_url",
        "/api/v1/files/overlay",
    )
    assert labeled is not None
    assert labeled.key == "labeled_particles_url"
    assert preferred_preview_artifact({"artifacts": {"overlay_url": ""}}) is None


def test_api_origin_validation_and_relative_artifact_resolution() -> None:
    assert normalize_base_url(" https://lab.example/ ") == "https://lab.example"
    assert (
        absolute_api_url("https://lab.example", "/api/v1/files/token_1")
        == "https://lab.example/api/v1/files/token_1"
    )
    assert (
        absolute_api_url("https://lab.example/prefix", "api/v1/files/token_2")
        == "https://lab.example/prefix/api/v1/files/token_2"
    )
    assert (
        absolute_api_url("https://lab.example", "https://lab.example/api/v1/files/token_3")
        == "https://lab.example/api/v1/files/token_3"
    )
    with pytest.raises(ValueError, match="absolute"):
        normalize_base_url("localhost:8000")
    with pytest.raises(ValueError, match="query"):
        normalize_base_url("https://lab.example?token=secret")


def test_health_rollup_keeps_degraded_and_core_unavailable_explicit() -> None:
    healthy = {
        name: {"status": "healthy"}
        for name in ("service", "database", "model_registry", "rag_index")
    }
    assert health_rollup(healthy).status == "healthy"

    degraded = {**healthy, "rag_index": {"status": "degraded", "detail": "keyword only"}}
    report = health_rollup(degraded)
    assert report.status == "degraded"
    assert report.unhealthy_components == ("rag_index",)
    assert "keyword only" in report.detail

    unavailable = {**healthy, "database": {"status": "unavailable", "detail": "offline"}}
    assert health_rollup(unavailable).status == "unavailable"
    assert health_rollup(None).status == "unknown"


def test_run_status_helpers_only_export_successful_terminal_runs() -> None:
    runs = {
        "queued": {"run_id": "queued", "status": "QUEUED"},
        "done": {"run_id": "done", "status": "COMPLETED"},
        "warn": {"run_id": "warn", "status": "COMPLETED_WITH_WARNINGS"},
        "failed": {"run_id": "failed", "status": "FAILED"},
    }

    assert pollable_run_ids(runs) == ["queued"]
    assert exportable_run_ids(runs.values()) == ["done", "warn"]


def test_alias_and_json_parsers_are_deterministic() -> None:
    assert parse_aliases("SrNiO3-x, Sr-Ni；SrNiO3-x\nperovskite") == [
        "SrNiO3-x",
        "Sr-Ni",
        "perovskite",
    ]
    assert parse_json_object('{"temperature_c": 700}', field_name="conditions") == {
        "temperature_c": 700
    }
    with pytest.raises(ValueError, match="JSON object"):
        parse_json_object("[]", field_name="conditions")
    with pytest.raises(ValueError, match="line 1"):
        parse_json_object("{broken", field_name="conditions")


def test_project_metadata_requires_explicit_sample_and_physical_scale() -> None:
    metadata = build_analysis_metadata(
        "Catalyst batch",
        ["sample.tif", "control.png"],
        {
            "sample.tif": {
                "sample_id": "S-01",
                "material_formula": "SrNiO3-x",
                "scale_mode": "nm_per_pixel",
                "scale_value": 0.55,
                "experiment_conditions": {"treatment": "CO2"},
            },
            "control.png": {
                "sample_id": "C-01",
                "scale_mode": "pixel_only",
                "scale_value": 99.0,
                "experiment_conditions": {},
            },
        },
    )

    assert metadata["images"][0]["scale"] == {
        "mode": "nm_per_pixel",
        "value": 0.55,
    }
    assert metadata["images"][1]["scale"] == {"mode": "pixel_only", "value": None}
    with pytest.raises(ValueError, match="sample ID"):
        build_analysis_metadata(
            "Project",
            ["missing.tif"],
            {"missing.tif": {"scale_mode": "pixel_only"}},
        )
    with pytest.raises(ValueError, match="scale is required"):
        build_analysis_metadata(
            "Project",
            ["missing.tif"],
            {
                "missing.tif": {
                    "sample_id": "S-1",
                    "scale_mode": "nm_per_pixel",
                    "scale_value": None,
                }
            },
        )


def test_numeric_roi_validation_uses_half_open_image_bounds() -> None:
    result = validate_box_rows(
        [
            {"label": "particle field", "x1": 0.0, "y1": "2", "x2": 80, "y2": 90},
            {"label": "", "x1": None, "y1": None, "x2": None, "y2": None},
        ],
        width=100,
        height=100,
    )
    assert not result.errors
    assert result.boxes == (
        {
            "label": "particle field",
            "x1": 0,
            "y1": 2,
            "x2": 80,
            "y2": 90,
            "active": True,
        },
    )

    invalid = validate_box_rows([{"x1": 90, "y1": 0, "x2": 101, "y2": 50}], width=100, height=100)
    assert invalid.boxes == ()
    assert "exceeds 100×100" in invalid.errors[0]


def test_numeric_roi_validation_respects_valid_and_invalid_analysis_regions() -> None:
    valid_rect = {"x1": 10, "y1": 10, "x2": 90, "y2": 90}
    invalid_rects = [{"x1": 10, "y1": 80, "x2": 90, "y2": 90, "reason": "instrument_bar"}]

    accepted = validate_box_rows(
        [{"x1": 10, "y1": 10, "x2": 50, "y2": 80}],
        width=100,
        height=100,
        valid_rect=valid_rect,
        invalid_rects=invalid_rects,
        minimum_size_px=1,
    )
    assert not accepted.errors
    assert len(accepted.boxes) == 1

    outside = validate_box_rows(
        [{"x1": 0, "y1": 10, "x2": 30, "y2": 40}],
        width=100,
        height=100,
        valid_rect=valid_rect,
        invalid_rects=invalid_rects,
        minimum_size_px=1,
    )
    assert "outside analysis valid_rect" in outside.errors[0]

    intersects = validate_box_rows(
        [{"x1": 20, "y1": 79, "x2": 40, "y2": 85}],
        width=100,
        height=100,
        valid_rect=valid_rect,
        invalid_rects=invalid_rects,
        minimum_size_px=1,
    )
    assert "instrument_bar" in intersects.errors[0]

    touches_without_overlap = validate_box_rows(
        [{"x1": 20, "y1": 70, "x2": 40, "y2": 80}],
        width=100,
        height=100,
        valid_rect=valid_rect,
        invalid_rects=invalid_rects,
        minimum_size_px=1,
    )
    assert not touches_without_overlap.errors


def test_numeric_roi_validation_enforces_backend_minimum_side_length() -> None:
    too_small = validate_box_rows(
        [{"x1": 0, "y1": 0, "x2": 31, "y2": 64}],
        width=100,
        height=100,
    )
    exact = validate_box_rows(
        [{"x1": 0, "y1": 0, "x2": 32, "y2": 32}],
        width=100,
        height=100,
    )

    assert too_small.boxes == ()
    assert "at least 32 pixels" in too_small.errors[0]
    assert not exact.errors


def test_run_payload_requires_saved_active_revisions_for_box_mode() -> None:
    payload = build_run_payload(
        image_ids=["img_1", "img_2"],
        model_ids=["model_a", "model_b"],
        roi_mode="boxes",
        box_sets={
            "img_1": {"revision": 2, "boxes": [{"active": True}]},
            "img_2": {"revision": 4, "boxes": [{"active": True}]},
        },
        threshold=None,
        min_area_px=8,
        watershed_enabled=False,
        exclude_border=True,
        device="auto",
    )
    assert payload["box_revisions"] == {"img_1": 2, "img_2": 4}
    assert payload["inference"]["threshold"] is None

    with pytest.raises(ValueError, match="img_2"):
        build_run_payload(
            image_ids=["img_1", "img_2"],
            model_ids=["model_a"],
            roi_mode="boxes",
            box_sets={"img_1": {"revision": 1, "boxes": [{"active": True}]}},
            threshold=0.5,
            min_area_px=8,
            watershed_enabled=False,
            exclude_border=True,
            device="auto",
        )


@dataclass(frozen=True)
class _ApiResult:
    data: dict[str, object]
    request_id: str = "req_1"


def test_api_result_unwrap_history_limit_and_safe_status_html() -> None:
    assert as_dict(_ApiResult({"job_id": "job_1"})) == {"job_id": "job_1"}
    history = append_history([{"question": "q1"}], {"question": "q2"}, limit=1)
    assert history == [{"question": "q2"}]
    badge = status_badge("ready", label='<script>alert("x")</script>')
    assert "<script>" not in badge
    assert "&lt;script&gt;" in badge


def test_machine_enums_receive_chinese_display_labels_only_at_render_time() -> None:
    assert display_enum("COMPLETED_WITH_WARNINGS") == "已完成（有警告）"
    assert display_enum("small_particle") == "小颗粒"
    assert display_enum("balanced") == "均衡"
    assert display_enum("material_knowledge") == "材料知识"
    assert display_enum("instrument_bar_detected") == "检测到仪器信息栏"
    assert "就绪" in status_badge("ready")
