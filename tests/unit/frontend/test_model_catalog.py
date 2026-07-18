"""Pure model-catalog normalization and fail-closed availability tests."""

from __future__ import annotations

from frontend.model_catalog import (
    model_availability,
    model_filter_query,
    model_is_runnable,
)


def test_model_filter_query_preserves_machine_values_and_omits_blank_material() -> None:
    assert model_filter_query(
        family="unet",
        variant="small_particle",
        quality_tier="balanced",
        status="unavailable",
        material="  TiO2  ",
    ) == {
        "status": "unavailable",
        "family": "unet",
        "variant": "small_particle",
        "quality_tier": "balanced",
        "material": "TiO2",
    }
    assert model_filter_query(
        family=None,
        variant=None,
        quality_tier=None,
        status=None,
        material="   ",
    ) == {
        "status": None,
        "family": None,
        "variant": None,
        "quality_tier": None,
        "material": None,
    }


def test_unavailable_model_never_becomes_runnable_when_reason_is_missing() -> None:
    model = {"status": "unavailable", "health_error": None}

    availability = model_availability(model)

    assert not availability.runnable
    assert availability.severity == "error"
    assert "后端未提供具体健康原因" in availability.message
    assert not model_is_runnable(model)


def test_inconsistent_ready_health_record_fails_closed_and_exposes_reason() -> None:
    model = {"status": "ready", "health_error": "checkpoint checksum mismatch"}

    availability = model_availability(model)

    assert not availability.runnable
    assert availability.severity == "warning"
    assert "checkpoint checksum mismatch" in availability.message
    assert not model_is_runnable(model)


def test_clean_ready_model_is_the_only_runnable_state() -> None:
    availability = model_availability({"status": "READY", "health_error": "  "})

    assert availability.runnable
    assert availability.severity == "success"
