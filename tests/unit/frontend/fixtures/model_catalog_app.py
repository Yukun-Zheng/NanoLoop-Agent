"""Deterministic Streamlit fixture for model-catalog interaction tests."""

from __future__ import annotations

from typing import Any

import streamlit as st

from frontend.app import _model_configuration

READY_MODEL: dict[str, Any] = {
    "model_id": "unet-general-balanced-v2",
    "family": "unet",
    "variant": "general",
    "quality_tier": "balanced",
    "version": "2.1.0",
    "status": "ready",
    "supports_box_prompt": False,
    "default_threshold": 0.48,
    "preprocess_profile": "sem-gray-normalize-v2",
    "postprocess_profile": "semantic-particles-v3",
    "applicable_materials": ["TiO2", "SiO2"],
    "metrics": {"dice": 0.93, "iou": 0.88},
    "metric_context": {"dataset": "SEM-val-2026", "images": 128},
    "notes": "Validated on catalyst microscopy images.",
    "health_error": None,
}

UNAVAILABLE_MODEL: dict[str, Any] = {
    "model_id": "unet-small-balanced-v1",
    "family": "unet",
    "variant": "small_particle",
    "quality_tier": "balanced",
    "version": "1.0.0",
    "status": "unavailable",
    "supports_box_prompt": False,
    "default_threshold": 0.5,
    "preprocess_profile": "sem-gray-normalize-v1",
    "postprocess_profile": "semantic-particles-v2",
    "applicable_materials": ["TiO2"],
    "metrics": {"dice": 0.89},
    "metric_context": {"dataset": "small-particle-holdout", "images": 42},
    "notes": "Checkpoint is supplied by the model owner.",
    "health_error": "Model checkpoint is not bundled.",
}


class _FixtureClient:
    def list_models(
        self,
        *,
        status: str | None = None,
        family: str | None = None,
        variant: str | None = None,
        quality_tier: str | None = None,
        material: str | None = None,
    ) -> dict[str, object]:
        filters = {
            "status": status,
            "family": family,
            "variant": variant,
            "quality_tier": quality_tier,
            "material": material,
        }
        st.session_state["fixture_model_filters"] = filters
        models = (
            [UNAVAILABLE_MODEL] if status == "unavailable" else [READY_MODEL, UNAVAILABLE_MODEL]
        )
        return {"models": models}

    def recommend_models(self, payload: dict[str, object]) -> dict[str, object]:
        del payload
        return {"candidates": []}

    def create_runs(self, job_id: str, payload: dict[str, object]) -> dict[str, object]:
        del job_id, payload
        return {"run_ids": []}


state: Any = st.session_state
state.setdefault("active_job_id", "job_1")
state.setdefault("box_sets", {})
state.setdefault("models", [READY_MODEL, UNAVAILABLE_MODEL])
state.setdefault("models_loaded", True)
state.setdefault("model_catalog_filters", {})
state.setdefault("model_catalog_selected_id", "")
state.setdefault("recommendations", [])
state.setdefault("selected_model_ids", [])
state.setdefault("run_ids", [])

_model_configuration(st, state, _FixtureClient(), "img_1", ["img_1"])
