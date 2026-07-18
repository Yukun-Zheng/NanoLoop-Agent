"""Deterministic Streamlit fixture for result-layer and quality-order tests."""

from __future__ import annotations

from io import BytesIO

import numpy as np
import streamlit as st
from PIL import Image

from frontend.api_client import ArtifactDownload
from frontend.app import _runs_page


def _png(value: int) -> bytes:
    output = BytesIO()
    Image.new("L", (32, 16), value).save(output, format="PNG")
    return output.getvalue()


def _probability() -> bytes:
    output = BytesIO()
    values = np.linspace(0.0, 1.0, 512, dtype=np.float32).reshape(16, 32)
    np.save(output, values, allow_pickle=False)
    return output.getvalue()


class _FixtureClient:
    def download_artifact(self, url: str) -> ArtifactDownload:
        calls = st.session_state.setdefault("fixture_download_urls", [])
        calls.append(url)
        if url.endswith("/probability"):
            return ArtifactDownload(
                content=_probability(),
                filename="probability.npy",
                content_type="application/octet-stream",
                request_id="req_probability",
            )
        return ArtifactDownload(
            content=_png(96),
            filename=f"{url.rsplit('/', maxsplit=1)[-1]}.png",
            content_type="image/png",
            request_id="req_image",
        )


def _run(run_id: str, model_id: str, reason: str, recommendation: str) -> dict[str, object]:
    return {
        "run_id": run_id,
        "image_id": "img_1",
        "model_id": model_id,
        "status": "COMPLETED_WITH_WARNINGS",
        "roi_mode": "full_image",
        "runtime_ms": 12,
        "configuration": {"model_version": "1.0"},
        "artifacts": {
            "mask_url": f"/api/v1/files/{run_id}/mask",
            "overlay_url": f"/api/v1/files/{run_id}/overlay",
            "labeled_particles_url": f"/api/v1/files/{run_id}/labeled",
            "probability_url": "/api/v1/files/probability",
        },
        "quality": {
            "status": "WARN",
            "reasons": [reason],
            "recommendations": [recommendation],
            "metrics": {"border_touch_ratio": 0.25},
        },
        "summary": {
            "particle_count": 4,
            "mean_equivalent_diameter_px": 3.0,
            "number_density_px2": 0.1,
            "coverage_ratio": 0.2,
        },
    }


runs = [
    _run("run_1", "model_a", "边界接触比例偏高", "检查 ROI 边界"),
    _run("run_2", "model_b", "小颗粒召回不稳定", "复核阈值"),
]
state = {
    "active_job_id": "job_1",
    "job_detail": {
        "job": {"job_id": "job_1"},
        "images": [
            {
                "image_id": "img_1",
                "filename": "sem.png",
                "original_download_url": "/api/v1/files/original",
            }
        ],
        "runs": runs,
    },
    "run_ids": ["run_1", "run_2"],
    "runs": {str(run["run_id"]): run for run in runs},
    "result_layer_previews": {},
    "comparison_previews": {},
    "comparison_downloads": {},
}

_runs_page(st, state, _FixtureClient())
