"""Result layer scientific display and Streamlit UX regression tests."""

from __future__ import annotations

from io import BytesIO
from pathlib import Path

import numpy as np
import pytest
from PIL import Image
from streamlit.testing.v1 import AppTest

from frontend.result_layers import prepare_result_layer_display, result_layer_sources

FIXTURE_APP = Path(__file__).parent / "fixtures" / "result_workbench_app.py"


def _probability_bytes(values: np.ndarray) -> bytes:
    output = BytesIO()
    np.save(output, values, allow_pickle=False)
    return output.getvalue()


def _radio(app: AppTest, label: str):  # type: ignore[no-untyped-def]
    return next(radio for radio in app.radio if radio.label == label)


def _markdown_values(app: AppTest) -> list[str]:
    return [str(item.value) for item in app.markdown]


def test_result_layers_include_original_and_all_visual_artifacts_in_stable_order() -> None:
    sources = result_layer_sources(
        {
            "artifacts": {
                "probability_url": "/api/v1/files/probability",
                "labeled_particles_url": "/api/v1/files/labeled",
                "overlay_url": "/api/v1/files/overlay",
                "mask_url": "/api/v1/files/mask",
                "instances_url": "/api/v1/files/instances",
            }
        },
        {"original_download_url": "/api/v1/files/original"},
    )

    assert [(source.key, source.download_url) for source in sources] == [
        ("original_image", "/api/v1/files/original"),
        ("mask_url", "/api/v1/files/mask"),
        ("overlay_url", "/api/v1/files/overlay"),
        ("labeled_particles_url", "/api/v1/files/labeled"),
        ("probability_url", "/api/v1/files/probability"),
    ]


def test_probability_npy_preview_uses_fixed_zero_to_one_grayscale() -> None:
    display = prepare_result_layer_display(
        layer_key="probability_url",
        content=_probability_bytes(np.array([[0.0, 0.5, 1.0]], dtype=np.float32)),
        content_type="application/octet-stream",
        filename="probability.npy",
    )

    with Image.open(BytesIO(display.content)) as preview:
        assert preview.size == (3, 1)
        assert np.asarray(preview).tolist() == [[0, 128, 255]]
    assert display.content_type == "image/png"
    assert display.note is not None and "固定 0–1" in display.note

    with pytest.raises(ValueError, match="0–1"):
        prepare_result_layer_display(
            layer_key="probability_url",
            content=_probability_bytes(np.array([[1.1]], dtype=np.float32)),
            content_type="application/octet-stream",
            filename="probability.npy",
        )


def test_single_run_app_exposes_all_layers_and_quality_before_numbers() -> None:
    app = AppTest.from_file(str(FIXTURE_APP), default_timeout=10).run()

    assert not app.exception
    layer = _radio(app, "预览图层")
    assert layer.options == ["原始图像", "分割掩膜", "叠加预览", "颗粒标注图", "概率图"]
    assert layer.value == "overlay_url"
    assert app.session_state["fixture_download_urls"] == ["/api/v1/files/run_1/overlay"]
    captions = [str(item.value) for item in app.caption]
    assert any("叠加预览 · 32 × 16 px · 所有图层共用同一固定高度视口" in item for item in captions)

    markdown = _markdown_values(app)
    assert markdown.index("#### 质量判断") < markdown.index("#### 数值汇总")
    assert any("检查 ROI 边界" in item for item in markdown)
    assert [item.value for item in app.warning] == ["边界接触比例偏高"]

    layer.set_value("probability_url")
    app.run()
    assert not app.exception
    assert app.session_state["fixture_download_urls"][-1] == "/api/v1/files/probability"
    assert any("固定 0–1 灰度" in str(item.value) for item in app.info)
    assert any("概率图 · 32 × 16 px" in str(item.value) for item in app.caption)


def test_comparison_app_places_quality_and_recommendations_before_each_summary() -> None:
    app = AppTest.from_file(str(FIXTURE_APP), default_timeout=10).run()
    _radio(app, "结果视图").set_value("comparison")
    app.run()

    assert not app.exception
    markdown = _markdown_values(app)
    quality_positions = [index for index, value in enumerate(markdown) if value == "#### 质量判断"]
    summary_positions = [index for index, value in enumerate(markdown) if value == "#### 核心统计"]
    assert len(quality_positions) == len(summary_positions) == 2
    assert all(
        quality < summary
        for quality, summary in zip(quality_positions, summary_positions, strict=True)
    )
    assert any("检查 ROI 边界" in item for item in markdown)
    assert any("复核阈值" in item for item in markdown)
