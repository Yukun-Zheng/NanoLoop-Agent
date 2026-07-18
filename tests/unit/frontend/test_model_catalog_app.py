"""Streamlit AppTest coverage for model filters and honest model details."""

from __future__ import annotations

from pathlib import Path

from streamlit.testing.v1 import AppTest

FIXTURE_APP = Path(__file__).parent / "fixtures" / "model_catalog_app.py"


def _selectbox(app: AppTest, label: str):  # type: ignore[no-untyped-def]
    return next(widget for widget in app.selectbox if widget.label == label)


def _button(app: AppTest, label: str):  # type: ignore[no-untyped-def]
    return next(widget for widget in app.button if widget.label == label)


def test_filters_are_sent_to_list_models_as_backend_machine_values() -> None:
    app = AppTest.from_file(str(FIXTURE_APP), default_timeout=10).run()

    _selectbox(app, "模型族").set_value("unet")
    _selectbox(app, "变体").set_value("small_particle")
    _selectbox(app, "质量档位").set_value("balanced")
    _selectbox(app, "状态").set_value("unavailable")
    next(widget for widget in app.text_input if widget.label == "适用材料").set_value(" TiO2 ")
    _button(app, "应用筛选 / 刷新模型").click()
    app.run()

    assert not app.exception
    assert app.session_state["fixture_model_filters"] == {
        "status": "unavailable",
        "family": "unet",
        "variant": "small_particle",
        "quality_tier": "balanced",
        "material": "TiO2",
    }
    assert [model["model_id"] for model in app.session_state["models"]] == [
        "unet-small-balanced-v1"
    ]


def test_selected_unavailable_model_shows_full_details_and_is_not_runnable() -> None:
    app = AppTest.from_file(str(FIXTURE_APP), default_timeout=10).run()
    _selectbox(app, "查看模型详情").set_value("unet-small-balanced-v1")
    app.run()

    assert not app.exception
    assert any(item.value == "unet-small-balanced-v1" for item in app.subheader)
    assert any("版本 1.0.0" in str(item.value) for item in app.caption)
    assert any("TiO2" in str(item.value) for item in app.caption)
    assert [str(item.value) for item in app.code][-2:] == [
        "sem-gray-normalize-v1",
        "semantic-particles-v2",
    ]
    assert any("Model checkpoint is not bundled." in str(item.value) for item in app.error)
    assert any(
        "Checkpoint is supplied by the model owner." in str(item.value)
        for item in app.markdown
    )
    run_selector = next(
        widget for widget in app.multiselect if widget.label == "已确认的就绪模型（1–3 个）"
    )
    assert run_selector.options == ["unet-general-balanced-v2 · 均衡"]
