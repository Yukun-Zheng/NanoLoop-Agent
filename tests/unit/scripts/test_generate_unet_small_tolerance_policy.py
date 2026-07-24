from __future__ import annotations

import json
from pathlib import Path

import pytest

import scripts.models.generate_unet_small_tolerance_policy as generator
from scripts.models.generate_unet_small_frozen_predefined_evidence import (
    generate_evidence,
)
from scripts.models.generate_unet_small_tolerance_policy import (
    build_parser,
    generate_tolerance_policy,
)
from scripts.models.small_b_tolerance_policy import (
    load_tolerance_policy,
    sha256_file,
)

THRESHOLD_SOURCE = (
    "Small model predefined operating threshold from the established training/design "
    "workflow, frozen before Independent Test unblinding."
)
MIN_AREA_SOURCE = (
    "Post-processing minimum-area parameter predefined and frozen before Independent "
    "Test unblinding."
)
APPROVED_AT = "2026-07-24T08:00:00+08:00"
RATIONALE = (
    "Prespecified Small-B scientific tolerances frozen before Independent Test "
    "evaluation."
)


def _write_json(path: Path, payload: dict[str, object]) -> Path:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return path


def _evidence_files(tmp_path: Path) -> tuple[Path, Path]:
    inputs = tmp_path / "inputs"
    inputs.mkdir(parents=True)
    manifest = inputs / "split-manifest.csv"
    model = inputs / "best_unet_small.pth"
    config = inputs / "unet-small-balanced-v1.yaml"
    weight = inputs / "unet-small-balanced-v1.pt"
    manifest.write_bytes(b"manifest")
    model.write_bytes(b"checkpoint")
    config.write_bytes(b"config")
    weight.write_bytes(b"torchscript")
    threshold = tmp_path / "threshold.json"
    min_area = tmp_path / "min-area.json"
    generate_evidence(
        manifest=manifest,
        model=model,
        config=config,
        weight=weight,
        threshold_output=threshold,
        min_area_output=min_area,
        threshold_parameter_source=THRESHOLD_SOURCE,
        min_area_parameter_source=MIN_AREA_SOURCE,
        frozen_at=APPROVED_AT,
    )
    return threshold, min_area


def _parameters(tmp_path: Path) -> dict[str, object]:
    threshold, min_area = _evidence_files(tmp_path)
    return {
        "threshold_evidence": threshold,
        "min_area_evidence": min_area,
        "output": tmp_path / "tolerance-policy.json",
        "policy_id": "small-b-policy",
        "policy_version": "1",
        "approved_by": "Guo Jinghao",
        "approved_at": APPROVED_AT,
        "rationale": RATIONALE,
        "mask_iou_threshold": 0.50,
        "minimum_instance_precision": 0.80,
        "minimum_instance_recall": 0.70,
        "minimum_instance_f1": 0.75,
        "maximum_count_absolute_error": 2,
        "maximum_count_relative_error": 0.20,
        "maximum_mean_area_relative_error": 0.25,
        "maximum_mean_equivalent_diameter_relative_error": 0.20,
        "maximum_number_density_relative_error": 0.25,
        "maximum_perimeter_density_relative_error": 0.25,
    }


def test_generates_policy_and_passes_formal_loader(tmp_path: Path) -> None:
    parameters = _parameters(tmp_path)

    result = generate_tolerance_policy(**parameters)
    policy = load_tolerance_policy(
        parameters["output"],
        threshold_evidence_path=parameters["threshold_evidence"],
        min_area_evidence_path=parameters["min_area_evidence"],
    )
    payload = json.loads(parameters["output"].read_text(encoding="utf-8"))

    assert result["formal_policy_validation"] == "PASSED"
    assert result["policy_sha256"] == sha256_file(parameters["output"])
    assert payload["threshold_evidence_sha256"] == sha256_file(
        parameters["threshold_evidence"]
    )
    assert payload["min_area_evidence_sha256"] == sha256_file(
        parameters["min_area_evidence"]
    )
    assert payload["frozen_scientific_parameters"] == {
        "threshold": 0.30,
        "threshold_comparison": "gt",
        "min_area_px": 64,
        "bottom_crop_px": 130,
    }
    assert payload["not_evaluable_rule"] == policy.not_evaluable_rule == "fail"
    assert parameters["output"].read_bytes().endswith(b"\n")


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("mask_iou_threshold", 0.0, "mask_iou_threshold"),
        ("minimum_instance_precision", -0.1, "minimum_instance_precision"),
        ("minimum_instance_recall", 1.1, "minimum_instance_recall"),
        ("minimum_instance_f1", float("inf"), "minimum_instance_f1"),
        ("maximum_count_absolute_error", -1, "maximum_count_absolute_error"),
        ("maximum_count_relative_error", -0.1, "maximum_count_relative_error"),
        ("maximum_mean_area_relative_error", float("nan"), "mean_area"),
    ],
)
def test_rejects_invalid_tolerance(
    tmp_path: Path,
    field: str,
    value: object,
    message: str,
) -> None:
    parameters = _parameters(tmp_path)
    parameters[field] = value

    with pytest.raises(ValueError, match=message):
        generate_tolerance_policy(**parameters)

    assert not parameters["output"].exists()


def test_rejects_approved_at_without_timezone(tmp_path: Path) -> None:
    parameters = _parameters(tmp_path)
    parameters["approved_at"] = "2026-07-24T08:00:00"

    with pytest.raises(ValueError, match="timezone"):
        generate_tolerance_policy(**parameters)

    assert not parameters["output"].exists()


def test_rejects_missing_evidence(tmp_path: Path) -> None:
    parameters = _parameters(tmp_path)
    parameters["threshold_evidence"] = tmp_path / "missing.json"

    with pytest.raises(FileNotFoundError):
        generate_tolerance_policy(**parameters)

    assert not parameters["output"].exists()


def test_existing_output_requires_overwrite(tmp_path: Path) -> None:
    parameters = _parameters(tmp_path)
    parameters["output"].write_text("existing", encoding="utf-8")

    with pytest.raises(FileExistsError, match="--overwrite"):
        generate_tolerance_policy(**parameters)

    assert parameters["output"].read_text(encoding="utf-8") == "existing"


def test_atomic_publish_failure_leaves_no_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parameters = _parameters(tmp_path)

    def fail_replace(_source, _destination):
        raise OSError("simulated atomic publish failure")

    monkeypatch.setattr(generator.os, "replace", fail_replace)
    with pytest.raises(OSError, match="simulated"):
        generate_tolerance_policy(**parameters)

    assert not parameters["output"].exists()


@pytest.mark.parametrize(
    ("target", "field", "value", "message"),
    [
        ("threshold_evidence", "threshold", 0.31, "threshold"),
        ("min_area_evidence", "min_area_px", 63, "min_area_px"),
        ("threshold_evidence", "bottom_crop_px", 129, "bottom_crop_px"),
        (
            "min_area_evidence",
            "comparison_rule",
            "probability >= threshold",
            "comparison_rule",
        ),
    ],
)
def test_rejects_evidence_parameter_drift(
    tmp_path: Path,
    target: str,
    field: str,
    value: object,
    message: str,
) -> None:
    parameters = _parameters(tmp_path)
    evidence_path = parameters[target]
    payload = json.loads(evidence_path.read_text(encoding="utf-8"))
    payload[field] = value
    _write_json(evidence_path, payload)

    with pytest.raises(ValueError, match=message):
        generate_tolerance_policy(**parameters)

    assert not parameters["output"].exists()


def test_cli_does_not_allow_not_evaluable_rule_override() -> None:
    option_strings = {
        option
        for action in build_parser()._actions
        for option in action.option_strings
    }

    assert "--not-evaluable-rule" not in option_strings
    assert "--independent-test-results" not in option_strings
