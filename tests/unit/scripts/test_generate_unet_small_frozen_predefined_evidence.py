from __future__ import annotations

import csv
import json
import os
from pathlib import Path

import pytest

import scripts.models.generate_unet_small_frozen_predefined_evidence as generator
from scripts.models.evaluate_unet_small_independent_test import (
    load_evaluation_contract,
)
from scripts.models.generate_unet_small_frozen_predefined_evidence import (
    COMPARISON_RULE,
    build_parser,
    generate_evidence,
)
from scripts.models.small_b_tolerance_policy import sha256_file

MANIFEST_FIELDS = (
    "sample_id",
    "source_sample_id",
    "source_image_id",
    "field_of_view_id",
    "split",
    "image_path",
    "mask_path",
    "image_sha256",
    "mask_sha256",
    "included",
    "exclusion_reason",
)
THRESHOLD_SOURCE = (
    "Small model predefined operating threshold from the established training/design "
    "workflow, frozen before Independent Test unblinding."
)
MIN_AREA_SOURCE = (
    "Post-processing minimum-area parameter predefined and frozen before Independent "
    "Test unblinding."
)
FROZEN_AT = "2026-07-24T08:00:00+08:00"


def _row(sample_id: str, split: str, digit: str) -> dict[str, str]:
    evaluated = split in {"calibration", "independent_test"}
    return {
        "sample_id": sample_id,
        "source_sample_id": f"source-sample-{digit}",
        "source_image_id": f"source-image-{digit}",
        "field_of_view_id": f"field-{digit}",
        "split": split,
        "image_path": f"images/{sample_id}.tif",
        "mask_path": f"masks/{sample_id}.tif" if evaluated else "",
        "image_sha256": digit * 64,
        "mask_sha256": ("f" * 63 + digit) if evaluated else "",
        "included": "true",
        "exclusion_reason": "",
    }


def _write_manifest(path: Path) -> Path:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows(
            [
                _row("train-1", "train", "1"),
                _row("calibration-2", "calibration", "2"),
                _row("test-3", "independent_test", "3"),
            ]
        )
    return path


def _write_json(path: Path, payload: dict[str, object]) -> Path:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return path


def _inputs(tmp_path: Path) -> dict[str, Path]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    manifest = _write_manifest(tmp_path / "split-manifest.csv")
    model = tmp_path / "best_unet_small.pth"
    config = tmp_path / "unet-small-balanced-v1.yaml"
    weight = tmp_path / "unet-small-balanced-v1.pt"
    model.write_bytes(b"source-checkpoint")
    config.write_bytes(b"config")
    weight.write_bytes(b"torchscript")
    return {
        "manifest": manifest,
        "model": model,
        "config": config,
        "weight": weight,
    }


def _generate(tmp_path: Path, **overrides):
    inputs = _inputs(tmp_path / "inputs")
    parameters = {
        **inputs,
        "threshold_output": tmp_path / "output" / "threshold.json",
        "min_area_output": tmp_path / "output" / "min-area.json",
        "threshold_parameter_source": THRESHOLD_SOURCE,
        "min_area_parameter_source": MIN_AREA_SOURCE,
        "frozen_at": FROZEN_AT,
    }
    parameters.update(overrides)
    result = generate_evidence(**parameters)
    return inputs, parameters, result


def _policy_payload(*, threshold_sha: str, min_area_sha: str) -> dict[str, object]:
    return {
        "schema_version": "1",
        "policy_id": "small-b-policy",
        "policy_version": "1",
        "model_id": "unet-small-balanced-v1",
        "threshold_evidence_sha256": threshold_sha,
        "min_area_evidence_sha256": min_area_sha,
        "frozen_scientific_parameters": {
            "threshold": 0.30,
            "threshold_comparison": "gt",
            "min_area_px": 64,
            "bottom_crop_px": 130,
        },
        "instance_matching": {
            "metric": "mask_iou",
            "mask_iou_threshold": 0.7,
        },
        "per_image_tolerances": {
            "minimum_instance_precision": 0.0,
            "minimum_instance_recall": 0.0,
            "minimum_instance_f1": 0.0,
            "maximum_count_absolute_error": 100,
            "maximum_count_relative_error": 1.0,
            "maximum_mean_area_relative_error": 1.0,
            "maximum_mean_equivalent_diameter_relative_error": 1.0,
            "maximum_number_density_relative_error": 1.0,
            "maximum_perimeter_density_relative_error": 1.0,
        },
        "not_evaluable_rule": "fail",
        "approval": {
            "frozen_before_independent_test": True,
            "approved_by": "Guo Jinghao",
            "approved_at": "2026-07-24T08:00:00+08:00",
            "rationale": "Frozen before Independent Test unblinding.",
        },
    }


def test_generates_stable_schema_and_threshold_cross_reference(
    tmp_path: Path,
) -> None:
    inputs, parameters, result = _generate(tmp_path)
    threshold_path = parameters["threshold_output"]
    min_area_path = parameters["min_area_output"]
    threshold = json.loads(threshold_path.read_text(encoding="utf-8"))
    min_area = json.loads(min_area_path.read_text(encoding="utf-8"))

    assert result["formal_contract_validation"] == "PASSED"
    assert threshold == {
        "bottom_crop_px": 130,
        "calibration_performed": False,
        "comparison_rule": COMPARISON_RULE,
        "config_sha256": sha256_file(inputs["config"]),
        "frozen_at": FROZEN_AT,
        "frozen_before_test": True,
        "independent_test_used_for_tuning": False,
        "manifest_sha256": sha256_file(inputs["manifest"]),
        "model_sha256": sha256_file(inputs["model"]),
        "parameter_source": THRESHOLD_SOURCE,
        "parameters_changed_after_test": False,
        "selection_status": "FROZEN_PREDEFINED",
        "threshold": 0.30,
        "weight_sha256": sha256_file(inputs["weight"]),
    }
    assert min_area["parameter_source"] == MIN_AREA_SOURCE
    assert min_area["threshold"] == 0.30
    assert min_area["min_area_px"] == 64
    assert min_area["threshold_evidence_sha256"] == sha256_file(threshold_path)
    assert result["threshold_evidence_sha256"] == sha256_file(threshold_path)
    assert result["min_area_evidence_sha256"] == sha256_file(min_area_path)
    assert threshold_path.read_bytes().endswith(b"\n")
    assert min_area_path.read_bytes().endswith(b"\n")


def test_generated_result_loads_through_formal_independent_test_contract(
    tmp_path: Path,
) -> None:
    inputs, parameters, _result = _generate(tmp_path)
    threshold_path = parameters["threshold_output"]
    min_area_path = parameters["min_area_output"]
    policy_path = _write_json(
        tmp_path / "tolerance-policy.json",
        _policy_payload(
            threshold_sha=sha256_file(threshold_path),
            min_area_sha=sha256_file(min_area_path),
        ),
    )

    contract = load_evaluation_contract(
        split_manifest_path=inputs["manifest"],
        threshold_evidence_path=threshold_path,
        min_area_evidence_path=min_area_path,
        tolerance_policy_path=policy_path,
        expected_policy_sha256=sha256_file(policy_path),
    )

    assert contract.threshold_evidence["selection_status"] == "FROZEN_PREDEFINED"
    assert contract.min_area_evidence["threshold_evidence_sha256"] == sha256_file(
        threshold_path
    )


def test_missing_input_fails_without_outputs(tmp_path: Path) -> None:
    inputs = _inputs(tmp_path / "inputs")
    inputs["weight"] = tmp_path / "missing.pt"
    threshold_output = tmp_path / "threshold.json"
    min_area_output = tmp_path / "min-area.json"

    with pytest.raises(FileNotFoundError):
        generate_evidence(
            **inputs,
            threshold_output=threshold_output,
            min_area_output=min_area_output,
            threshold_parameter_source=THRESHOLD_SOURCE,
            min_area_parameter_source=MIN_AREA_SOURCE,
            frozen_at=FROZEN_AT,
        )

    assert not threshold_output.exists()
    assert not min_area_output.exists()


def test_existing_output_requires_explicit_overwrite(tmp_path: Path) -> None:
    inputs = _inputs(tmp_path / "inputs")
    threshold_output = tmp_path / "threshold.json"
    min_area_output = tmp_path / "min-area.json"
    threshold_output.write_text("existing", encoding="utf-8")

    with pytest.raises(FileExistsError, match="--overwrite"):
        generate_evidence(
            **inputs,
            threshold_output=threshold_output,
            min_area_output=min_area_output,
            threshold_parameter_source=THRESHOLD_SOURCE,
            min_area_parameter_source=MIN_AREA_SOURCE,
            frozen_at=FROZEN_AT,
        )

    assert threshold_output.read_text(encoding="utf-8") == "existing"
    assert not min_area_output.exists()


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ({"frozen_at": "2026-07-24T08:00:00"}, "timezone"),
        ({"threshold": 0.31}, "exactly 0.30"),
        ({"min_area_px": 63}, "exactly 64"),
        ({"bottom_crop_px": 129}, "exactly 130"),
        ({"comparison_rule": "probability >= threshold"}, "comparison_rule"),
    ],
)
def test_rejects_invalid_contract_values(
    tmp_path: Path,
    override: dict[str, object],
    message: str,
) -> None:
    inputs = _inputs(tmp_path / "inputs")
    threshold_output = tmp_path / "threshold.json"
    min_area_output = tmp_path / "min-area.json"
    parameters = {
        **inputs,
        "threshold_output": threshold_output,
        "min_area_output": min_area_output,
        "threshold_parameter_source": THRESHOLD_SOURCE,
        "min_area_parameter_source": MIN_AREA_SOURCE,
        "frozen_at": FROZEN_AT,
        **override,
    }

    with pytest.raises(ValueError, match=message):
        generate_evidence(**parameters)

    assert not threshold_output.exists()
    assert not min_area_output.exists()


def test_input_change_fails_without_partial_outputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inputs = _inputs(tmp_path / "inputs")
    threshold_output = tmp_path / "threshold.json"
    min_area_output = tmp_path / "min-area.json"
    original = generator._assert_inputs_unchanged

    def mutate_then_validate(paths, expected):
        paths["model"].write_bytes(b"changed-after-hash")
        original(paths, expected)

    monkeypatch.setattr(generator, "_assert_inputs_unchanged", mutate_then_validate)
    with pytest.raises(ValueError, match="model changed"):
        generate_evidence(
            **inputs,
            threshold_output=threshold_output,
            min_area_output=min_area_output,
            threshold_parameter_source=THRESHOLD_SOURCE,
            min_area_parameter_source=MIN_AREA_SOURCE,
            frozen_at=FROZEN_AT,
        )

    assert not threshold_output.exists()
    assert not min_area_output.exists()


def test_atomic_publish_failure_leaves_no_partial_files(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inputs = _inputs(tmp_path / "inputs")
    threshold_output = tmp_path / "threshold.json"
    min_area_output = tmp_path / "min-area.json"
    real_replace = os.replace
    publish_calls = 0

    def fail_second_publish(source, destination):
        nonlocal publish_calls
        if Path(destination) in {threshold_output, min_area_output}:
            publish_calls += 1
            if publish_calls == 2:
                raise OSError("simulated second publish failure")
        real_replace(source, destination)

    monkeypatch.setattr(generator.os, "replace", fail_second_publish)
    with pytest.raises(OSError, match="simulated"):
        generate_evidence(
            **inputs,
            threshold_output=threshold_output,
            min_area_output=min_area_output,
            threshold_parameter_source=THRESHOLD_SOURCE,
            min_area_parameter_source=MIN_AREA_SOURCE,
            frozen_at=FROZEN_AT,
        )

    assert not threshold_output.exists()
    assert not min_area_output.exists()


def test_generator_cli_cannot_produce_engineering_preset() -> None:
    option_strings = {
        option
        for action in build_parser()._actions
        for option in action.option_strings
    }

    assert "--selection-status" not in option_strings
    assert "--calibration-performed" not in option_strings
    assert "--frozen-before-test" not in option_strings
    assert "--parameters-changed-after-test" not in option_strings
    assert "--independent-test-used-for-tuning" not in option_strings
