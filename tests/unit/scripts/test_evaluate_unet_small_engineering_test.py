from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest

from scripts.models.evaluate_unet_small_engineering_test import (
    ENGINEERING_MIN_AREA_PX,
    ENGINEERING_THRESHOLD,
    evaluate_engineering_test,
    load_engineering_contract,
    write_engineering_outputs,
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
    "annotator_review_status",
    "calibration_status",
)


def _row(
    source_image_id: str,
    digit: str,
    *,
    included: bool = True,
    split: str = "independent_test",
) -> dict[str, str]:
    return {
        "sample_id": f"small-test-{source_image_id}",
        "source_sample_id": "SrNi" if split == "independent_test" else f"train-{digit}",
        "source_image_id": source_image_id,
        "field_of_view_id": f"field-{digit}",
        "split": split,
        "image_path": f"test_images/{source_image_id}.tif",
        "mask_path": (
            f"test_mask_human/test_mask_human/{source_image_id}_mask.tif"
            if split == "independent_test"
            else ""
        ),
        "image_sha256": digit * 64,
        "mask_sha256": ("f" * 63 + digit) if split == "independent_test" else "",
        "included": str(included).lower(),
        "exclusion_reason": "" if included else "not selected",
        "annotator_review_status": "approved" if split == "independent_test" else "",
        "calibration_status": "BLOCKED_NO_CALIBRATION_DATA",
    }


def _write_manifest(path: Path, *, add_excluded: bool = False) -> Path:
    rows = [
        _row("train-1", "4", split="train"),
        _row("SrNi-1", "1"),
        _row("SrNi-2", "2"),
        _row("SrNi-3", "3"),
    ]
    if add_excluded:
        rows.append(_row("other-test", "5", included=False))
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    return path


def _write_json(path: Path, payload: dict[str, object]) -> Path:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return path


def _contract_files(
    tmp_path: Path,
    *,
    threshold: float = ENGINEERING_THRESHOLD,
    min_area: int = ENGINEERING_MIN_AREA_PX,
    threshold_status: str = "ENGINEERING_PRESET",
    min_area_status: str = "ENGINEERING_PRESET",
    add_excluded: bool = False,
) -> tuple[Path, Path, Path]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    manifest_path = _write_manifest(
        tmp_path / "split-manifest.csv",
        add_excluded=add_excluded,
    )
    manifest_sha = sha256_file(manifest_path)
    common = {
        "schema_version": "1.0",
        "model_id": "unet-small-balanced-v1",
        "evidence_type": "ENGINEERING_ONLY",
        "calibration_performed": False,
        "calibration_status": "BLOCKED_NO_CALIBRATION_DATA",
        "split_manifest_sha256": manifest_sha,
        "scientific_verdict": "NOT_EVALUATED",
        "independent_test_may_adjust_parameter": False,
    }
    threshold_path = _write_json(
        tmp_path / "threshold-engineering-preset-evidence.json",
        {
            **common,
            "parameter": "threshold",
            "engineering_value": threshold,
            "selection_status": threshold_status,
        },
    )
    min_area_path = _write_json(
        tmp_path / "min-area-engineering-preset-evidence.json",
        {
            **common,
            "parameter": "min_area_px",
            "engineering_value": min_area,
            "selection_status": min_area_status,
        },
    )
    return manifest_path, threshold_path, min_area_path


def _load_contract(tmp_path: Path, **kwargs):
    manifest, threshold, min_area = _contract_files(tmp_path, **kwargs)
    return load_engineering_contract(
        split_manifest_path=manifest,
        threshold_evidence_path=threshold,
        min_area_evidence_path=min_area,
    )


def _arrays(*, matching: bool = True) -> tuple[np.ndarray, np.ndarray]:
    prediction = np.zeros((142, 12), dtype=bool)
    truth = np.zeros((142, 12), dtype=bool)
    truth[2:10, 2:10] = True
    if matching:
        prediction[2:10, 2:10] = True
    return prediction, truth


def test_accepts_engineering_preset_and_only_reads_three_included_rows(
    tmp_path: Path,
) -> None:
    contract = _load_contract(tmp_path, add_excluded=True)
    requested: list[str] = []
    prediction, truth = _arrays()

    def prediction_provider(record):
        requested.append(record.source_image_id)
        return prediction

    result = evaluate_engineering_test(
        contract,
        prediction_provider=prediction_provider,
        truth_provider=lambda _record: truth,
    )

    assert requested == ["SrNi-1", "SrNi-2", "SrNi-3"]
    assert result["evaluation_scope"] == "ENGINEERING_ONLY"
    assert result["scientific_verdict"] == "NOT_EVALUATED"
    assert result["summary"]["total_images"] == 3
    assert result["summary"]["source_sample_count"] == 1
    assert result["summary"]["independent_region_count"] == 3


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"threshold": 0.31}, "exactly 0.30"),
        ({"min_area": 63}, "exactly 64"),
        ({"threshold_status": "SELECTED"}, "ENGINEERING_PRESET"),
        ({"min_area_status": "SELECTED"}, "ENGINEERING_PRESET"),
    ],
)
def test_rejects_other_parameters_or_scientific_selection(
    tmp_path: Path,
    kwargs: dict[str, object],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        _load_contract(tmp_path, **kwargs)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("scientific_verdict", "PASS", "NOT_EVALUATED"),
        ("calibration_performed", True, "must be false"),
    ],
)
def test_rejects_scientific_claims(
    tmp_path: Path,
    field: str,
    value: object,
    message: str,
) -> None:
    manifest, threshold, min_area = _contract_files(tmp_path)
    payload = json.loads(threshold.read_text(encoding="utf-8"))
    payload[field] = value
    _write_json(threshold, payload)

    with pytest.raises(ValueError, match=message):
        load_engineering_contract(
            split_manifest_path=manifest,
            threshold_evidence_path=threshold,
            min_area_evidence_path=min_area,
        )


def test_scientific_verdict_is_not_evaluated_even_with_failure_cases(
    tmp_path: Path,
) -> None:
    contract = _load_contract(tmp_path)
    prediction, truth = _arrays(matching=False)

    result = evaluate_engineering_test(
        contract,
        prediction_provider=lambda _record: prediction,
        truth_provider=lambda _record: truth,
    )

    assert result["scientific_verdict"] == "NOT_EVALUATED"
    assert result["failure_cases"]
    assert {
        failure["reason_code"] for failure in result["failure_cases"]
    } == {"METRIC_NOT_EVALUABLE"}
    assert all(
        row["scientific_verdict"] == "NOT_EVALUATED"
        for row in result["per_image"]
    )
    assert all(row["evaluation_status"] != "PASS" for row in result["per_image"])


def test_outputs_contain_per_image_summary_and_failure_cases_without_writeback(
    tmp_path: Path,
) -> None:
    contract = _load_contract(tmp_path / "inputs")
    prediction, truth = _arrays(matching=False)
    result = evaluate_engineering_test(
        contract,
        prediction_provider=lambda _record: prediction,
        truth_provider=lambda _record: truth,
    )
    parameter_sentinel = tmp_path / "parameters.json"
    policy_sentinel = tmp_path / "policy.json"
    parameter_sentinel.write_text("unchanged\n", encoding="utf-8")
    policy_sentinel.write_text("unchanged\n", encoding="utf-8")

    output_root = tmp_path / "outputs"
    write_engineering_outputs(output_root, contract, result)

    payload = json.loads(
        (output_root / "engineering-test-metrics.json").read_text(encoding="utf-8")
    )
    with (output_root / "engineering-test-per-image.csv").open(
        encoding="utf-8",
        newline="",
    ) as stream:
        per_image = list(csv.DictReader(stream))
    with (output_root / "failure-cases.csv").open(
        encoding="utf-8",
        newline="",
    ) as stream:
        failures = list(csv.DictReader(stream))

    assert payload["evaluation_scope"] == "ENGINEERING_ONLY"
    assert payload["scientific_verdict"] == "NOT_EVALUATED"
    assert payload["summary"]["total_images"] == 3
    assert len(per_image) == 3
    assert failures
    assert parameter_sentinel.read_text(encoding="utf-8") == "unchanged\n"
    assert policy_sentinel.read_text(encoding="utf-8") == "unchanged\n"
    assert not list(output_root.glob("*parameter*"))
    assert not list(output_root.glob("*policy*"))


def test_does_not_call_formal_scientific_verdict_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import scripts.models.evaluate_unet_small_independent_test as formal

    monkeypatch.setattr(
        formal,
        "evaluate_independent_test",
        lambda *args, **kwargs: pytest.fail("formal verdict path was called"),
    )
    contract = _load_contract(tmp_path)
    prediction, truth = _arrays()

    result = evaluate_engineering_test(
        contract,
        prediction_provider=lambda _record: prediction,
        truth_provider=lambda _record: truth,
    )

    assert result["formal_scientific_verdict_path_called"] is False
    assert result["scientific_verdict"] == "NOT_EVALUATED"
