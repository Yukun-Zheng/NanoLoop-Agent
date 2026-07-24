from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from scripts.models import calibrate_unet_small_threshold as calibration_module
from scripts.models.calibrate_unet_small_threshold import (
    SelectionRule,
    ThresholdCalibrationPlan,
    evaluate_threshold_calibration,
    load_threshold_plan,
    not_evaluated_payload,
    write_calibration_outputs,
)
from scripts.models.small_b_contracts import SplitManifestError, load_split_manifest

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


def _manifest_row(sample_id: str, split: str, digit: str) -> dict[str, str]:
    has_mask = split in {"calibration", "independent_test"}
    return {
        "sample_id": sample_id,
        "source_sample_id": f"source-sample-{digit}",
        "source_image_id": f"source-image-{digit}",
        "field_of_view_id": f"field-{digit}",
        "split": split,
        "image_path": f"images/{sample_id}.tif",
        "mask_path": f"masks/{sample_id}.png" if has_mask else "",
        "image_sha256": digit * 64,
        "mask_sha256": ("f" * 63 + digit) if has_mask else "",
        "included": "true",
        "exclusion_reason": "",
    }


def _write_manifest(path: Path, *, two_calibration_rows: bool = False) -> Path:
    rows = [
        _manifest_row("train-1", "train", "1"),
        _manifest_row("calibration-2", "calibration", "2"),
        _manifest_row("test-3", "independent_test", "3"),
    ]
    if two_calibration_rows:
        rows.insert(2, _manifest_row("calibration-4", "calibration", "4"))
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    return path


def _write_manifest_without_calibration(path: Path) -> Path:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows(
            [
                _manifest_row("train-1", "train", "1"),
                _manifest_row("test-3", "independent_test", "3"),
            ]
        )
    return path


def _plan(
    *,
    candidates: list[object] | None = None,
    tie_break: list[dict[str, object]] | None = None,
) -> dict[str, object]:
    return {
        "candidate_thresholds": candidates or [0.3, 0.5],
        "selection_metric": {"metric": "instance_f1", "direction": "maximize"},
        "ordered_tie_break": tie_break
        if tie_break is not None
        else [{"metric": "threshold", "direction": "minimize"}],
        "bottom_crop_px": 130,
    }


def _write_plan(path: Path, payload: dict[str, object] | None = None) -> Path:
    path.write_text(json.dumps(payload or _plan()), encoding="utf-8")
    return path


def _loaded_manifest(tmp_path: Path, *, two_calibration_rows: bool = False):
    return load_split_manifest(
        _write_manifest(
            tmp_path / "split-manifest.csv",
            two_calibration_rows=two_calibration_rows,
        )
    )


def _loaded_plan(tmp_path: Path) -> ThresholdCalibrationPlan:
    return load_threshold_plan(_write_plan(tmp_path / "plan.json"))


def _arrays() -> tuple[np.ndarray, np.ndarray]:
    probability = np.zeros((132, 3), dtype=np.float32)
    truth = np.zeros((132, 3), dtype=bool)
    probability[0, 0] = 0.5
    probability[0, 1] = 0.75
    truth[0, 1] = True
    return probability, truth


def test_threshold_calibration_manifest_still_requires_calibration(
    tmp_path: Path,
) -> None:
    with pytest.raises(SplitManifestError, match="SELECTED requires calibration"):
        load_split_manifest(
            _write_manifest_without_calibration(tmp_path / "split-manifest.csv")
        )


@pytest.mark.parametrize(
    "candidates",
    [
        [0.0, 0.5],
        [0.3, 1.0],
        [0.5, 0.3],
        [0.3, 0.3],
        [0.3, "0.5"],
        [0.3, float("nan")],
    ],
)
def test_rejects_invalid_candidate_thresholds(
    tmp_path: Path,
    candidates: list[object],
) -> None:
    plan_path = _write_plan(tmp_path / "plan.json", _plan(candidates=candidates))

    with pytest.raises(ValueError, match="candidate_thresholds"):
        load_threshold_plan(plan_path)


def test_uses_strict_greater_than_and_min_area_zero(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: list[dict[str, Any]] = []

    def fake_metrics(
        prediction: np.ndarray,
        truth: np.ndarray,
        **kwargs: Any,
    ) -> dict[str, Any]:
        captured.append({"prediction": prediction.copy(), "truth": truth.copy(), **kwargs})
        return {
            "instance_matching": {"precision": 1.0, "recall": 1.0, "f1": 1.0},
            "count": {"absolute_error": 0, "relative_error": 0.0},
        }

    monkeypatch.setattr(calibration_module, "compute_scientific_metrics", fake_metrics)
    probability, truth = _arrays()

    evaluate_threshold_calibration(
        _loaded_manifest(tmp_path),
        ThresholdCalibrationPlan(
            candidate_thresholds=(0.5,),
            selection_metric=SelectionRule("instance_f1", "maximize"),
            ordered_tie_break=(),
        ),
        probability_provider=lambda _record: probability,
        truth_provider=lambda _record: truth,
        manifest_sha256="a" * 64,
        plan_sha256="b" * 64,
    )

    assert captured[0]["prediction"][0].tolist() == [False, True, False]
    assert captured[0]["profile"].min_area_px == 0
    assert captured[0]["truth"].shape == (2, 3)


def test_calls_probability_provider_once_per_calibration_image_and_reuses_cache(
    tmp_path: Path,
) -> None:
    manifest = _loaded_manifest(tmp_path, two_calibration_rows=True)
    plan = _loaded_plan(tmp_path)
    calls: list[str] = []
    probability, truth = _arrays()

    def probability_provider(record):
        calls.append(record.sample_id)
        return probability

    result = evaluate_threshold_calibration(
        manifest,
        plan,
        probability_provider=probability_provider,
        truth_provider=lambda _record: truth,
        manifest_sha256="a" * 64,
        plan_sha256="b" * 64,
    )

    assert calls == ["calibration-2", "calibration-4"]
    assert len(result["candidate_results"]) == 2
    assert result["split_manifest_sha256"] == "a" * 64
    assert result["plan_sha256"] == "b" * 64


def test_never_requests_train_or_independent_test_records(tmp_path: Path) -> None:
    requested: list[tuple[str, str]] = []
    probability, truth = _arrays()

    def probability_provider(record):
        requested.append((record.sample_id, record.split.value))
        return probability

    evaluate_threshold_calibration(
        _loaded_manifest(tmp_path),
        _loaded_plan(tmp_path),
        probability_provider=probability_provider,
        truth_provider=lambda _record: truth,
        manifest_sha256="a" * 64,
        plan_sha256="b" * 64,
    )

    assert requested == [("calibration-2", "calibration")]


def test_unresolved_tie_fails(tmp_path: Path) -> None:
    probability, truth = _arrays()
    plan = ThresholdCalibrationPlan(
        candidate_thresholds=(0.3, 0.5),
        selection_metric=SelectionRule("instance_f1", "maximize"),
        ordered_tie_break=(),
    )

    with pytest.raises(ValueError, match="did not resolve the tie"):
        evaluate_threshold_calibration(
            _loaded_manifest(tmp_path),
            plan,
            probability_provider=lambda _record: probability,
            truth_provider=lambda _record: truth,
            manifest_sha256="a" * 64,
            plan_sha256="b" * 64,
        )


def test_not_evaluated_output_records_hashes_without_model_or_data_access(
    tmp_path: Path,
) -> None:
    manifest = _loaded_manifest(tmp_path)
    plan = _loaded_plan(tmp_path)
    payload = not_evaluated_payload(
        manifest,
        plan,
        manifest_sha256="a" * 64,
        plan_sha256="b" * 64,
    )
    output_root = tmp_path / "output"

    write_calibration_outputs(output_root, payload)

    assert payload["selected_threshold"] is None
    assert payload["selection_status"] == "NOT_EVALUATED"
    assert payload["independent_test_accessed"] is False
    assert {path.name for path in output_root.iterdir()} == {
        "threshold-calibration.json",
        "threshold-calibration.csv",
    }
    written = json.loads(
        (output_root / "threshold-calibration.json").read_text(encoding="utf-8")
    )
    assert written["split_manifest_sha256"] == "a" * 64
    assert written["plan_sha256"] == "b" * 64


def test_module_does_not_write_model_contract_files(tmp_path: Path) -> None:
    source = Path(calibration_module.__file__).read_text(encoding="utf-8")

    assert "model_artifacts/configs" not in source
    assert "model_artifacts/registry.yaml" not in source
    assert "model_artifacts/model_cards" not in source
