from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from scripts.models.calibrate_unet_small_min_area import (
    MinAreaCalibrationPlan,
    ThresholdEvidence,
    evaluate_min_area_calibration,
    load_min_area_plan,
    load_threshold_evidence,
    write_min_area_outputs,
)
from scripts.models.calibrate_unet_small_threshold import SelectionRule
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
EVIDENCE_SHA = "e" * 64
MANIFEST_SHA = "a" * 64
PLAN_SHA = "b" * 64


def _row(sample_id: str, split: str, digit: str) -> dict[str, str]:
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


def _manifest(tmp_path: Path):
    path = tmp_path / "split-manifest.csv"
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
    return load_split_manifest(path)


def test_min_area_calibration_manifest_still_requires_calibration(
    tmp_path: Path,
) -> None:
    path = tmp_path / "split-manifest.csv"
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows(
            [
                _row("train-1", "train", "1"),
                _row("test-3", "independent_test", "3"),
            ]
        )

    with pytest.raises(SplitManifestError, match="SELECTED requires calibration"):
        load_split_manifest(path)


def _evidence_payload(*, selected_threshold: float | None = 0.5) -> dict[str, object]:
    return {
        "selection_status": "SELECTED" if selected_threshold is not None else "NOT_EVALUATED",
        "selected_threshold": selected_threshold,
        "comparison_rule": "probability > threshold",
        "prediction_min_area_px": 0,
        "ground_truth_min_area_px": 0,
        "bottom_crop_px": 130,
        "independent_test_accessed": False,
        "split_manifest_sha256": MANIFEST_SHA,
        "calibration_sample_ids": ["calibration-2"],
    }


def _write_evidence(
    path: Path,
    *,
    selected_threshold: float | None = 0.5,
) -> tuple[Path, str]:
    path.write_text(
        json.dumps(_evidence_payload(selected_threshold=selected_threshold), sort_keys=True),
        encoding="utf-8",
    )
    return path, hashlib.sha256(path.read_bytes()).hexdigest()


def _plan(
    *,
    candidates: list[object] | None = None,
    evidence_sha: str = EVIDENCE_SHA,
    tie_break: list[dict[str, object]] | None = None,
) -> dict[str, object]:
    return {
        "candidate_min_area_px": candidates if candidates is not None else [0, 2],
        "selection_metric": {"metric": "instance_f1", "direction": "maximize"},
        "ordered_tie_break": tie_break
        if tie_break is not None
        else [{"metric": "min_area_px", "direction": "minimize"}],
        "minimum_gt_retention": 0.5,
        "threshold_evidence_sha256": evidence_sha,
    }


def _write_plan(path: Path, payload: dict[str, object]) -> Path:
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _loaded_plan(tmp_path: Path, **overrides) -> MinAreaCalibrationPlan:
    return load_min_area_plan(_write_plan(tmp_path / "plan.json", _plan(**overrides)))


def _threshold_evidence() -> ThresholdEvidence:
    return ThresholdEvidence(
        payload=_evidence_payload(),
        sha256=EVIDENCE_SHA,
        selected_threshold=0.5,
    )


def _arrays() -> tuple[np.ndarray, np.ndarray]:
    probability = np.zeros((140, 10), dtype=np.float32)
    truth = np.zeros((140, 10), dtype=bool)
    truth[2, 2] = True
    truth[4:6, 4:6] = True
    probability[truth] = 0.9
    return probability, truth


def test_threshold_evidence_missing_or_sha_mismatch_fails(tmp_path: Path) -> None:
    with pytest.raises((FileNotFoundError, ValueError)):
        load_threshold_evidence(
            tmp_path / "missing.json",
            expected_sha256="a" * 64,
        )

    evidence_path, _actual_sha = _write_evidence(tmp_path / "threshold.json")
    with pytest.raises(ValueError, match="SHA mismatch"):
        load_threshold_evidence(evidence_path, expected_sha256="a" * 64)


def test_threshold_must_be_selected(tmp_path: Path) -> None:
    evidence_path, evidence_sha = _write_evidence(
        tmp_path / "threshold.json",
        selected_threshold=None,
    )

    with pytest.raises(ValueError, match="no frozen selected threshold"):
        load_threshold_evidence(evidence_path, expected_sha256=evidence_sha)


@pytest.mark.parametrize(
    "candidates",
    [
        [-1, 2],
        [0, 2, 2],
        [2, 0],
        [0, 1.5],
        [False, 2],
    ],
)
def test_rejects_invalid_min_area_candidates(
    tmp_path: Path,
    candidates: list[object],
) -> None:
    with pytest.raises(ValueError, match="candidate_min_area_px"):
        _loaded_plan(tmp_path, candidates=candidates)


def test_reuses_probability_cache_once_and_only_for_calibration(tmp_path: Path) -> None:
    calls: list[tuple[str, str]] = []
    probability, truth = _arrays()

    def cache_provider(record):
        calls.append((record.sample_id, record.split.value))
        return probability

    result = evaluate_min_area_calibration(
        _manifest(tmp_path),
        _loaded_plan(tmp_path),
        _threshold_evidence(),
        probability_cache_provider=cache_provider,
        truth_provider=lambda _record: truth,
        manifest_sha256=MANIFEST_SHA,
        plan_sha256=PLAN_SHA,
    )

    assert calls == [("calibration-2", "calibration")]
    assert len(result["candidate_results"]) == 2
    assert result["probability_source"] == "threshold-stage cache; no model inference"
    assert result["independent_test_accessed"] is False


def test_candidate_area_applies_to_prediction_but_gt_remains_zero(
    tmp_path: Path,
) -> None:
    probability, truth = _arrays()

    result = evaluate_min_area_calibration(
        _manifest(tmp_path),
        _loaded_plan(tmp_path),
        _threshold_evidence(),
        probability_cache_provider=lambda _record: probability,
        truth_provider=lambda _record: truth,
        manifest_sha256=MANIFEST_SHA,
        plan_sha256=PLAN_SHA,
    )

    candidate = next(item for item in result["candidate_results"] if item["min_area_px"] == 2)
    per_image = candidate["per_image"][0]
    assert per_image["prediction_count"] == 1
    assert per_image["ground_truth_count"] == 2
    assert result["ground_truth_min_area_px"] == 0


def test_gt_retention_is_reported_from_canonical_instances(tmp_path: Path) -> None:
    probability, truth = _arrays()

    result = evaluate_min_area_calibration(
        _manifest(tmp_path),
        _loaded_plan(tmp_path),
        _threshold_evidence(),
        probability_cache_provider=lambda _record: probability,
        truth_provider=lambda _record: truth,
        manifest_sha256=MANIFEST_SHA,
        plan_sha256=PLAN_SHA,
    )

    candidate = next(item for item in result["candidate_results"] if item["min_area_px"] == 2)
    assert candidate["gt_baseline_count"] == 2
    assert candidate["gt_retained_count"] == 1
    assert candidate["aggregate"]["gt_retention"] == pytest.approx(0.5)
    assert candidate["per_image"][0]["gt_retention"] == pytest.approx(0.5)


def test_unresolved_tie_fails(tmp_path: Path) -> None:
    probability, truth = _arrays()
    plan = MinAreaCalibrationPlan(
        candidate_min_area_px=(0, 1),
        selection_metric=SelectionRule("instance_f1", "maximize"),
        ordered_tie_break=(),
        minimum_gt_retention=0.0,
        threshold_evidence_sha256=EVIDENCE_SHA,
    )

    with pytest.raises(ValueError, match="did not resolve the tie"):
        evaluate_min_area_calibration(
            _manifest(tmp_path),
            plan,
            _threshold_evidence(),
            probability_cache_provider=lambda _record: probability,
            truth_provider=lambda _record: truth,
            manifest_sha256=MANIFEST_SHA,
            plan_sha256=PLAN_SHA,
        )


def test_writes_json_csv_and_upstream_evidence_sha(tmp_path: Path) -> None:
    probability, truth = _arrays()
    payload = evaluate_min_area_calibration(
        _manifest(tmp_path),
        _loaded_plan(tmp_path),
        _threshold_evidence(),
        probability_cache_provider=lambda _record: probability,
        truth_provider=lambda _record: truth,
        manifest_sha256=MANIFEST_SHA,
        plan_sha256=PLAN_SHA,
    )
    output_root = tmp_path / "output"

    write_min_area_outputs(output_root, payload)

    assert {path.name for path in output_root.iterdir()} == {
        "min-area-calibration.json",
        "min-area-calibration.csv",
    }
    written = json.loads(
        (output_root / "min-area-calibration.json").read_text(encoding="utf-8")
    )
    assert written["threshold_evidence_sha256"] == EVIDENCE_SHA
    assert written["split_manifest_sha256"] == MANIFEST_SHA


def test_module_does_not_write_model_contract_files() -> None:
    source = Path(__import__(
        "scripts.models.calibrate_unet_small_min_area",
        fromlist=["__file__"],
    ).__file__).read_text(encoding="utf-8")

    assert "model_artifacts/configs" not in source
    assert "model_artifacts/registry.yaml" not in source
    assert "model_artifacts/model_cards" not in source
