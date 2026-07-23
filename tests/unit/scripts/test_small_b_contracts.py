from __future__ import annotations

import csv
from pathlib import Path

import pytest

from scripts.models.small_b_contracts import (
    ManifestSplit,
    SplitManifestError,
    load_split_manifest,
)

FIELDS = (
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


def _row(
    sample_id: str,
    split: str,
    *,
    source_sample_id: str | None = None,
    source_image_id: str | None = None,
    field_of_view_id: str | None = None,
    image_sha256: str | None = None,
    included: str = "true",
    exclusion_reason: str = "",
) -> dict[str, str]:
    index = sample_id.rsplit("-", 1)[-1]
    requires_mask = split in {"calibration", "independent_test"}
    return {
        "sample_id": sample_id,
        "source_sample_id": source_sample_id or f"source-sample-{index}",
        "source_image_id": source_image_id or f"source-image-{index}",
        "field_of_view_id": field_of_view_id or f"field-{index}",
        "split": split,
        "image_path": f"images/{sample_id}.tif",
        "mask_path": f"masks/{sample_id}.png" if requires_mask else "",
        "image_sha256": image_sha256 or index.zfill(64),
        "mask_sha256": ("f" * 63 + index[-1]) if requires_mask else "",
        "included": included,
        "exclusion_reason": exclusion_reason,
    }


def _valid_rows() -> list[dict[str, str]]:
    return [
        _row("train-1", "train"),
        _row("calibration-2", "calibration"),
        _row("test-3", "independent_test"),
        _row(
            "excluded-4",
            "excluded",
            included="false",
            exclusion_reason="annotation rejected",
        ),
    ]


def _write_manifest(path: Path, rows: list[dict[str, str]]) -> Path:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    return path


def test_loads_valid_manifest_and_selects_only_included_formal_rows(tmp_path: Path) -> None:
    manifest = load_split_manifest(_write_manifest(tmp_path / "split.csv", _valid_rows()))

    assert len(manifest.records) == 4
    assert [item.sample_id for item in manifest.select(ManifestSplit.CALIBRATION)] == [
        "calibration-2"
    ]
    assert [item.sample_id for item in manifest.select("independent_test")] == ["test-3"]
    assert manifest.select(ManifestSplit.EXCLUDED) == ()


def test_rejects_duplicate_sample_id_with_line_and_field(tmp_path: Path) -> None:
    rows = _valid_rows()
    rows[2]["sample_id"] = rows[1]["sample_id"]

    with pytest.raises(SplitManifestError, match=r"line 4, field 'sample_id'.*line 3"):
        load_split_manifest(_write_manifest(tmp_path / "split.csv", rows))


def test_rejects_same_image_sha_across_splits(tmp_path: Path) -> None:
    rows = _valid_rows()
    rows[2]["image_sha256"] = rows[0]["image_sha256"]

    with pytest.raises(SplitManifestError, match=r"line 4, field 'image_sha256'.*train"):
        load_split_manifest(_write_manifest(tmp_path / "split.csv", rows))


@pytest.mark.parametrize(
    "field",
    ["source_sample_id", "source_image_id", "field_of_view_id"],
)
def test_rejects_calibration_test_source_overlap(tmp_path: Path, field: str) -> None:
    rows = _valid_rows()
    rows[2][field] = rows[1][field]

    with pytest.raises(SplitManifestError, match=rf"line 4, field '{field}'.*calibration"):
        load_split_manifest(_write_manifest(tmp_path / "split.csv", rows))


@pytest.mark.parametrize("missing_split", ["calibration", "independent_test"])
def test_requires_included_calibration_and_independent_test(
    tmp_path: Path,
    missing_split: str,
) -> None:
    rows = _valid_rows()
    for row in rows:
        if row["split"] == missing_split:
            row["included"] = "false"
            row["exclusion_reason"] = "not approved"

    with pytest.raises(
        SplitManifestError,
        match=(
            r"line 1, field 'split'.*SELECTED requires calibration"
            if missing_split == "calibration"
            else r"line 1, field 'split'.*independent_test is required"
        ),
    ):
        load_split_manifest(_write_manifest(tmp_path / "split.csv", rows))


@pytest.mark.parametrize("field", ["mask_path", "mask_sha256"])
def test_requires_mask_identity_for_calibration_and_test(
    tmp_path: Path,
    field: str,
) -> None:
    rows = _valid_rows()
    rows[1][field] = ""

    with pytest.raises(SplitManifestError, match=rf"line 3, field '{field}'"):
        load_split_manifest(_write_manifest(tmp_path / "split.csv", rows))


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("image_sha256", "A" * 64),
        ("image_sha256", "a" * 63),
        ("mask_sha256", "g" * 64),
    ],
)
def test_rejects_invalid_sha(
    tmp_path: Path,
    field: str,
    value: str,
) -> None:
    rows = _valid_rows()
    rows[1][field] = value

    with pytest.raises(SplitManifestError, match=rf"line 3, field '{field}'.*lowercase"):
        load_split_manifest(_write_manifest(tmp_path / "split.csv", rows))


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("image_path", "/private/sample.tif"),
        ("image_path", "../sample.tif"),
        ("image_path", "https://example.invalid/sample.tif"),
        ("mask_path", r"C:\private\mask.png"),
    ],
)
def test_rejects_unsafe_paths(
    tmp_path: Path,
    field: str,
    value: str,
) -> None:
    rows = _valid_rows()
    rows[1][field] = value

    with pytest.raises(SplitManifestError, match=rf"line 3, field '{field}'"):
        load_split_manifest(_write_manifest(tmp_path / "split.csv", rows))


def test_excluded_rows_never_enter_formal_selection(tmp_path: Path) -> None:
    rows = _valid_rows()
    rows[3]["source_sample_id"] = rows[1]["source_sample_id"]
    rows[3]["source_image_id"] = rows[1]["source_image_id"]
    rows[3]["field_of_view_id"] = rows[1]["field_of_view_id"]
    rows[3]["image_sha256"] = rows[1]["image_sha256"]

    manifest = load_split_manifest(_write_manifest(tmp_path / "split.csv", rows))

    selected_ids = {
        item.sample_id
        for split in (
            ManifestSplit.TRAIN,
            ManifestSplit.CALIBRATION,
            ManifestSplit.INDEPENDENT_TEST,
        )
        for item in manifest.select(split)
    }
    assert "excluded-4" not in selected_ids


def test_requires_exclusion_reason_for_not_included_row(tmp_path: Path) -> None:
    rows = _valid_rows()
    rows[3]["exclusion_reason"] = ""

    with pytest.raises(SplitManifestError, match=r"line 5, field 'exclusion_reason'"):
        load_split_manifest(_write_manifest(tmp_path / "split.csv", rows))


def test_default_loader_still_rejects_empty_calibration(tmp_path: Path) -> None:
    rows = [row for row in _valid_rows() if row["split"] != "calibration"]

    with pytest.raises(
        SplitManifestError,
        match="SELECTED requires calibration",
    ):
        load_split_manifest(_write_manifest(tmp_path / "split.csv", rows))


def test_frozen_mode_accepts_empty_calibration_but_requires_independent_test(
    tmp_path: Path,
) -> None:
    rows = [row for row in _valid_rows() if row["split"] != "calibration"]
    manifest = load_split_manifest(
        _write_manifest(tmp_path / "split.csv", rows),
        require_calibration=False,
    )

    assert manifest.select(ManifestSplit.CALIBRATION) == ()
    assert [record.sample_id for record in manifest.select("independent_test")] == [
        "test-3"
    ]

    rows = [row for row in rows if row["split"] != "independent_test"]
    with pytest.raises(
        SplitManifestError,
        match="independent_test is required",
    ):
        load_split_manifest(
            _write_manifest(tmp_path / "no-test.csv", rows),
            require_calibration=False,
        )


def test_frozen_mode_still_validates_existing_calibration_rows(tmp_path: Path) -> None:
    rows = _valid_rows()
    rows[1]["mask_path"] = ""

    with pytest.raises(SplitManifestError, match=r"field 'mask_path'"):
        load_split_manifest(
            _write_manifest(tmp_path / "split.csv", rows),
            require_calibration=False,
        )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("image_sha256", "A" * 64, "image_sha256"),
        ("image_path", "../private.tif", "image_path"),
        ("mask_path", "", "mask_path"),
    ],
)
def test_empty_calibration_does_not_weaken_test_row_validation(
    tmp_path: Path,
    field: str,
    value: str,
    message: str,
) -> None:
    rows = [row for row in _valid_rows() if row["split"] != "calibration"]
    test_row = next(row for row in rows if row["split"] == "independent_test")
    test_row[field] = value

    with pytest.raises(SplitManifestError, match=message):
        load_split_manifest(
            _write_manifest(tmp_path / "split.csv", rows),
            require_calibration=False,
        )


def test_empty_calibration_does_not_weaken_cross_split_sha_check(
    tmp_path: Path,
) -> None:
    rows = [row for row in _valid_rows() if row["split"] != "calibration"]
    train_row = next(row for row in rows if row["split"] == "train")
    test_row = next(row for row in rows if row["split"] == "independent_test")
    test_row["image_sha256"] = train_row["image_sha256"]

    with pytest.raises(SplitManifestError, match="crosses split 'train'"):
        load_split_manifest(
            _write_manifest(tmp_path / "split.csv", rows),
            require_calibration=False,
        )


def test_require_calibration_must_be_boolean(tmp_path: Path) -> None:
    with pytest.raises(TypeError, match="must be a bool"):
        load_split_manifest(  # type: ignore[arg-type]
            _write_manifest(tmp_path / "split.csv", _valid_rows()),
            require_calibration=0,
        )
