"""Small-B private dataset split-manifest contract.

This module validates metadata only. It never opens model assets, images, or masks.
"""

from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path, PurePosixPath, PureWindowsPath

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_URI_SCHEME_RE = re.compile(r"^[A-Za-z][A-Za-z0-9+.-]*:")

REQUIRED_COLUMNS = (
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


class ManifestSplit(StrEnum):
    TRAIN = "train"
    CALIBRATION = "calibration"
    INDEPENDENT_TEST = "independent_test"
    EXCLUDED = "excluded"


class SplitManifestError(ValueError):
    """Raised when a split-manifest row violates the Small-B contract."""


@dataclass(frozen=True, slots=True)
class SplitManifestRecord:
    line_number: int
    sample_id: str
    source_sample_id: str
    source_image_id: str
    field_of_view_id: str
    split: ManifestSplit
    image_path: str
    mask_path: str | None
    image_sha256: str
    mask_sha256: str | None
    included: bool
    exclusion_reason: str | None


@dataclass(frozen=True, slots=True)
class SmallBSplitManifest:
    records: tuple[SplitManifestRecord, ...]

    def select(self, split: ManifestSplit | str) -> tuple[SplitManifestRecord, ...]:
        """Return included records for one formal split.

        Excluded records are never returned by this formal-selection interface.
        """

        resolved = _coerce_split(split, line_number=1)
        if resolved is ManifestSplit.EXCLUDED:
            return ()
        return tuple(
            record
            for record in self.records
            if record.included and record.split is resolved
        )


def _error(line_number: int, field: str, message: str) -> SplitManifestError:
    return SplitManifestError(f"line {line_number}, field '{field}': {message}")


def _coerce_split(value: ManifestSplit | str, *, line_number: int) -> ManifestSplit:
    if isinstance(value, ManifestSplit):
        return value
    try:
        return ManifestSplit(value)
    except ValueError as error:
        allowed = ", ".join(item.value for item in ManifestSplit)
        raise _error(line_number, "split", f"must be one of: {allowed}") from error


def _required_value(row: dict[str, str | None], field: str, *, line_number: int) -> str:
    value = (row.get(field) or "").strip()
    if not value:
        raise _error(line_number, field, "must not be empty")
    return value


def _optional_value(row: dict[str, str | None], field: str) -> str | None:
    value = (row.get(field) or "").strip()
    return value or None


def _parse_boolean(value: str, *, line_number: int) -> bool:
    if value == "true":
        return True
    if value == "false":
        return False
    raise _error(line_number, "included", "must be exactly 'true' or 'false'")


def _validate_sha256(value: str, *, line_number: int, field: str) -> str:
    if not _SHA256_RE.fullmatch(value):
        raise _error(line_number, field, "must be 64 lowercase hexadecimal characters")
    return value


def _validate_relative_path(value: str, *, line_number: int, field: str) -> str:
    if _URI_SCHEME_RE.match(value):
        raise _error(line_number, field, "must be a safe relative path, not a URL or drive path")

    posix = PurePosixPath(value)
    windows = PureWindowsPath(value)
    if posix.is_absolute() or windows.is_absolute() or windows.drive:
        raise _error(line_number, field, "must be a safe relative path, not an absolute path")
    if ".." in posix.parts or ".." in windows.parts:
        raise _error(line_number, field, "must not contain '..'")
    if value.startswith(("//", "\\\\")):
        raise _error(line_number, field, "must not be a network path")
    return value


def _parse_record(
    row: dict[str, str | None],
    *,
    line_number: int,
) -> SplitManifestRecord:
    sample_id = _required_value(row, "sample_id", line_number=line_number)
    source_sample_id = _required_value(row, "source_sample_id", line_number=line_number)
    source_image_id = _required_value(row, "source_image_id", line_number=line_number)
    field_of_view_id = _required_value(row, "field_of_view_id", line_number=line_number)
    split = _coerce_split(
        _required_value(row, "split", line_number=line_number),
        line_number=line_number,
    )
    image_path = _validate_relative_path(
        _required_value(row, "image_path", line_number=line_number),
        line_number=line_number,
        field="image_path",
    )
    image_sha256 = _validate_sha256(
        _required_value(row, "image_sha256", line_number=line_number),
        line_number=line_number,
        field="image_sha256",
    )
    included = _parse_boolean(
        _required_value(row, "included", line_number=line_number),
        line_number=line_number,
    )
    exclusion_reason = _optional_value(row, "exclusion_reason")
    if not included and exclusion_reason is None:
        raise _error(
            line_number,
            "exclusion_reason",
            "is required when included is false",
        )

    mask_path = _optional_value(row, "mask_path")
    mask_sha256 = _optional_value(row, "mask_sha256")
    if split in {ManifestSplit.CALIBRATION, ManifestSplit.INDEPENDENT_TEST}:
        if mask_path is None:
            raise _error(line_number, "mask_path", "is required for calibration and test rows")
        if mask_sha256 is None:
            raise _error(line_number, "mask_sha256", "is required for calibration and test rows")
    if mask_path is not None:
        mask_path = _validate_relative_path(
            mask_path,
            line_number=line_number,
            field="mask_path",
        )
    if mask_sha256 is not None:
        mask_sha256 = _validate_sha256(
            mask_sha256,
            line_number=line_number,
            field="mask_sha256",
        )
    if (mask_path is None) != (mask_sha256 is None):
        missing_field = "mask_path" if mask_path is None else "mask_sha256"
        raise _error(line_number, missing_field, "mask path and SHA must be provided together")

    return SplitManifestRecord(
        line_number=line_number,
        sample_id=sample_id,
        source_sample_id=source_sample_id,
        source_image_id=source_image_id,
        field_of_view_id=field_of_view_id,
        split=split,
        image_path=image_path,
        mask_path=mask_path,
        image_sha256=image_sha256,
        mask_sha256=mask_sha256,
        included=included,
        exclusion_reason=exclusion_reason,
    )


def _validate_unique_sample_ids(records: tuple[SplitManifestRecord, ...]) -> None:
    seen: dict[str, SplitManifestRecord] = {}
    for record in records:
        previous = seen.get(record.sample_id)
        if previous is not None:
            raise _error(
                record.line_number,
                "sample_id",
                f"duplicates line {previous.line_number}",
            )
        seen[record.sample_id] = record


def _validate_no_cross_split_source_or_sha(
    records: tuple[SplitManifestRecord, ...],
) -> None:
    formal = tuple(record for record in records if record.split is not ManifestSplit.EXCLUDED)
    for field in ("source_image_id", "image_sha256"):
        seen: dict[str, SplitManifestRecord] = {}
        for record in formal:
            value = getattr(record, field)
            previous = seen.get(value)
            if previous is not None and previous.split is not record.split:
                raise _error(
                    record.line_number,
                    field,
                    f"crosses split '{previous.split.value}' from line {previous.line_number}",
                )
            seen.setdefault(value, record)


def _validate_calibration_test_disjoint(
    records: tuple[SplitManifestRecord, ...],
) -> None:
    calibration = {
        field: {
            getattr(record, field): record
            for record in records
            if record.split is ManifestSplit.CALIBRATION
        }
        for field in (
            "source_sample_id",
            "source_image_id",
            "field_of_view_id",
            "image_sha256",
        )
    }
    for record in records:
        if record.split is not ManifestSplit.INDEPENDENT_TEST:
            continue
        for field, values in calibration.items():
            previous = values.get(getattr(record, field))
            if previous is not None:
                raise _error(
                    record.line_number,
                    field,
                    f"overlaps calibration row at line {previous.line_number}",
                )


def _validate_required_formal_splits(
    records: tuple[SplitManifestRecord, ...],
    *,
    require_calibration: bool,
) -> None:
    if require_calibration and not any(
        record.split is ManifestSplit.CALIBRATION and record.included
        for record in records
    ):
        raise _error(
            1,
            "split",
            "SELECTED requires calibration: "
            "at least one included=true calibration row is required",
        )
    if not any(
        record.split is ManifestSplit.INDEPENDENT_TEST and record.included
        for record in records
    ):
        raise _error(
            1,
            "split",
            "independent_test is required: "
            "at least one included=true independent_test row is required",
        )


def load_split_manifest(
    path: Path,
    *,
    require_calibration: bool = True,
) -> SmallBSplitManifest:
    """Load and validate one Small-B split-manifest CSV.

    The default protects Calibration/SELECTED workflows.  Only callers that
    have already authenticated a non-calibrated frozen-parameter contract may
    explicitly allow an empty Calibration split.
    """

    if not isinstance(require_calibration, bool):
        raise TypeError("require_calibration must be a bool")
    resolved = path.expanduser().resolve(strict=True)
    if not resolved.is_file():
        raise SplitManifestError(f"line 1, field 'manifest': not a file: {resolved}")

    with resolved.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        fieldnames = reader.fieldnames or []
        missing = [field for field in REQUIRED_COLUMNS if field not in fieldnames]
        if missing:
            raise _error(1, missing[0], "required column is missing")
        records = tuple(
            _parse_record(row, line_number=reader.line_num)
            for row in reader
        )

    _validate_unique_sample_ids(records)
    _validate_no_cross_split_source_or_sha(records)
    _validate_calibration_test_disjoint(records)
    _validate_required_formal_splits(
        records,
        require_calibration=require_calibration,
    )
    return SmallBSplitManifest(records)
