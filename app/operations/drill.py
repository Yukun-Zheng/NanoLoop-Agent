"""Repeatable, explicitly limited offline filesystem-restore drills."""

from __future__ import annotations

import math
import os
import stat
import tempfile
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from itertools import pairwise
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.operations.backup import (
    BackupComponent,
    BackupLayout,
    BackupManifest,
    BackupResult,
    BackupVerificationResult,
    RestoreResult,
    create_backup,
    restore_backup,
    verify_backup,
)

Clock = Callable[[], datetime]
MonotonicClock = Callable[[], float]


class DrillError(RuntimeError):
    """Base class for offline restore-drill failures."""


class DrillValidationError(DrillError, ValueError):
    """The drill inputs or results could not form a truthful report."""


class DrillDurations(BaseModel):
    """Measured local operation durations; these are not an RTO claim."""

    model_config = ConfigDict(extra="forbid", strict=True, frozen=True, allow_inf_nan=False)

    backup_seconds: float = Field(ge=0)
    verification_seconds: float = Field(ge=0)
    filesystem_restore_seconds: float = Field(ge=0)
    total_seconds: float = Field(ge=0)


class DrillCounts(BaseModel):
    """Non-sensitive inventory totals from the verified archive manifest."""

    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    file_count: int = Field(ge=0)
    total_bytes: int = Field(ge=0)
    component_count: int = Field(ge=0)
    files_by_component: dict[BackupComponent, int]
    bytes_by_component: dict[BackupComponent, int]

    @model_validator(mode="after")
    def component_totals_are_complete(self) -> DrillCounts:
        expected = set(BackupComponent)
        if set(self.files_by_component) != expected:
            raise ValueError("files_by_component must contain every backup component")
        if set(self.bytes_by_component) != expected:
            raise ValueError("bytes_by_component must contain every backup component")
        if any(value < 0 for value in self.files_by_component.values()):
            raise ValueError("component file counts cannot be negative")
        if any(value < 0 for value in self.bytes_by_component.values()):
            raise ValueError("component byte counts cannot be negative")
        if sum(self.files_by_component.values()) != self.file_count:
            raise ValueError("component file counts do not match file_count")
        if sum(self.bytes_by_component.values()) != self.total_bytes:
            raise ValueError("component byte counts do not match total_bytes")
        if self.component_count != len(expected):
            raise ValueError("component_count must include every backup component")
        return self


class OfflineFilesystemRestoreDrillReport(BaseModel):
    """Strict success report for the limited create/verify/restore exercise."""

    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    schema_version: Literal[1] = 1
    status: Literal["success"] = "success"
    scope: Literal["offline_filesystem_restore"] = "offline_filesystem_restore"
    started_at: datetime
    snapshot_at: datetime
    completed_at: datetime
    durations: DrillDurations
    archive_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    database_revision: str = Field(min_length=1, max_length=255)
    counts: DrillCounts
    application_startup_verified: Literal[False] = False
    rpo: Literal["not_measured"] = "not_measured"
    rto: Literal["not_measured"] = "not_measured"

    @model_validator(mode="after")
    def timestamps_are_utc(self) -> OfflineFilesystemRestoreDrillReport:
        for field_name in ("started_at", "snapshot_at", "completed_at"):
            value = getattr(self, field_name)
            if value.tzinfo is None or value.utcoffset() != UTC.utcoffset(value):
                raise ValueError(f"{field_name} must use UTC")
        if not self.started_at <= self.snapshot_at <= self.completed_at:
            raise ValueError("timestamps must satisfy started_at <= snapshot_at <= completed_at")
        return self


@dataclass(frozen=True, slots=True)
class OfflineFilesystemRestoreDrillResult:
    """Published drill report and its in-memory validated representation."""

    report_path: Path
    report: OfflineFilesystemRestoreDrillReport


def run_offline_filesystem_restore_drill(
    layout: BackupLayout,
    archive_path: str | Path,
    destination_root: str | Path,
    report_path: str | Path,
    *,
    offline_confirmed: bool,
    clock: Clock | None = None,
    monotonic: MonotonicClock | None = None,
) -> OfflineFilesystemRestoreDrillResult:
    """Create, verify, and restore once, then publish a narrowly scoped report.

    This function intentionally does not start the application and therefore does
    not report an RPO or RTO. The archive, restore root, checksum sidecar, and report
    are all new-only targets.
    """

    if offline_confirmed is not True:
        raise DrillValidationError("offline_confirmed=True is required")
    archive = Path(archive_path).expanduser().absolute()
    destination = Path(destination_root).expanduser().absolute()
    report = _resolve_new_report_path(report_path)
    _preflight_new_targets(archive, destination, report)

    active_clock = clock or _utc_now
    active_monotonic = monotonic or time.monotonic
    started_at = _read_utc_clock(active_clock, "started_at")
    tick_start = _read_monotonic(active_monotonic)

    created = create_backup(layout, archive, offline_confirmed=True)
    tick_after_backup = _read_monotonic(active_monotonic)
    verified = verify_backup(created.archive_path, checksum_path=created.checksum_path)
    tick_after_verification = _read_monotonic(active_monotonic)
    restored = restore_backup(
        verified.archive_path,
        destination,
        offline_confirmed=True,
        checksum_path=verified.checksum_path,
    )
    tick_after_restore = _read_monotonic(active_monotonic)
    completed_at = _read_utc_clock(active_clock, "completed_at")

    _validate_stage_ticks(
        tick_start,
        tick_after_backup,
        tick_after_verification,
        tick_after_restore,
    )
    manifest = _validated_consistent_results(created, verified, restored)
    drill_report = OfflineFilesystemRestoreDrillReport(
        started_at=started_at,
        snapshot_at=_as_utc(manifest.created_at, "snapshot_at"),
        completed_at=completed_at,
        durations=DrillDurations(
            backup_seconds=_duration(tick_start, tick_after_backup),
            verification_seconds=_duration(
                tick_after_backup,
                tick_after_verification,
            ),
            filesystem_restore_seconds=_duration(
                tick_after_verification,
                tick_after_restore,
            ),
            total_seconds=_duration(tick_start, tick_after_restore),
        ),
        archive_sha256=verified.archive_sha256,
        database_revision=manifest.database_revision,
        counts=_manifest_counts(manifest),
    )
    _publish_report(report, drill_report)
    return OfflineFilesystemRestoreDrillResult(report_path=report, report=drill_report)


def _validated_consistent_results(
    created: BackupResult,
    verified: BackupVerificationResult,
    restored: RestoreResult,
) -> BackupManifest:
    digests = {
        created.archive_sha256,
        verified.archive_sha256,
        restored.archive_sha256,
    }
    if len(digests) != 1:
        raise DrillValidationError("drill stages did not use one archive digest")
    if created.manifest != verified.manifest or verified.manifest != restored.manifest:
        raise DrillValidationError("drill stages did not use one backup manifest")
    return verified.manifest


def _manifest_counts(manifest: BackupManifest) -> DrillCounts:
    files_by_component = {component: 0 for component in BackupComponent}
    bytes_by_component = {component: 0 for component in BackupComponent}
    for record in manifest.files:
        files_by_component[record.component] += 1
        bytes_by_component[record.component] += record.size
    return DrillCounts(
        file_count=len(manifest.files),
        total_bytes=sum(record.size for record in manifest.files),
        component_count=len(manifest.components),
        files_by_component=files_by_component,
        bytes_by_component=bytes_by_component,
    )


def _resolve_new_report_path(value: str | Path) -> Path:
    supplied = Path(value).expanduser().absolute()
    if supplied.name in {"", ".", ".."}:
        raise DrillValidationError("report path must name a file")
    try:
        parent_state = supplied.parent.lstat()
    except OSError as error:
        raise DrillValidationError("report destination parent does not exist") from error
    if stat.S_ISLNK(parent_state.st_mode) or not stat.S_ISDIR(parent_state.st_mode):
        raise DrillValidationError("report destination parent must be a non-symlink directory")
    parent = supplied.parent.resolve(strict=True)
    return parent / supplied.name


def _preflight_new_targets(archive: Path, destination: Path, report: Path) -> None:
    checksum = Path(f"{archive}.sha256")
    if len({archive, checksum, report}) != 3:
        raise DrillValidationError("archive, checksum, and report paths must be distinct")
    if report == destination or report.is_relative_to(destination):
        raise DrillValidationError("report cannot be located inside the restore root")
    for candidate, label in (
        (archive, "archive"),
        (checksum, "checksum"),
        (destination, "restore destination"),
        (report, "report"),
    ):
        if os.path.lexists(candidate):
            raise FileExistsError(f"{label} already exists")


def _publish_report(path: Path, report: OfflineFilesystemRestoreDrillReport) -> None:
    payload = report.model_dump_json(indent=2).encode("utf-8") + b"\n"
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    temporary = Path(temporary_name)
    linked = False
    complete = False
    try:
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path, follow_symlinks=False)
        linked = True
        _fsync_directory(path.parent)
        complete = True
    except FileExistsError as error:
        raise FileExistsError("drill report already exists") from error
    finally:
        if linked and not complete and _same_inode(path, temporary):
            path.unlink(missing_ok=True)
            _fsync_directory(path.parent)
        temporary.unlink(missing_ok=True)


def _same_inode(first: Path, second: Path) -> bool:
    try:
        left = first.stat(follow_symlinks=False)
        right = second.stat(follow_symlinks=False)
    except OSError:
        return False
    return left.st_dev == right.st_dev and left.st_ino == right.st_ino


def _read_utc_clock(clock: Clock, label: str) -> datetime:
    return _as_utc(clock(), label)


def _as_utc(value: datetime, label: str) -> datetime:
    if not isinstance(value, datetime) or value.tzinfo is None:
        raise DrillValidationError(f"{label} clock must return a timezone-aware datetime")
    return value.astimezone(UTC)


def _read_monotonic(clock: MonotonicClock) -> float:
    value = clock()
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise DrillValidationError("monotonic clock must return a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise DrillValidationError("monotonic clock must return a finite number")
    return result


def _validate_stage_ticks(*ticks: float) -> None:
    if any(current < previous for previous, current in pairwise(ticks)):
        raise DrillValidationError("monotonic clock moved backwards")


def _duration(start: float, end: float) -> float:
    return round(end - start, 6)


def _utc_now() -> datetime:
    return datetime.now(UTC)


def _fsync_directory(path: Path) -> None:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_CLOEXEC", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError:
        return
    try:
        os.fsync(descriptor)
    except OSError:
        pass
    finally:
        os.close(descriptor)


__all__ = [
    "DrillCounts",
    "DrillDurations",
    "DrillError",
    "DrillValidationError",
    "OfflineFilesystemRestoreDrillReport",
    "OfflineFilesystemRestoreDrillResult",
    "run_offline_filesystem_restore_drill",
]
