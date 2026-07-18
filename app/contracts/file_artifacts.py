"""Strict contracts for immutable, tenant-scoped file artifact metadata."""

from __future__ import annotations

import re
from datetime import datetime
from enum import StrEnum
from pathlib import PurePosixPath
from typing import Self

from pydantic import ConfigDict, Field, field_validator, model_validator

from app.contracts.common import ContractModel

_ARTIFACT_ID_PATTERN = re.compile(r"\Aart_[0-9a-f]{32}\Z")
_SHA256_PATTERN = re.compile(r"\A[0-9a-f]{64}\Z")
_MEDIA_TYPE_PATTERN = re.compile(r"\A[a-z0-9][a-z0-9!#$&^_.+-]*/[a-z0-9][a-z0-9!#$&^_.+-]*\Z")


def _has_control_character(value: str) -> bool:
    return any(ord(character) < 0x20 or 0x7F <= ord(character) <= 0x9F for character in value)


def validate_artifact_id(value: str) -> str:
    """Return one canonical registry ID without echoing malformed input."""

    if not isinstance(value, str) or _ARTIFACT_ID_PATTERN.fullmatch(value) is None:
        raise ValueError("invalid artifact ID")
    return value


class FileArtifactKind(StrEnum):
    """Frozen relationship shapes understood by the file-token boundary."""

    ORIGINAL_IMAGE = "original_image"
    RUN_ARTIFACT = "run_artifact"
    ANALYSIS_EXPORT = "analysis_export"
    CORRECTED_MASK_INPUT = "corrected_mask_input"


class FileArtifactState(StrEnum):
    """One-way lifecycle for registered bytes."""

    ACTIVE = "active"
    CONSUMED = "consumed"
    REVOKED = "revoked"


class _FileArtifactFacts(ContractModel):
    """Facts that never change after one storage path is registered."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        populate_by_name=True,
        strict=True,
        use_enum_values=False,
    )

    job_id: str = Field(min_length=1, max_length=64)
    image_id: str | None = Field(default=None, min_length=1, max_length=64)
    run_id: str | None = Field(default=None, min_length=1, max_length=64)
    artifact_kind: FileArtifactKind
    storage_path: str = Field(min_length=1, max_length=4096)
    filename: str = Field(min_length=1, max_length=255)
    media_type: str = Field(min_length=3, max_length=255)
    sha256: str = Field(min_length=64, max_length=64)
    size_bytes: int = Field(ge=0)

    @field_validator("storage_path")
    @classmethod
    def validate_storage_path(cls, value: str) -> str:
        if (
            "\\" in value
            or "\x00" in value
            or _has_control_character(value)
            or value != value.strip()
            or value.endswith("/")
        ):
            raise ValueError("storage_path must be a canonical relative POSIX path")
        path = PurePosixPath(value)
        if (
            path.is_absolute()
            or value != path.as_posix()
            or any(part in {"", ".", ".."} for part in path.parts)
        ):
            raise ValueError("storage_path must be a canonical relative POSIX path")
        return value

    @field_validator("filename")
    @classmethod
    def validate_filename(cls, value: str) -> str:
        if (
            value != value.strip()
            or value in {".", ".."}
            or "/" in value
            or "\\" in value
            or "\x00" in value
            or _has_control_character(value)
        ):
            raise ValueError("filename must be a plain basename")
        return value

    @field_validator("media_type")
    @classmethod
    def validate_media_type(cls, value: str) -> str:
        if _has_control_character(value) or _MEDIA_TYPE_PATTERN.fullmatch(value) is None:
            raise ValueError("media_type must be a canonical lowercase MIME type")
        return value

    @field_validator("sha256")
    @classmethod
    def validate_sha256(cls, value: str) -> str:
        if _SHA256_PATTERN.fullmatch(value) is None:
            raise ValueError("sha256 must be a lowercase hexadecimal digest")
        return value

    @model_validator(mode="after")
    def validate_relationship_shape(self) -> Self:
        if self.artifact_kind is FileArtifactKind.ORIGINAL_IMAGE:
            valid = self.image_id is not None and self.run_id is None
        elif self.artifact_kind in {
            FileArtifactKind.RUN_ARTIFACT,
            FileArtifactKind.CORRECTED_MASK_INPUT,
        }:
            valid = self.image_id is not None and self.run_id is not None
        else:
            valid = self.image_id is None and self.run_id is None
        if not valid:
            raise ValueError("artifact kind has an invalid image/run relationship shape")
        return self


class FileArtifactRegistration(_FileArtifactFacts):
    """Caller-supplied immutable facts; persistence allocates the registry ID."""


class FileArtifactDTO(_FileArtifactFacts):
    """Persisted artifact facts plus their one-way lifecycle state."""

    artifact_id: str = Field(min_length=36, max_length=36)
    state: FileArtifactState
    created_at: datetime
    consumed_at: datetime | None = None
    revoked_at: datetime | None = None

    @field_validator("artifact_id")
    @classmethod
    def validate_artifact_id(cls, value: str) -> str:
        return validate_artifact_id(value)

    @field_validator("created_at", "consumed_at", "revoked_at")
    @classmethod
    def validate_aware_datetime(cls, value: datetime | None) -> datetime | None:
        if value is not None and (value.tzinfo is None or value.utcoffset() is None):
            raise ValueError("artifact timestamps must be timezone-aware")
        return value

    @model_validator(mode="after")
    def validate_lifecycle(self) -> Self:
        if self.state is FileArtifactState.ACTIVE:
            valid = self.consumed_at is None and self.revoked_at is None
        elif self.state is FileArtifactState.CONSUMED:
            valid = (
                self.artifact_kind is FileArtifactKind.CORRECTED_MASK_INPUT
                and self.consumed_at is not None
                and self.revoked_at is None
                and self.consumed_at >= self.created_at
            )
        else:
            valid = (
                self.consumed_at is None
                and self.revoked_at is not None
                and self.revoked_at >= self.created_at
            )
        if not valid:
            raise ValueError("artifact lifecycle timestamps are inconsistent with state")
        return self
