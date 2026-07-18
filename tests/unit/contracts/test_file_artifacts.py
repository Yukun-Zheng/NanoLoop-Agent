from datetime import UTC, datetime, timedelta

import pytest
from pydantic import ValidationError

from app.contracts.file_artifacts import (
    FileArtifactDTO,
    FileArtifactKind,
    FileArtifactRegistration,
    FileArtifactState,
)


def _registration(**updates: object) -> FileArtifactRegistration:
    values: dict[str, object] = {
        "job_id": "job_a",
        "image_id": "image_a",
        "run_id": None,
        "artifact_kind": FileArtifactKind.ORIGINAL_IMAGE,
        "storage_path": "jobs/job_a/input/image_a/original.tif",
        "filename": "sample.tif",
        "media_type": "image/tiff",
        "sha256": "a" * 64,
        "size_bytes": 1024,
    }
    values.update(updates)
    return FileArtifactRegistration(**values)


@pytest.mark.parametrize(
    ("artifact_kind", "image_id", "run_id"),
    [
        (FileArtifactKind.ORIGINAL_IMAGE, "image_a", None),
        (FileArtifactKind.RUN_ARTIFACT, "image_a", "run_a"),
        (FileArtifactKind.CORRECTED_MASK_INPUT, "image_a", "run_a"),
        (FileArtifactKind.ANALYSIS_EXPORT, None, None),
    ],
)
def test_registration_accepts_only_frozen_relationship_shapes(
    artifact_kind: FileArtifactKind,
    image_id: str | None,
    run_id: str | None,
) -> None:
    registration = _registration(
        artifact_kind=artifact_kind,
        image_id=image_id,
        run_id=run_id,
    )
    assert registration.artifact_kind is artifact_kind


@pytest.mark.parametrize(
    ("artifact_kind", "image_id", "run_id"),
    [
        (FileArtifactKind.ORIGINAL_IMAGE, None, None),
        (FileArtifactKind.ORIGINAL_IMAGE, "image_a", "run_a"),
        (FileArtifactKind.RUN_ARTIFACT, None, "run_a"),
        (FileArtifactKind.RUN_ARTIFACT, "image_a", None),
        (FileArtifactKind.CORRECTED_MASK_INPUT, None, "run_a"),
        (FileArtifactKind.ANALYSIS_EXPORT, "image_a", None),
        (FileArtifactKind.ANALYSIS_EXPORT, None, "run_a"),
    ],
)
def test_registration_rejects_ambiguous_relationship_shapes(
    artifact_kind: FileArtifactKind,
    image_id: str | None,
    run_id: str | None,
) -> None:
    with pytest.raises(ValidationError, match="relationship shape"):
        _registration(
            artifact_kind=artifact_kind,
            image_id=image_id,
            run_id=run_id,
        )


@pytest.mark.parametrize(
    "storage_path",
    [
        "/absolute/file.tif",
        "../escape/file.tif",
        "job/../escape.tif",
        "job/./file.tif",
        "job//file.tif",
        "job\\file.tif",
        "job/file.tif/",
        "job/line\nfeed.tif",
    ],
)
def test_registration_rejects_noncanonical_storage_paths(storage_path: str) -> None:
    with pytest.raises(ValidationError, match="relative POSIX path"):
        _registration(storage_path=storage_path)


@pytest.mark.parametrize(
    "filename",
    [
        "../sample.tif",
        "nested/sample.tif",
        "sample\\file.tif",
        "sample\r\ncontent-disposition.tif",
        "sample\x00.tif",
        "sample\u0085.tif",
    ],
)
def test_registration_rejects_unsafe_or_controlled_filenames(filename: str) -> None:
    with pytest.raises(ValidationError, match="plain basename"):
        _registration(filename=filename)


@pytest.mark.parametrize("media_type", ["Image/TIFF", "image/tiff\r\nx-test", "image/\u0085tiff"])
def test_registration_rejects_noncanonical_or_controlled_media_types(
    media_type: str,
) -> None:
    with pytest.raises(ValidationError, match="MIME type"):
        _registration(media_type=media_type)


def test_registration_is_strict_and_forbids_unknown_fields() -> None:
    with pytest.raises(ValidationError):
        _registration(size_bytes="1024")
    with pytest.raises(ValidationError):
        FileArtifactRegistration(
            **_registration().model_dump(mode="python"),
            unknown=True,
        )


def test_persisted_lifecycle_requires_canonical_id_aware_time_and_terminal_shape() -> None:
    created_at = datetime.now(UTC)
    active = FileArtifactDTO(
        **_registration().model_dump(mode="python"),
        artifact_id=f"art_{'1' * 32}",
        state=FileArtifactState.ACTIVE,
        created_at=created_at,
    )
    assert active.state is FileArtifactState.ACTIVE

    corrected_facts = _registration(
        artifact_kind=FileArtifactKind.CORRECTED_MASK_INPUT,
        image_id="image_a",
        run_id="run_a",
    ).model_dump(mode="python")
    consumed = FileArtifactDTO(
        **corrected_facts,
        artifact_id=f"art_{'2' * 32}",
        state=FileArtifactState.CONSUMED,
        created_at=created_at,
        consumed_at=created_at + timedelta(seconds=1),
    )
    assert consumed.state is FileArtifactState.CONSUMED

    revoked = FileArtifactDTO(
        **_registration().model_dump(mode="python"),
        artifact_id=f"art_{'3' * 32}",
        state=FileArtifactState.REVOKED,
        created_at=created_at,
        revoked_at=created_at + timedelta(seconds=1),
    )
    assert revoked.state is FileArtifactState.REVOKED

    with pytest.raises(ValidationError, match="lifecycle"):
        FileArtifactDTO(
            **_registration().model_dump(mode="python"),
            artifact_id=f"art_{'4' * 32}",
            state=FileArtifactState.CONSUMED,
            created_at=created_at,
            consumed_at=created_at,
        )
    with pytest.raises(ValidationError, match="timezone-aware"):
        FileArtifactDTO(
            **_registration().model_dump(mode="python"),
            artifact_id=f"art_{'5' * 32}",
            state=FileArtifactState.ACTIVE,
            created_at=created_at.replace(tzinfo=None),
        )
    with pytest.raises(ValidationError, match="invalid artifact ID"):
        FileArtifactDTO(
            **_registration().model_dump(mode="python"),
            artifact_id=f"art_{'g' * 32}",
            state=FileArtifactState.ACTIVE,
            created_at=created_at,
        )
