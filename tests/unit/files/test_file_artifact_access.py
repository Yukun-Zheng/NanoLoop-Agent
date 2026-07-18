from __future__ import annotations

import base64
import hashlib
import json
import os
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import pytest
from sqlalchemy import func, select

import app.files.application as file_application
from app.contracts.file_artifacts import FileArtifactKind, FileArtifactState
from app.contracts.identity import (
    LEGACY_PRINCIPAL_ID,
    LEGACY_TENANT_ID,
    AuthMode,
    PrincipalContext,
    PrincipalKind,
    PrincipalRole,
)
from app.core.config import Settings
from app.core.identity import legacy_principal_context
from app.db.base import Base
from app.db.models import (
    AnalysisJob,
    ApiCredential,
    FileArtifact,
    ImageAsset,
    ModelRegistryRecord,
    Principal,
    SegmentationRun,
    Tenant,
)
from app.db.repositories import SqlAlchemyUnitOfWork
from app.db.session import Database
from app.files import (
    FileAccessTokenError,
    FileArtifactAccessService,
    FileArtifactUnavailableError,
)
from app.storage import FileTokenV2KeyRing, LocalFileStore, PinnedManagedFile, StoragePaths

_TENANT_A = f"tnt_{'a' * 32}"
_TENANT_B = f"tnt_{'b' * 32}"
_PRINCIPAL_A = f"prn_{'a' * 32}"
_PRINCIPAL_A_PEER = f"prn_{'b' * 32}"
_PRINCIPAL_B = f"prn_{'c' * 32}"
_LEGACY_PEER = f"prn_{'d' * 32}"
_CREDENTIAL_A = f"crd_{'a' * 32}"
_CREDENTIAL_A_PEER = f"crd_{'b' * 32}"
_CREDENTIAL_B = f"crd_{'c' * 32}"
_MODEL_ID = "file-access-model"


@dataclass(frozen=True, slots=True)
class _Harness:
    database: Database
    file_store: LocalFileStore
    keyring: FileTokenV2KeyRing
    service: FileArtifactAccessService
    principal_a: PrincipalContext
    principal_a_peer: PrincipalContext
    principal_b: PrincipalContext


@pytest.fixture
def harness(tmp_path: Path) -> Iterator[_Harness]:
    database = Database(Settings(database_url=f"sqlite:///{tmp_path / 'files.db'}"))
    Base.metadata.create_all(database.engine)
    _seed_database(database)
    file_store = LocalFileStore(
        StoragePaths(tmp_path / "outputs"),
        max_upload_bytes=1024 * 1024,
        token_secret=b"legacy-file-token-secret-material",
    )
    keyring = FileTokenV2KeyRing(
        {"test-key": b"file-token-v2-test-key-material-0001"},
        active_kid="test-key",
        clock_skew_seconds=0,
    )
    service = _service(database, file_store=file_store, keyring=keyring)
    result = _Harness(
        database=database,
        file_store=file_store,
        keyring=keyring,
        service=service,
        principal_a=_principal(
            tenant_id=_TENANT_A,
            principal_id=_PRINCIPAL_A,
            credential_id=_CREDENTIAL_A,
            role=PrincipalRole.ANALYST,
        ),
        principal_a_peer=_principal(
            tenant_id=_TENANT_A,
            principal_id=_PRINCIPAL_A_PEER,
            credential_id=_CREDENTIAL_A_PEER,
            role=PrincipalRole.TENANT_ADMIN,
        ),
        principal_b=_principal(
            tenant_id=_TENANT_B,
            principal_id=_PRINCIPAL_B,
            credential_id=_CREDENTIAL_B,
            role=PrincipalRole.ANALYST,
        ),
    )
    try:
        yield result
    finally:
        database.dispose()


def test_new_issuance_is_pathless_v2_and_registration_is_idempotent(
    harness: _Harness,
) -> None:
    relative_path = "job_a/input/img_a/original.png"
    _write_file(harness, relative_path, b"immutable-original")

    first = harness.service.issue_download_token(
        principal=harness.principal_a,
        job_id="job_a",
        image_id="img_a",
        artifact_kind=FileArtifactKind.ORIGINAL_IMAGE,
        storage_path=relative_path,
        filename="sample.png",
        media_type="image/png",
    )
    second = harness.service.issue_download_token(
        principal=harness.principal_a,
        job_id="job_a",
        image_id="img_a",
        artifact_kind=FileArtifactKind.ORIGINAL_IMAGE,
        storage_path=relative_path,
        filename="sample.png",
        media_type="image/png",
    )

    first_claims = harness.keyring.verify(first)
    second_claims = harness.keyring.verify(second)
    payload = _token_payload(first)
    assert first.startswith("v2.test-key.")
    assert first_claims.aid == second_claims.aid
    assert payload == first_claims.as_payload()
    assert set(payload) == {
        "aid",
        "aud",
        "exp",
        "iat",
        "jid",
        "jti",
        "nbf",
        "pur",
        "sha256",
        "sub",
        "tid",
        "v",
    }
    assert "path" not in payload
    assert "credential_id" not in payload
    assert relative_path not in first
    assert _CREDENTIAL_A not in first
    assert _artifact_count(harness) == 1


def test_subject_and_tenant_mismatches_fail_before_filesystem_open(
    harness: _Harness,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    token, _path, _content = _issue_original(harness)
    open_calls = 0

    def unexpected_open(*_args: object, **_kwargs: object) -> PinnedManagedFile:
        nonlocal open_calls
        open_calls += 1
        raise AssertionError("filesystem open must follow identity verification")

    monkeypatch.setattr(file_application, "open_pinned_managed_file", unexpected_open)

    with pytest.raises(FileAccessTokenError, match=r"^invalid file token$"):
        harness.service.resolve_download(token, principal=harness.principal_a_peer)
    with pytest.raises(FileAccessTokenError, match=r"^invalid file token$"):
        harness.service.resolve_download(token, principal=harness.principal_b)

    assert open_calls == 0


def test_download_and_review_purposes_cannot_be_interchanged(harness: _Harness) -> None:
    download_token, _path, _content = _issue_original(harness)
    review_token, _review_path, _review_content = _issue_corrected_mask(harness)

    with pytest.raises(FileAccessTokenError, match=r"^invalid corrected-mask token$"):
        harness.service.resolve_corrected_mask(
            download_token,
            principal=harness.principal_a,
            job_id="job_a",
            image_id="img_a",
            run_id="run_a",
        )
    with pytest.raises(FileAccessTokenError, match=r"^invalid file token$"):
        harness.service.resolve_download(review_token, principal=harness.principal_a)


def test_revoked_registry_entry_and_removed_parent_invalidate_download(
    harness: _Harness,
) -> None:
    token, _path, _content = _issue_original(harness)
    artifact_id = harness.keyring.verify(token).aid
    with harness.database.session_factory() as session:
        record = session.get(FileArtifact, artifact_id)
        assert record is not None
        record.state = FileArtifactState.REVOKED.value
        record.revoked_at = datetime.now(UTC)
        session.commit()

    with pytest.raises(FileAccessTokenError, match=r"^invalid file token$"):
        harness.service.resolve_download(token, principal=harness.principal_a)

    removed_path = "job_a2/input/img_a2/original.png"
    _write_file(harness, removed_path, b"second-original")
    removed_token = harness.service.issue_download_token(
        principal=harness.principal_a,
        job_id="job_a2",
        image_id="img_a2",
        artifact_kind=FileArtifactKind.ORIGINAL_IMAGE,
        storage_path=removed_path,
    )
    with harness.database.session_factory() as session:
        job = session.get(AnalysisJob, "job_a2")
        assert job is not None
        session.delete(job)
        session.commit()

    with pytest.raises(FileAccessTokenError, match=r"^invalid file token$"):
        harness.service.resolve_download(removed_token, principal=harness.principal_a)


def test_consumed_corrected_mask_registry_entry_is_rejected(harness: _Harness) -> None:
    token, _path, _content = _issue_corrected_mask(harness)
    artifact_id = harness.keyring.verify(token).aid
    with SqlAlchemyUnitOfWork(harness.database.session_factory) as uow:
        assert uow.repositories.file_artifacts.consume_corrected_mask(
            artifact_id,
            tenant_id=_TENANT_A,
        )
        uow.commit()

    with pytest.raises(FileAccessTokenError, match=r"^invalid corrected-mask token$"):
        harness.service.resolve_corrected_mask(
            token,
            principal=harness.principal_a,
            job_id="job_a",
            image_id="img_a",
            run_id="run_a",
        )


def test_atomic_path_replacement_with_stale_hash_is_rejected(harness: _Harness) -> None:
    token, path, content = _issue_original(harness)
    replacement = path.with_name("replacement.tmp")
    replacement.write_bytes(b"changed-content-with-a-different-hash")
    os.replace(replacement, path)

    assert path.read_bytes() != content
    with pytest.raises(FileAccessTokenError, match=r"^invalid file token$"):
        harness.service.resolve_download(token, principal=harness.principal_a)


def test_resolved_download_streams_pinned_inode_after_atomic_path_replacement(
    harness: _Harness,
) -> None:
    token, path, content = _issue_original(harness)
    resolved = harness.service.resolve_download(token, principal=harness.principal_a)
    replacement = path.with_name("replacement.tmp")
    replacement.write_bytes(b"replacement-inode")
    os.replace(replacement, path)

    assert path.read_bytes() == b"replacement-inode"
    assert b"".join(resolved.pinned_file.iter_chunks()) == content
    assert resolved.pinned_file.closed


def test_principal_mode_rejects_v1_before_decode_or_filesystem_open(
    harness: _Harness,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    relative_path = "job_a/input/img_a/original.png"
    path = _write_file(harness, relative_path, b"principal-v1")
    token = harness.file_store.create_file_token(path)
    decode_calls = 0
    open_calls = 0

    def unexpected_decode(_token: str, *, now: int | None = None) -> str:
        del now
        nonlocal decode_calls
        decode_calls += 1
        raise AssertionError("principal mode must not decode v1 tokens")

    def unexpected_open(*_args: object, **_kwargs: object) -> PinnedManagedFile:
        nonlocal open_calls
        open_calls += 1
        raise AssertionError("principal mode must not open v1 token paths")

    monkeypatch.setattr(harness.file_store, "decode_file_token_path", unexpected_decode)
    monkeypatch.setattr(file_application, "open_pinned_managed_file", unexpected_open)

    with pytest.raises(FileAccessTokenError, match=r"^invalid file token$"):
        harness.service.resolve_download(token, principal=harness.principal_a)

    assert decode_calls == 0
    assert open_calls == 0


@pytest.mark.parametrize("auth_mode", [AuthMode.DISABLED, AuthMode.SHARED_KEY])
def test_compatibility_modes_accept_v1_only_for_fixed_legacy_owned_jobs(
    harness: _Harness,
    monkeypatch: pytest.MonkeyPatch,
    auth_mode: AuthMode,
) -> None:
    legacy_path = _write_file(
        harness,
        "job_legacy/input/img_legacy/original.png",
        b"legacy-owned",
    )
    legacy_token = harness.file_store.create_file_token(legacy_path)
    principal = legacy_principal_context(auth_mode)
    resolved = harness.service.resolve_download(legacy_token, principal=principal)
    assert b"".join(resolved.pinned_file.iter_chunks()) == b"legacy-owned"
    assert resolved.artifact_id is None

    nonlegacy_path = _write_file(
        harness,
        "job_legacy_peer/input/original.png",
        b"not-fixed-legacy-owner",
    )
    nonlegacy_token = harness.file_store.create_file_token(nonlegacy_path)
    open_calls = 0

    def unexpected_open(*_args: object, **_kwargs: object) -> PinnedManagedFile:
        nonlocal open_calls
        open_calls += 1
        raise AssertionError("legacy ownership must be checked before opening")

    monkeypatch.setattr(file_application, "open_pinned_managed_file", unexpected_open)
    with pytest.raises(FileAccessTokenError, match=r"^invalid file token$"):
        harness.service.resolve_download(nonlegacy_token, principal=principal)
    assert open_calls == 0


def test_legacy_v1_corrected_mask_requires_review_path_and_registers_one_shot(
    harness: _Harness,
) -> None:
    principal = legacy_principal_context(AuthMode.DISABLED)
    canonical_path = _write_file(
        harness,
        "job_legacy/input/review_mask_stage/original.png",
        b"legacy-mask",
    )
    canonical_token = harness.file_store.create_file_token(canonical_path)

    resolved = harness.service.resolve_corrected_mask(
        canonical_token,
        principal=principal,
        job_id="job_legacy",
        image_id="img_legacy",
        run_id="run_legacy",
    )
    assert resolved.content == b"legacy-mask"
    assert resolved.legacy_v1 is True
    assert _artifact_count(harness) == 1

    with SqlAlchemyUnitOfWork(harness.database.session_factory) as uow:
        artifact = uow.repositories.file_artifacts.get_active(
            resolved.artifact_id,
            tenant_id=LEGACY_TENANT_ID,
        )
        assert artifact.artifact_kind is FileArtifactKind.CORRECTED_MASK_INPUT
        assert artifact.storage_path == "job_legacy/input/review_mask_stage/original.png"
        assert uow.repositories.file_artifacts.consume_corrected_mask(
            resolved.artifact_id,
            tenant_id=LEGACY_TENANT_ID,
        )
        assert not uow.repositories.file_artifacts.consume_corrected_mask(
            resolved.artifact_id,
            tenant_id=LEGACY_TENANT_ID,
        )
        uow.commit()

    with pytest.raises(FileAccessTokenError, match=r"^invalid corrected-mask token$"):
        harness.service.resolve_corrected_mask(
            canonical_token,
            principal=principal,
            job_id="job_legacy",
            image_id="img_legacy",
            run_id="run_legacy",
        )

    original_path = _write_file(
        harness,
        "job_legacy/input/img_legacy/original.png",
        b"durable-original",
    )
    original_token = harness.file_store.create_file_token(original_path)
    with pytest.raises(FileAccessTokenError, match=r"^invalid corrected-mask token$"):
        harness.service.resolve_corrected_mask(
            original_token,
            principal=principal,
            job_id="job_legacy",
            image_id="img_legacy",
            run_id="run_legacy",
        )
    assert _artifact_count(harness) == 1


def test_corrected_v2_is_exactly_parent_and_subject_bound_and_size_limited(
    harness: _Harness,
) -> None:
    token, _path, content = _issue_corrected_mask(harness)
    resolved = harness.service.resolve_corrected_mask(
        token,
        principal=harness.principal_a,
        job_id="job_a",
        image_id="img_a",
        run_id="run_a",
    )
    assert resolved.content == content

    mismatched_targets = (
        ("job_a2", "img_a2", "run_a2"),
        ("job_a", "img_a_other", "run_a_other"),
        ("job_a", "img_a", "run_a_alt"),
    )
    for job_id, image_id, run_id in mismatched_targets:
        with pytest.raises(FileAccessTokenError, match=r"^invalid corrected-mask token$"):
            harness.service.resolve_corrected_mask(
                token,
                principal=harness.principal_a,
                job_id=job_id,
                image_id=image_id,
                run_id=run_id,
            )

    with pytest.raises(FileAccessTokenError, match=r"^invalid corrected-mask token$"):
        harness.service.resolve_corrected_mask(
            token,
            principal=harness.principal_a_peer,
            job_id="job_a",
            image_id="img_a",
            run_id="run_a",
        )

    oversized_service = _service(
        harness.database,
        file_store=harness.file_store,
        keyring=harness.keyring,
        max_corrected_mask_bytes=4,
    )
    oversized_relative_path = "job_a/input/review_mask_oversized/original.png"
    oversized_content = b"12345"
    _write_file(harness, oversized_relative_path, oversized_content)
    artifact_count_before = _artifact_count(harness)
    with pytest.raises(FileArtifactUnavailableError) as oversized_error:
        oversized_service.issue_corrected_mask_token(
            principal=harness.principal_a,
            job_id="job_a",
            image_id="img_a",
            run_id="run_a",
            storage_path=oversized_relative_path,
            filename="original.png",
            media_type="image/png",
            expected_sha256=hashlib.sha256(oversized_content).hexdigest(),
            expected_size_bytes=len(oversized_content),
        )
    assert oversized_error.value.details == {
        "job_id": "job_a",
        "artifact_kind": FileArtifactKind.CORRECTED_MASK_INPUT.value,
    }
    assert oversized_relative_path not in str(oversized_error.value)
    assert _artifact_count(harness) == artifact_count_before


def test_resolution_errors_do_not_echo_token_or_managed_path(harness: _Harness) -> None:
    token, path, _content = _issue_original(harness)
    sensitive_path = "job_a/input/img_a/original.png"
    path.unlink()

    with pytest.raises(FileAccessTokenError) as stale_error:
        harness.service.resolve_download(token, principal=harness.principal_a)
    rendered = f"{stale_error.value!r} {stale_error.value}"
    assert rendered == "FileAccessTokenError('invalid file token') invalid file token"
    assert token not in rendered
    assert sensitive_path not in rendered

    malformed = f"v2.test-key.{base64.urlsafe_b64encode(sensitive_path.encode()).decode()}.bad"
    with pytest.raises(FileAccessTokenError) as malformed_error:
        harness.service.resolve_download(malformed, principal=harness.principal_a)
    assert str(malformed_error.value) == "invalid file token"
    assert malformed not in str(malformed_error.value)
    assert sensitive_path not in str(malformed_error.value)


def _service(
    database: Database,
    *,
    file_store: LocalFileStore,
    keyring: FileTokenV2KeyRing,
    max_corrected_mask_bytes: int | None = None,
) -> FileArtifactAccessService:
    return FileArtifactAccessService(
        uow_factory=lambda: SqlAlchemyUnitOfWork(database.session_factory),
        file_store=file_store,
        keyring=keyring,
        max_corrected_mask_bytes=max_corrected_mask_bytes,
    )


def _principal(
    *,
    tenant_id: str,
    principal_id: str,
    credential_id: str,
    role: PrincipalRole,
) -> PrincipalContext:
    return PrincipalContext(
        tenant_id=tenant_id,
        principal_id=principal_id,
        credential_id=credential_id,
        kind=PrincipalKind.USER,
        role=role,
        auth_mode=AuthMode.PRINCIPAL,
    )


def _seed_database(database: Database) -> None:
    now = datetime(2026, 7, 18, tzinfo=UTC)
    with database.session_factory() as session:
        session.add_all(
            [
                Tenant(
                    tenant_id=_TENANT_A,
                    slug="file-access-a",
                    display_name="File Access A",
                    enabled=True,
                    version=1,
                    created_at=now,
                    updated_at=now,
                ),
                Tenant(
                    tenant_id=_TENANT_B,
                    slug="file-access-b",
                    display_name="File Access B",
                    enabled=True,
                    version=1,
                    created_at=now,
                    updated_at=now,
                ),
            ]
        )
        session.flush()
        session.add_all(
            [
                _principal_record(
                    _PRINCIPAL_A,
                    _TENANT_A,
                    handle="file-owner-a",
                    role=PrincipalRole.ANALYST,
                    now=now,
                ),
                _principal_record(
                    _PRINCIPAL_A_PEER,
                    _TENANT_A,
                    handle="file-admin-a",
                    role=PrincipalRole.TENANT_ADMIN,
                    now=now,
                ),
                _principal_record(
                    _PRINCIPAL_B,
                    _TENANT_B,
                    handle="file-owner-b",
                    role=PrincipalRole.ANALYST,
                    now=now,
                ),
                _principal_record(
                    _LEGACY_PEER,
                    LEGACY_TENANT_ID,
                    handle="legacy-peer",
                    role=PrincipalRole.ANALYST,
                    now=now,
                ),
            ]
        )
        session.flush()
        session.add_all(
            [
                _credential(_CREDENTIAL_A, _PRINCIPAL_A, b"a" * 32, now=now),
                _credential(
                    _CREDENTIAL_A_PEER,
                    _PRINCIPAL_A_PEER,
                    b"b" * 32,
                    now=now,
                ),
                _credential(_CREDENTIAL_B, _PRINCIPAL_B, b"c" * 32, now=now),
                ModelRegistryRecord(
                    model_id=_MODEL_ID,
                    family="unet",
                    variant="general",
                    quality_tier="balanced",
                    version="1",
                    adapter="tests.fake:FileAccessAdapter",
                    status="ready",
                ),
            ]
        )
        session.flush()
        session.add_all(
            [
                _job("job_a", _TENANT_A, _PRINCIPAL_A),
                _job("job_a2", _TENANT_A, _PRINCIPAL_A),
                _job("job_b", _TENANT_B, _PRINCIPAL_B),
                _job("job_legacy", LEGACY_TENANT_ID, LEGACY_PRINCIPAL_ID),
                _job("job_legacy_peer", LEGACY_TENANT_ID, _LEGACY_PEER),
            ]
        )
        session.flush()
        session.add_all(
            [
                _image("img_a", "job_a", "a"),
                _image("img_a_other", "job_a", "b"),
                _image("img_a2", "job_a2", "c"),
                _image("img_b", "job_b", "d"),
                _image("img_legacy", "job_legacy", "e"),
            ]
        )
        session.flush()
        session.add_all(
            [
                _run("run_a", "job_a", "img_a"),
                _run("run_a_alt", "job_a", "img_a"),
                _run("run_a_other", "job_a", "img_a_other"),
                _run("run_a2", "job_a2", "img_a2"),
                _run("run_b", "job_b", "img_b"),
                _run("run_legacy", "job_legacy", "img_legacy"),
            ]
        )
        session.commit()


def _principal_record(
    principal_id: str,
    tenant_id: str,
    *,
    handle: str,
    role: PrincipalRole,
    now: datetime,
) -> Principal:
    return Principal(
        principal_id=principal_id,
        tenant_id=tenant_id,
        handle=handle,
        display_name=handle,
        kind=PrincipalKind.USER.value,
        role=role.value,
        enabled=True,
        version=1,
        created_at=now,
        updated_at=now,
    )


def _credential(
    credential_id: str,
    principal_id: str,
    digest: bytes,
    *,
    now: datetime,
) -> ApiCredential:
    return ApiCredential(
        credential_id=credential_id,
        principal_id=principal_id,
        label=credential_id,
        token_digest=digest,
        enabled=True,
        version=1,
        created_at=now,
        updated_at=now,
    )


def _job(job_id: str, tenant_id: str, owner_principal_id: str) -> AnalysisJob:
    return AnalysisJob(
        job_id=job_id,
        tenant_id=tenant_id,
        owner_principal_id=owner_principal_id,
        name=job_id,
        status="CREATED",
        config_json={},
    )


def _image(image_id: str, job_id: str, digest_character: str) -> ImageAsset:
    return ImageAsset(
        image_id=image_id,
        job_id=job_id,
        filename=f"{image_id}.png",
        storage_path=f"{job_id}/input/{image_id}/original.png",
        sha256=digest_character * 64,
        width=16,
        height=16,
        bit_depth=8,
        sample_id=image_id,
        experiment_conditions_json={},
        analysis_roi_json={
            "valid_rect": {"x1": 0, "y1": 0, "x2": 16, "y2": 16},
        },
        box_revision=0,
    )


def _run(run_id: str, job_id: str, image_id: str) -> SegmentationRun:
    analysis_roi = {
        "valid_rect": {"x1": 0, "y1": 0, "x2": 16, "y2": 16},
    }
    return SegmentationRun(
        run_id=run_id,
        job_id=job_id,
        image_id=image_id,
        model_id=_MODEL_ID,
        roi_mode="full_image",
        status="CREATED",
        inference_json={},
        run_config_json={
            "model_id": _MODEL_ID,
            "model_version": "1",
            "roi_mode": "full_image",
            "analysis_roi": analysis_roi,
            "inference": {},
            "preprocess_profile": "fixture",
            "postprocess_profile": "fixture",
            "created_at": "2026-07-18T00:00:00Z",
        },
        paths_json={},
    )


def _write_file(harness: _Harness, relative_path: str, content: bytes) -> Path:
    path = harness.file_store.paths.root.joinpath(*relative_path.split("/"))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return path


def _issue_original(harness: _Harness) -> tuple[str, Path, bytes]:
    relative_path = "job_a/input/img_a/original.png"
    content = b"original-inode-content"
    path = _write_file(harness, relative_path, content)
    token = harness.service.issue_download_token(
        principal=harness.principal_a,
        job_id="job_a",
        image_id="img_a",
        artifact_kind=FileArtifactKind.ORIGINAL_IMAGE,
        storage_path=relative_path,
        filename="original.png",
        media_type="image/png",
    )
    return token, path, content


def _issue_corrected_mask(
    harness: _Harness,
    *,
    suffix: str = "stage",
) -> tuple[str, Path, bytes]:
    relative_path = f"job_a/input/review_mask_{suffix}/original.png"
    content = b"corrected-mask"
    path = _write_file(harness, relative_path, content)
    token = harness.service.issue_corrected_mask_token(
        principal=harness.principal_a,
        job_id="job_a",
        image_id="img_a",
        run_id="run_a",
        storage_path=relative_path,
        filename="original.png",
        media_type="image/png",
        expected_sha256=hashlib.sha256(content).hexdigest(),
        expected_size_bytes=len(content),
    )
    return token, path, content


def _token_payload(token: str) -> dict[str, object]:
    encoded = token.split(".")[2]
    payload = base64.urlsafe_b64decode(encoded + "=" * (-len(encoded) % 4))
    decoded = json.loads(payload)
    assert isinstance(decoded, dict)
    return decoded


def _artifact_count(harness: _Harness) -> int:
    with harness.database.session_factory() as session:
        count = session.scalar(select(func.count()).select_from(FileArtifact))
    assert count is not None
    return count
