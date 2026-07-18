"""Application boundary for registered, subject-bound managed files.

Every new URL is backed by immutable database metadata and a v2 token.  The
service deliberately performs authorization before filesystem work, and then
rechecks the database after pinning the exact file descriptor.  Compatibility
v1 tokens never cross this boundary in principal authentication mode.
"""

from __future__ import annotations

import mimetypes
import threading
from dataclasses import dataclass
from pathlib import PurePosixPath

from app.analysis.authorization import require_mutation, require_read
from app.contracts.file_artifacts import (
    FileArtifactDTO,
    FileArtifactKind,
    FileArtifactRegistration,
    FileArtifactState,
)
from app.contracts.identity import (
    LEGACY_PRINCIPAL_ID,
    LEGACY_TENANT_ID,
    AuthMode,
    PrincipalContext,
)
from app.contracts.repositories import RepositorySet, UnitOfWorkFactory
from app.core.errors import ForbiddenError, ResourceNotFoundError, StorageError
from app.storage import (
    FileTokenError,
    FileTokenV2Audience,
    FileTokenV2Error,
    FileTokenV2KeyRing,
    FileTokenV2Purpose,
    LocalFileStore,
    PinnedManagedFile,
    StoragePathError,
    open_pinned_managed_file,
)

_DOWNLOAD_PURPOSES = {
    FileArtifactKind.ORIGINAL_IMAGE: FileTokenV2Purpose.DOWNLOAD_ORIGINAL_IMAGE,
    FileArtifactKind.RUN_ARTIFACT: FileTokenV2Purpose.DOWNLOAD_RUN_ARTIFACT,
    FileArtifactKind.ANALYSIS_EXPORT: FileTokenV2Purpose.DOWNLOAD_ANALYSIS_EXPORT,
}
_DEFAULT_DOWNLOAD_TTL_SECONDS = 900
_DEFAULT_REVIEW_TTL_SECONDS = 900


class FileArtifactUnavailableError(StorageError):
    """Raised when an authorized artifact cannot safely be registered."""


class FileAccessTokenError(ValueError):
    """One non-oracular error for invalid, stale, or context-mismatched tokens."""


@dataclass(frozen=True, slots=True)
class ResolvedFileDownload:
    """A verified descriptor plus response-safe immutable metadata."""

    pinned_file: PinnedManagedFile
    filename: str
    media_type: str
    artifact_id: str | None


@dataclass(frozen=True, slots=True)
class ResolvedCorrectedMask:
    """Verified corrected-mask bytes and the one-shot registry grant to consume."""

    content: bytes
    artifact_id: str
    relative_path: str
    filename: str
    sha256: str
    legacy_v1: bool


class FileArtifactAccessService:
    """Register artifacts, issue v2 capabilities, and resolve them fail closed."""

    def __init__(
        self,
        *,
        uow_factory: UnitOfWorkFactory,
        file_store: LocalFileStore,
        keyring: FileTokenV2KeyRing,
        download_ttl_seconds: int = _DEFAULT_DOWNLOAD_TTL_SECONDS,
        review_ttl_seconds: int = _DEFAULT_REVIEW_TTL_SECONDS,
        max_corrected_mask_bytes: int | None = None,
    ) -> None:
        if not isinstance(file_store, LocalFileStore):
            raise TypeError("file_store must be LocalFileStore")
        if not isinstance(keyring, FileTokenV2KeyRing):
            raise TypeError("keyring must be FileTokenV2KeyRing")
        self.uow_factory = uow_factory
        self.file_store = file_store
        self.keyring = keyring
        self.download_ttl_seconds = _positive_int(
            download_ttl_seconds,
            field="download_ttl_seconds",
        )
        self.review_ttl_seconds = _positive_int(
            review_ttl_seconds,
            field="review_ttl_seconds",
        )
        limit = (
            file_store.max_upload_bytes
            if max_corrected_mask_bytes is None
            else (max_corrected_mask_bytes)
        )
        self.max_corrected_mask_bytes = _positive_int(
            limit,
            field="max_corrected_mask_bytes",
        )
        # The supported deployment is one API process. Serializing the short
        # idempotent registration transaction closes same-path races inside it;
        # a future multi-process topology must add a database-native retry loop.
        self._registration_lock = threading.Lock()

    def issue_download_token(
        self,
        *,
        principal: PrincipalContext,
        job_id: str,
        artifact_kind: FileArtifactKind,
        storage_path: str,
        image_id: str | None = None,
        run_id: str | None = None,
        filename: str | None = None,
        media_type: str | None = None,
        expected_sha256: str | None = None,
        expected_size_bytes: int | None = None,
    ) -> str:
        """Register one authorized immutable file and issue a short-lived v2 token."""

        try:
            purpose = _DOWNLOAD_PURPOSES[artifact_kind]
        except (KeyError, TypeError):
            raise ValueError("artifact_kind is not downloadable") from None
        artifact = self._register_verified_file(
            principal=principal,
            job_id=job_id,
            artifact_kind=artifact_kind,
            storage_path=storage_path,
            image_id=image_id,
            run_id=run_id,
            filename=filename,
            media_type=media_type,
            expected_sha256=expected_sha256,
            expected_size_bytes=expected_size_bytes,
            mutation=False,
        )
        return self._issue(artifact, principal=principal, purpose=purpose)

    def issue_corrected_mask_token(
        self,
        *,
        principal: PrincipalContext,
        job_id: str,
        image_id: str,
        run_id: str,
        storage_path: str,
        filename: str,
        media_type: str,
        expected_sha256: str,
        expected_size_bytes: int,
    ) -> str:
        """Register a validated staged mask and issue a one-purpose review token."""

        artifact = self._register_verified_file(
            principal=principal,
            job_id=job_id,
            artifact_kind=FileArtifactKind.CORRECTED_MASK_INPUT,
            storage_path=storage_path,
            image_id=image_id,
            run_id=run_id,
            filename=filename,
            media_type=media_type,
            expected_sha256=expected_sha256,
            expected_size_bytes=expected_size_bytes,
            mutation=True,
        )
        return self._issue(
            artifact,
            principal=principal,
            purpose=FileTokenV2Purpose.REVIEW_CORRECTED_MASK,
        )

    def resolve_download(
        self,
        token: str,
        *,
        principal: PrincipalContext,
    ) -> ResolvedFileDownload:
        """Resolve a v2 download, or a tightly scoped legacy v1 compatibility token."""

        if not isinstance(principal, PrincipalContext):
            raise TypeError("principal must be PrincipalContext")
        if isinstance(token, str) and token.startswith("v1."):
            if principal.auth_mode is AuthMode.PRINCIPAL:
                raise FileAccessTokenError("invalid file token")
            return self._resolve_legacy_download(token, principal=principal)
        try:
            tenant_id, principal_id = _identity(principal)
            claims = self.keyring.verify(
                token,
                expected_tenant_id=tenant_id,
                expected_principal_id=principal_id,
                expected_audience=FileTokenV2Audience.FILE_DOWNLOAD,
            )
            artifact = self._get_active_authorized(
                claims.aid,
                principal=principal,
                mutation=False,
            )
            purpose = _DOWNLOAD_PURPOSES.get(artifact.artifact_kind)
            if (
                purpose is None
                or claims.pur is not purpose
                or claims.jid != artifact.job_id
                or claims.sha256 != artifact.sha256
            ):
                raise FileAccessTokenError("invalid file token")
            pinned = self._pin_registered(artifact)
            try:
                self._recheck_active(artifact, principal=principal, mutation=False)
            except BaseException:
                pinned.close()
                raise
            return ResolvedFileDownload(
                pinned_file=pinned,
                filename=artifact.filename,
                media_type=artifact.media_type,
                artifact_id=artifact.artifact_id,
            )
        except FileAccessTokenError:
            raise
        except (
            FileTokenV2Error,
            ForbiddenError,
            ResourceNotFoundError,
            StorageError,
            StoragePathError,
            FileNotFoundError,
            OSError,
            TypeError,
            ValueError,
        ):
            raise FileAccessTokenError("invalid file token") from None

    def resolve_corrected_mask(
        self,
        token: str,
        *,
        principal: PrincipalContext,
        job_id: str,
        image_id: str,
        run_id: str,
    ) -> ResolvedCorrectedMask:
        """Resolve an exact parent-bound review grant without consuming it yet."""

        if not isinstance(principal, PrincipalContext):
            raise TypeError("principal must be PrincipalContext")
        # Parent authorization intentionally precedes token parsing and filesystem work.
        self._authorize_target(
            principal=principal,
            job_id=job_id,
            image_id=image_id,
            run_id=run_id,
            mutation=True,
        )
        try:
            if isinstance(token, str) and token.startswith("v1."):
                if principal.auth_mode is AuthMode.PRINCIPAL:
                    raise FileAccessTokenError("invalid corrected-mask token")
                return self._resolve_legacy_corrected_mask(
                    token,
                    principal=principal,
                    job_id=job_id,
                    image_id=image_id,
                    run_id=run_id,
                )
            tenant_id, principal_id = _identity(principal)
            claims = self.keyring.verify(
                token,
                expected_tenant_id=tenant_id,
                expected_principal_id=principal_id,
                expected_audience=FileTokenV2Audience.REVIEW_CORRECTED_MASK,
                expected_purpose=FileTokenV2Purpose.REVIEW_CORRECTED_MASK,
            )
            artifact = self._get_active_authorized(
                claims.aid,
                principal=principal,
                mutation=True,
            )
            if (
                artifact.artifact_kind is not FileArtifactKind.CORRECTED_MASK_INPUT
                or claims.jid != job_id
                or artifact.job_id != job_id
                or artifact.image_id != image_id
                or artifact.run_id != run_id
                or claims.sha256 != artifact.sha256
            ):
                raise FileAccessTokenError("invalid corrected-mask token")
            return self._read_registered_corrected_mask(
                artifact,
                principal=principal,
                legacy_v1=False,
            )
        except FileAccessTokenError:
            raise
        except (
            FileTokenError,
            FileTokenV2Error,
            ForbiddenError,
            ResourceNotFoundError,
            StorageError,
            StoragePathError,
            FileNotFoundError,
            OSError,
            TypeError,
            ValueError,
        ):
            raise FileAccessTokenError("invalid corrected-mask token") from None

    def _register_verified_file(
        self,
        *,
        principal: PrincipalContext,
        job_id: str,
        artifact_kind: FileArtifactKind,
        storage_path: str,
        image_id: str | None,
        run_id: str | None,
        filename: str | None,
        media_type: str | None,
        expected_sha256: str | None,
        expected_size_bytes: int | None,
        mutation: bool,
    ) -> FileArtifactDTO:
        self._authorize_target(
            principal=principal,
            job_id=job_id,
            image_id=image_id,
            run_id=run_id,
            mutation=mutation,
        )
        try:
            with open_pinned_managed_file(
                self.file_store.paths.root,
                storage_path,
                expected_sha256=expected_sha256,
                expected_size_bytes=expected_size_bytes,
            ) as pinned:
                if (
                    artifact_kind is FileArtifactKind.CORRECTED_MASK_INPUT
                    and pinned.size_bytes > self.max_corrected_mask_bytes
                ):
                    raise FileArtifactUnavailableError(
                        "人工修正掩膜超过安全大小限制",
                        details={"job_id": job_id, "artifact_kind": artifact_kind.value},
                    )
                registration = FileArtifactRegistration(
                    job_id=job_id,
                    image_id=image_id,
                    run_id=run_id,
                    artifact_kind=artifact_kind,
                    storage_path=pinned.relative_path,
                    filename=filename or pinned.filename,
                    media_type=_media_type(media_type, filename or pinned.filename),
                    sha256=pinned.sha256,
                    size_bytes=pinned.size_bytes,
                )
        except FileArtifactUnavailableError:
            raise
        except (OSError, StorageError, StoragePathError, ValueError) as error:
            raise FileArtifactUnavailableError(
                "文件制品无法安全登记",
                details={"job_id": job_id, "artifact_kind": artifact_kind.value},
            ) from error

        tenant_id, _principal_id = _identity(principal)
        try:
            with self._registration_lock, self.uow_factory() as uow:
                self._authorize_target_in_repositories(
                    uow.repositories,
                    principal=principal,
                    job_id=job_id,
                    image_id=image_id,
                    run_id=run_id,
                    mutation=mutation,
                )
                artifact = uow.repositories.file_artifacts.register(
                    registration,
                    tenant_id=tenant_id,
                )
                uow.commit()
        except (ResourceNotFoundError, ValueError) as error:
            raise FileArtifactUnavailableError(
                "文件制品无法安全登记",
                details={"job_id": job_id, "artifact_kind": artifact_kind.value},
            ) from error
        return artifact

    def _issue(
        self,
        artifact: FileArtifactDTO,
        *,
        principal: PrincipalContext,
        purpose: FileTokenV2Purpose,
    ) -> str:
        if artifact.state is not FileArtifactState.ACTIVE:
            raise FileArtifactUnavailableError(
                "文件制品已失效",
                details={
                    "job_id": artifact.job_id,
                    "artifact_kind": artifact.artifact_kind.value,
                },
            )
        tenant_id, principal_id = _identity(principal)
        ttl = (
            self.review_ttl_seconds
            if purpose is FileTokenV2Purpose.REVIEW_CORRECTED_MASK
            else self.download_ttl_seconds
        )
        claims = self.keyring.create_claims(
            tenant_id=tenant_id,
            principal_id=principal_id,
            job_id=artifact.job_id,
            artifact_id=artifact.artifact_id,
            purpose=purpose,
            sha256=artifact.sha256,
            ttl_seconds=ttl,
        )
        return self.keyring.issue(claims)

    def _get_active_authorized(
        self,
        artifact_id: str,
        *,
        principal: PrincipalContext,
        mutation: bool,
    ) -> FileArtifactDTO:
        tenant_id, _principal_id = _identity(principal)
        with self.uow_factory() as uow:
            artifact = uow.repositories.file_artifacts.get_active(
                artifact_id,
                tenant_id=tenant_id,
            )
            self._authorize_target_in_repositories(
                uow.repositories,
                principal=principal,
                job_id=artifact.job_id,
                image_id=artifact.image_id,
                run_id=artifact.run_id,
                mutation=mutation,
            )
            return artifact

    def _recheck_active(
        self,
        expected: FileArtifactDTO,
        *,
        principal: PrincipalContext,
        mutation: bool,
    ) -> None:
        current = self._get_active_authorized(
            expected.artifact_id,
            principal=principal,
            mutation=mutation,
        )
        if current != expected:
            raise FileAccessTokenError("invalid file token")

    def _pin_registered(self, artifact: FileArtifactDTO) -> PinnedManagedFile:
        return open_pinned_managed_file(
            self.file_store.paths.root,
            artifact.storage_path,
            expected_size_bytes=artifact.size_bytes,
            expected_sha256=artifact.sha256,
        )

    def _read_registered_corrected_mask(
        self,
        artifact: FileArtifactDTO,
        *,
        principal: PrincipalContext,
        legacy_v1: bool,
    ) -> ResolvedCorrectedMask:
        if artifact.size_bytes > self.max_corrected_mask_bytes:
            raise FileAccessTokenError("invalid corrected-mask token")
        with self._pin_registered(artifact) as pinned:
            content = b"".join(pinned.iter_chunks())
        self._recheck_active(artifact, principal=principal, mutation=True)
        return ResolvedCorrectedMask(
            content=content,
            artifact_id=artifact.artifact_id,
            relative_path=artifact.storage_path,
            filename=artifact.filename,
            sha256=artifact.sha256,
            legacy_v1=legacy_v1,
        )

    def _resolve_legacy_download(
        self,
        token: str,
        *,
        principal: PrincipalContext,
    ) -> ResolvedFileDownload:
        try:
            relative_path = self.file_store.decode_file_token_path(token)
            job_id = PurePosixPath(relative_path).parts[0]
            self._authorize_legacy_job(job_id, principal=principal, mutation=False)
            pinned = open_pinned_managed_file(
                self.file_store.paths.root,
                relative_path,
            )
            try:
                self._authorize_legacy_job(job_id, principal=principal, mutation=False)
            except BaseException:
                pinned.close()
                raise
            return ResolvedFileDownload(
                pinned_file=pinned,
                filename=pinned.filename,
                media_type=_media_type(None, pinned.filename),
                artifact_id=None,
            )
        except (
            FileTokenError,
            ForbiddenError,
            ResourceNotFoundError,
            StorageError,
            StoragePathError,
            FileNotFoundError,
            OSError,
            IndexError,
            TypeError,
            ValueError,
        ):
            raise FileAccessTokenError("invalid file token") from None

    def _resolve_legacy_corrected_mask(
        self,
        token: str,
        *,
        principal: PrincipalContext,
        job_id: str,
        image_id: str,
        run_id: str,
    ) -> ResolvedCorrectedMask:
        relative_path = self.file_store.decode_file_token_path(token)
        parts = PurePosixPath(relative_path).parts
        if (
            len(parts) != 4
            or parts[0] != job_id
            or parts[1] != "input"
            or not parts[2].startswith("review_mask_")
            or not (parts[3] == "original" or parts[3].startswith("original."))
        ):
            raise FileAccessTokenError("invalid corrected-mask token")
        self._authorize_legacy_job(job_id, principal=principal, mutation=True)
        try:
            with open_pinned_managed_file(
                self.file_store.paths.root,
                relative_path,
            ) as pinned:
                if pinned.size_bytes > self.max_corrected_mask_bytes:
                    raise FileAccessTokenError("invalid corrected-mask token")
                content = b"".join(pinned.iter_chunks())
                registration = FileArtifactRegistration(
                    job_id=job_id,
                    image_id=image_id,
                    run_id=run_id,
                    artifact_kind=FileArtifactKind.CORRECTED_MASK_INPUT,
                    storage_path=relative_path,
                    filename=pinned.filename,
                    media_type=_media_type(None, pinned.filename),
                    sha256=pinned.sha256,
                    size_bytes=pinned.size_bytes,
                )
        except FileAccessTokenError:
            raise
        except (OSError, StorageError, StoragePathError, ValueError):
            raise FileAccessTokenError("invalid corrected-mask token") from None
        tenant_id, _principal_id = _identity(principal)
        with self._registration_lock, self.uow_factory() as uow:
            self._authorize_target_in_repositories(
                uow.repositories,
                principal=principal,
                job_id=job_id,
                image_id=image_id,
                run_id=run_id,
                mutation=True,
            )
            artifact = uow.repositories.file_artifacts.register(
                registration,
                tenant_id=tenant_id,
            )
            uow.commit()
        self._recheck_active(artifact, principal=principal, mutation=True)
        return ResolvedCorrectedMask(
            content=content,
            artifact_id=artifact.artifact_id,
            relative_path=artifact.storage_path,
            filename=artifact.filename,
            sha256=artifact.sha256,
            legacy_v1=True,
        )

    def _authorize_target(
        self,
        *,
        principal: PrincipalContext,
        job_id: str,
        image_id: str | None,
        run_id: str | None,
        mutation: bool,
    ) -> None:
        with self.uow_factory() as uow:
            self._authorize_target_in_repositories(
                uow.repositories,
                principal=principal,
                job_id=job_id,
                image_id=image_id,
                run_id=run_id,
                mutation=mutation,
            )

    @staticmethod
    def _authorize_target_in_repositories(
        repositories: RepositorySet,
        *,
        principal: PrincipalContext,
        job_id: str,
        image_id: str | None,
        run_id: str | None,
        mutation: bool,
    ) -> None:
        tenant_id, _principal_id = _identity(principal)
        scope = repositories.jobs.get_scope(job_id, tenant_id=tenant_id)
        if mutation:
            require_mutation(principal, scope)
        else:
            require_read(principal, scope)
        if image_id is not None:
            repositories.images.get_scoped(
                job_id,
                image_id,
                tenant_id=tenant_id,
            )
        if run_id is not None:
            run, run_scope = repositories.runs.get_with_scope(
                run_id,
                tenant_id=tenant_id,
            )
            if run.job_id != job_id or (image_id is not None and run.image_id != image_id):
                raise ResourceNotFoundError(details={"resource": "file"})
            if mutation:
                require_mutation(principal, run_scope)
            else:
                require_read(principal, run_scope)

    def _authorize_legacy_job(
        self,
        job_id: str,
        *,
        principal: PrincipalContext,
        mutation: bool,
    ) -> None:
        if (
            principal.auth_mode is AuthMode.PRINCIPAL
            or principal.tenant_id != LEGACY_TENANT_ID
            or principal.principal_id != LEGACY_PRINCIPAL_ID
        ):
            raise FileAccessTokenError("invalid file token")
        with self.uow_factory() as uow:
            scope = uow.repositories.jobs.get_scope(job_id, tenant_id=LEGACY_TENANT_ID)
            if scope.owner_principal_id != LEGACY_PRINCIPAL_ID:
                raise FileAccessTokenError("invalid file token")
            if mutation:
                require_mutation(principal, scope)
            else:
                require_read(principal, scope)


def _identity(principal: PrincipalContext) -> tuple[str, str]:
    if not isinstance(principal, PrincipalContext):
        raise TypeError("principal must be PrincipalContext")
    if principal.tenant_id is None or principal.principal_id is None:
        raise ValueError("principal must carry tenant and principal IDs")
    return principal.tenant_id, principal.principal_id


def _positive_int(value: object, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{field} must be a positive integer")
    return value


def _media_type(value: str | None, filename: str) -> str:
    candidate = value or mimetypes.guess_type(filename)[0] or "application/octet-stream"
    normalized = candidate.casefold()
    if (
        "/" not in normalized
        or normalized != normalized.strip()
        or any(ord(character) < 32 or 127 <= ord(character) <= 159 for character in normalized)
    ):
        return "application/octet-stream"
    return normalized
