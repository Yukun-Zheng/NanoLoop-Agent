"""Canonical tenant- and principal-bound file-token v2 primitives.

This module intentionally has no HTTP, database, or filesystem dependency.  The artifact
registry owns paths; a v2 token only names the immutable artifact and the caller/purpose that
may use it.
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import hmac
import json
import re
import secrets
import time
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from pydantic import SecretStr

from app.contracts.identity import validate_principal_id, validate_tenant_id

FILE_TOKEN_V2_PREFIX = "v2"
FILE_TOKEN_V2_VERSION = 2
FILE_TOKEN_V2_DEFAULT_MAX_TTL_SECONDS = 86_400
FILE_TOKEN_V2_DEFAULT_CLOCK_SKEW_SECONDS = 30
FILE_TOKEN_V2_MAX_CLOCK_SKEW_SECONDS = 300
FILE_TOKEN_V2_MAX_LENGTH = 4096
FILE_TOKEN_V2_MAX_UNIX_TIMESTAMP = 253_402_300_799
FILE_TOKEN_V2_JTI_BYTES = 16
FILE_TOKEN_V2_MINIMUM_KEY_BYTES = 32

_FILE_TOKEN_V2_PAYLOAD_KEYS = frozenset(
    {"v", "tid", "sub", "jid", "aid", "pur", "aud", "sha256", "iat", "nbf", "exp", "jti"}
)
_KID_PATTERN = re.compile(r"\A[a-z0-9](?:[a-z0-9_-]{0,30}[a-z0-9])?\Z")
_JOB_ID_PATTERN = re.compile(r"\A[A-Za-z0-9][A-Za-z0-9._-]{0,63}\Z")
_ARTIFACT_ID_PATTERN = re.compile(r"\Aart_[0-9a-f]{32}\Z")
_SHA256_PATTERN = re.compile(r"\A[0-9a-f]{64}\Z")
_DUMMY_VERIFICATION_KEY = hashlib.sha256(b"NanoLoop file-token v2 unknown signing key").digest()
_DUMMY_SIGNATURE = bytes(hashlib.sha256().digest_size)


class FileTokenV2Error(ValueError):
    """Raised for every malformed, forged, inactive, or context-mismatched v2 token."""


class FileTokenV2Purpose(StrEnum):
    """An exact operation for which a file-token v2 may be used."""

    DOWNLOAD_ORIGINAL_IMAGE = "download.original_image"
    DOWNLOAD_RUN_ARTIFACT = "download.run_artifact"
    DOWNLOAD_ANALYSIS_EXPORT = "download.analysis_export"
    REVIEW_CORRECTED_MASK = "review.corrected_mask"


class FileTokenV2Audience(StrEnum):
    """The endpoint family allowed to consume a file-token v2."""

    FILE_DOWNLOAD = "nanoloop-api:file-download"
    REVIEW_CORRECTED_MASK = "nanoloop-api:review-corrected-mask"


_PURPOSE_AUDIENCE = {
    FileTokenV2Purpose.DOWNLOAD_ORIGINAL_IMAGE: FileTokenV2Audience.FILE_DOWNLOAD,
    FileTokenV2Purpose.DOWNLOAD_RUN_ARTIFACT: FileTokenV2Audience.FILE_DOWNLOAD,
    FileTokenV2Purpose.DOWNLOAD_ANALYSIS_EXPORT: FileTokenV2Audience.FILE_DOWNLOAD,
    FileTokenV2Purpose.REVIEW_CORRECTED_MASK: FileTokenV2Audience.REVIEW_CORRECTED_MASK,
}


@dataclass(frozen=True, slots=True)
class FileTokenV2Claims:
    """The exact, immutable claim set serialized into a file-token v2 payload."""

    v: int
    tid: str
    sub: str
    jid: str
    aid: str
    pur: FileTokenV2Purpose
    aud: FileTokenV2Audience
    sha256: str
    iat: int
    nbf: int
    exp: int
    jti: str

    def __post_init__(self) -> None:
        _validate_version(self.v)
        validate_tenant_id(self.tid)
        validate_principal_id(self.sub)
        validate_file_token_v2_job_id(self.jid)
        validate_file_token_v2_artifact_id(self.aid)
        if not isinstance(self.pur, FileTokenV2Purpose):
            raise ValueError("invalid file-token v2 purpose")
        if not isinstance(self.aud, FileTokenV2Audience):
            raise ValueError("invalid file-token v2 audience")
        if _PURPOSE_AUDIENCE[self.pur] is not self.aud:
            raise ValueError("file-token v2 purpose does not match its audience")
        validate_file_token_v2_sha256(self.sha256)
        issued_at = _unix_timestamp(self.iat, field="iat")
        not_before = _unix_timestamp(self.nbf, field="nbf")
        expires_at = _unix_timestamp(self.exp, field="exp")
        if not issued_at <= not_before <= expires_at:
            raise ValueError("file-token v2 timestamps must satisfy iat <= nbf <= exp")
        validate_file_token_v2_jti(self.jti)

    @property
    def tenant_id(self) -> str:
        return self.tid

    @property
    def principal_id(self) -> str:
        return self.sub

    @property
    def job_id(self) -> str:
        return self.jid

    @property
    def artifact_id(self) -> str:
        return self.aid

    @property
    def purpose(self) -> FileTokenV2Purpose:
        return self.pur

    @property
    def audience(self) -> FileTokenV2Audience:
        return self.aud

    @property
    def issued_at(self) -> int:
        return self.iat

    @property
    def not_before(self) -> int:
        return self.nbf

    @property
    def expires_at(self) -> int:
        return self.exp

    @property
    def token_id(self) -> str:
        return self.jti

    def as_payload(self) -> dict[str, object]:
        """Return the exact JSON payload with no path or credential material."""

        return {
            "v": self.v,
            "tid": self.tid,
            "sub": self.sub,
            "jid": self.jid,
            "aid": self.aid,
            "pur": self.pur.value,
            "aud": self.aud.value,
            "sha256": self.sha256,
            "iat": self.iat,
            "nbf": self.nbf,
            "exp": self.exp,
            "jti": self.jti,
        }


class FileTokenV2KeyRing:
    """Issue with one active key and verify against retained rotation keys."""

    __slots__ = (
        "_active_kid",
        "_clock_skew_seconds",
        "_keys",
        "_max_ttl_seconds",
    )

    def __init__(
        self,
        keys: Mapping[str, SecretStr | str | bytes],
        *,
        active_kid: str,
        max_ttl_seconds: int = FILE_TOKEN_V2_DEFAULT_MAX_TTL_SECONDS,
        clock_skew_seconds: int = FILE_TOKEN_V2_DEFAULT_CLOCK_SKEW_SECONDS,
    ) -> None:
        validated_keys: dict[str, bytes] = {}
        for kid, key in keys.items():
            canonical_kid = validate_file_token_v2_kid(kid)
            key_bytes = _secret_bytes(key)
            if len(key_bytes) < FILE_TOKEN_V2_MINIMUM_KEY_BYTES:
                raise ValueError("file-token v2 signing keys must contain at least 32 bytes")
            validated_keys[canonical_kid] = key_bytes
        canonical_active_kid = validate_file_token_v2_kid(active_kid)
        if canonical_active_kid not in validated_keys:
            raise ValueError("active file-token v2 key ID is not present in the key ring")
        self._max_ttl_seconds = _positive_int(max_ttl_seconds, field="max_ttl_seconds")
        self._clock_skew_seconds = _bounded_nonnegative_int(
            clock_skew_seconds,
            field="clock_skew_seconds",
            maximum=FILE_TOKEN_V2_MAX_CLOCK_SKEW_SECONDS,
        )
        self._keys = validated_keys
        self._active_kid = canonical_active_kid

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(active_kid={self._active_kid!r}, "
            f"retained_kids={tuple(self._keys)!r}, keys=<redacted>)"
        )

    @property
    def active_kid(self) -> str:
        return self._active_kid

    @property
    def retained_kids(self) -> tuple[str, ...]:
        return tuple(self._keys)

    @property
    def max_ttl_seconds(self) -> int:
        return self._max_ttl_seconds

    @property
    def clock_skew_seconds(self) -> int:
        return self._clock_skew_seconds

    def create_claims(
        self,
        *,
        tenant_id: str,
        principal_id: str,
        job_id: str,
        artifact_id: str,
        purpose: FileTokenV2Purpose | str,
        sha256: str,
        ttl_seconds: int,
        now: int | None = None,
        not_before: int | None = None,
    ) -> FileTokenV2Claims:
        """Create canonical claims, deriving the audience and a random 128-bit token ID."""

        issued_at = _observed_time(now)
        ttl = _positive_int(ttl_seconds, field="ttl_seconds")
        if ttl > self._max_ttl_seconds:
            raise ValueError("ttl_seconds must not exceed max_ttl_seconds")
        if issued_at > FILE_TOKEN_V2_MAX_UNIX_TIMESTAMP - ttl:
            raise ValueError("file-token v2 expiration exceeds the supported timestamp range")
        activation_time = (
            issued_at if not_before is None else _unix_timestamp(not_before, field="not_before")
        )
        canonical_purpose = _purpose(purpose)
        return FileTokenV2Claims(
            v=FILE_TOKEN_V2_VERSION,
            tid=tenant_id,
            sub=principal_id,
            jid=job_id,
            aid=artifact_id,
            pur=canonical_purpose,
            aud=_PURPOSE_AUDIENCE[canonical_purpose],
            sha256=sha256,
            iat=issued_at,
            nbf=activation_time,
            exp=issued_at + ttl,
            jti=_base64url_encode(secrets.token_bytes(FILE_TOKEN_V2_JTI_BYTES)),
        )

    def issue(self, claims: FileTokenV2Claims) -> str:
        """Serialize and sign canonical claims with the active key."""

        if not isinstance(claims, FileTokenV2Claims):
            raise TypeError("claims must be FileTokenV2Claims")
        self._validate_ttl(claims)
        payload_bytes = _canonical_payload_bytes(claims.as_payload())
        encoded_payload = _base64url_encode(payload_bytes)
        signed_value = f"{FILE_TOKEN_V2_PREFIX}.{self._active_kid}.{encoded_payload}".encode(
            "ascii"
        )
        signature = hmac.new(self._keys[self._active_kid], signed_value, hashlib.sha256).digest()
        token = f"{signed_value.decode('ascii')}.{_base64url_encode(signature)}"
        if len(token) > FILE_TOKEN_V2_MAX_LENGTH:
            raise ValueError("file-token v2 exceeds the supported length")
        return token

    def verify(
        self,
        token: str,
        *,
        now: int | None = None,
        expected_tenant_id: str | None = None,
        expected_principal_id: str | None = None,
        expected_audience: FileTokenV2Audience | str | None = None,
        expected_purpose: FileTokenV2Purpose | str | None = None,
    ) -> FileTokenV2Claims:
        """Verify the signature, canonical claims, time window, and optional request context."""

        try:
            parts = self._verified_parts(token)
            claims = _claims_from_payload(parts[2])
            self._validate_ttl(claims)
            current_time = _observed_time(now)
            if current_time + self._clock_skew_seconds < claims.nbf:
                raise ValueError("file-token v2 is not active")
            if current_time >= claims.exp + self._clock_skew_seconds:
                raise ValueError("file-token v2 has expired")
            if expected_tenant_id is not None and claims.tid != validate_tenant_id(
                expected_tenant_id
            ):
                raise ValueError("file-token v2 tenant mismatch")
            if expected_principal_id is not None and claims.sub != validate_principal_id(
                expected_principal_id
            ):
                raise ValueError("file-token v2 principal mismatch")
            if expected_audience is not None and claims.aud is not _audience(expected_audience):
                raise ValueError("file-token v2 audience mismatch")
            if expected_purpose is not None and claims.pur is not _purpose(expected_purpose):
                raise ValueError("file-token v2 purpose mismatch")
            return claims
        except (KeyError, TypeError, ValueError, UnicodeError, json.JSONDecodeError):
            raise FileTokenV2Error("invalid file token") from None

    def _verified_parts(self, token: str) -> tuple[str, str, str, str]:
        if not isinstance(token, str) or not token or len(token) > FILE_TOKEN_V2_MAX_LENGTH:
            raise ValueError("invalid file-token v2 envelope")
        parts = token.split(".")
        if len(parts) != 4 or parts[0] != FILE_TOKEN_V2_PREFIX:
            raise ValueError("invalid file-token v2 envelope")
        _, kid, encoded_payload, encoded_signature = parts

        try:
            canonical_kid = validate_file_token_v2_kid(kid)
            kid_is_canonical = True
        except (TypeError, ValueError):
            canonical_kid = ""
            kid_is_canonical = False
        key = self._keys.get(canonical_kid, _DUMMY_VERIFICATION_KEY)
        kid_is_known = kid_is_canonical and canonical_kid in self._keys
        try:
            signed_value = f"{parts[0]}.{kid}.{encoded_payload}".encode("ascii", errors="strict")
        except UnicodeEncodeError:
            signed_value = b"v2.invalid.invalid"
        expected_signature = hmac.new(key, signed_value, hashlib.sha256).digest()
        try:
            supplied_signature = _base64url_decode(encoded_signature)
            signature_is_canonical = len(supplied_signature) == hashlib.sha256().digest_size
        except (TypeError, ValueError, UnicodeError):
            supplied_signature = _DUMMY_SIGNATURE
            signature_is_canonical = False
        signature_matches = hmac.compare_digest(supplied_signature, expected_signature)
        if not kid_is_known or not signature_is_canonical or not signature_matches:
            raise ValueError("invalid file-token v2 signature")
        return parts[0], canonical_kid, encoded_payload, encoded_signature

    def _validate_ttl(self, claims: FileTokenV2Claims) -> None:
        if claims.exp - claims.iat > self._max_ttl_seconds:
            raise ValueError("file-token v2 exceeds the configured maximum TTL")


def validate_file_token_v2_kid(value: str) -> str:
    """Return a canonical short signing-key ID."""

    if not isinstance(value, str) or _KID_PATTERN.fullmatch(value) is None:
        raise ValueError("invalid file-token v2 key ID")
    return value


def validate_file_token_v2_job_id(value: str) -> str:
    """Return a bounded ASCII job ID that is safe to put in structured logs."""

    if not isinstance(value, str) or _JOB_ID_PATTERN.fullmatch(value) is None:
        raise ValueError("invalid file-token v2 job ID")
    return value


def validate_file_token_v2_artifact_id(value: str) -> str:
    """Return a canonical immutable artifact-registry ID."""

    if not isinstance(value, str) or _ARTIFACT_ID_PATTERN.fullmatch(value) is None:
        raise ValueError("invalid file-token v2 artifact ID")
    return value


def validate_file_token_v2_sha256(value: str) -> str:
    """Return a canonical lowercase SHA-256 hex digest."""

    if not isinstance(value, str) or _SHA256_PATTERN.fullmatch(value) is None:
        raise ValueError("invalid file-token v2 SHA-256 digest")
    return value


def validate_file_token_v2_jti(value: str) -> str:
    """Return a canonical unpadded base64url encoding of exactly 128 random bits."""

    if not isinstance(value, str):
        raise ValueError("invalid file-token v2 token ID")
    try:
        decoded = _base64url_decode(value)
    except (TypeError, ValueError, UnicodeError):
        raise ValueError("invalid file-token v2 token ID") from None
    if len(decoded) != FILE_TOKEN_V2_JTI_BYTES:
        raise ValueError("invalid file-token v2 token ID")
    return value


def _claims_from_payload(encoded_payload: str) -> FileTokenV2Claims:
    payload_bytes = _base64url_decode(encoded_payload)
    payload: Any = json.loads(
        payload_bytes.decode("utf-8", errors="strict"),
        object_pairs_hook=_json_object_without_duplicate_keys,
    )
    if not isinstance(payload, dict) or set(payload) != _FILE_TOKEN_V2_PAYLOAD_KEYS:
        raise ValueError("invalid file-token v2 payload shape")
    if payload_bytes != _canonical_payload_bytes(payload):
        raise ValueError("non-canonical file-token v2 payload")
    return FileTokenV2Claims(
        v=payload["v"],
        tid=payload["tid"],
        sub=payload["sub"],
        jid=payload["jid"],
        aid=payload["aid"],
        pur=FileTokenV2Purpose(payload["pur"]),
        aud=FileTokenV2Audience(payload["aud"]),
        sha256=payload["sha256"],
        iat=payload["iat"],
        nbf=payload["nbf"],
        exp=payload["exp"],
        jti=payload["jti"],
    )


def _canonical_payload_bytes(payload: Mapping[str, object]) -> bytes:
    return json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8")


def _json_object_without_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for key, value in pairs:
        if key in payload:
            raise ValueError("duplicate JSON object key")
        payload[key] = value
    return payload


def _validate_version(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value != FILE_TOKEN_V2_VERSION:
        raise ValueError("invalid file-token v2 version")
    return value


def _positive_int(value: object, *, field: str) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value <= 0
        or value > FILE_TOKEN_V2_MAX_UNIX_TIMESTAMP
    ):
        raise ValueError(f"{field} must be a supported positive integer")
    return value


def _bounded_nonnegative_int(value: object, *, field: str, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0 or value > maximum:
        raise ValueError(f"{field} must be an integer between 0 and {maximum}")
    return value


def _unix_timestamp(value: object, *, field: str) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value < 0
        or value > FILE_TOKEN_V2_MAX_UNIX_TIMESTAMP
    ):
        raise ValueError(f"{field} must be a supported integer Unix timestamp")
    return value


def _observed_time(now: int | None) -> int:
    observed = int(time.time()) if now is None else now
    return _unix_timestamp(observed, field="now")


def _purpose(value: FileTokenV2Purpose | str) -> FileTokenV2Purpose:
    try:
        return FileTokenV2Purpose(value)
    except (TypeError, ValueError):
        raise ValueError("invalid file-token v2 purpose") from None


def _audience(value: FileTokenV2Audience | str) -> FileTokenV2Audience:
    try:
        return FileTokenV2Audience(value)
    except (TypeError, ValueError):
        raise ValueError("invalid file-token v2 audience") from None


def _secret_bytes(value: SecretStr | str | bytes) -> bytes:
    if isinstance(value, bytes):
        return bytes(value)
    if isinstance(value, SecretStr):
        return value.get_secret_value().encode("utf-8")
    if isinstance(value, str):
        return value.encode("utf-8")
    raise TypeError("file-token v2 signing key must be text or bytes")


def _base64url_encode(value: bytes) -> str:
    return base64.urlsafe_b64encode(value).rstrip(b"=").decode("ascii")


def _base64url_decode(value: str) -> bytes:
    if not isinstance(value, str) or not value or "=" in value:
        raise ValueError("invalid base64url value")
    try:
        value.encode("ascii", errors="strict")
    except UnicodeEncodeError as error:
        raise ValueError("invalid base64url value") from error
    padding = "=" * (-len(value) % 4)
    try:
        decoded = base64.b64decode(value + padding, altchars=b"-_", validate=True)
    except (binascii.Error, ValueError) as error:
        raise ValueError("invalid base64url value") from error
    if _base64url_encode(decoded) != value:
        raise ValueError("non-canonical base64url value")
    return decoded
