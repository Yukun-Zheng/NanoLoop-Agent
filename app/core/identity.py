"""Secret-safe identity and bearer-credential primitives.

This module deliberately has no database or HTTP dependency. Persistence can store the returned
credential ID and digest while the raw token is shown exactly once to an operator.
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import hmac
import re
import secrets
from dataclasses import dataclass, field

from pydantic import SecretStr

from app.contracts.identity import (
    CREDENTIAL_ID_PREFIX,
    LEGACY_PRINCIPAL_ID,
    LEGACY_TENANT_ID,
    AuthMode,
    PrincipalContext,
    PrincipalKind,
    PrincipalRole,
    validate_credential_id,
    validate_principal_id,
    validate_tenant_id,
)

CREDENTIAL_TOKEN_VERSION = "v1"
CREDENTIAL_TOKEN_PREFIX = f"nlk_{CREDENTIAL_TOKEN_VERSION}"
CREDENTIAL_SECRET_BYTES = 32
CREDENTIAL_SECRET_LENGTH = 43
CREDENTIAL_DIGEST_BYTES = hashlib.sha256().digest_size
MINIMUM_PEPPER_BYTES = 32

_TOKEN_PATTERN = re.compile(
    rf"\A{CREDENTIAL_TOKEN_PREFIX}_"
    rf"(?P<credential_id>{CREDENTIAL_ID_PREFIX}_[0-9a-f]{{32}})_"
    rf"(?P<secret>[A-Za-z0-9_-]{{{CREDENTIAL_SECRET_LENGTH}}})\Z"
)
_DUMMY_TOKEN = (
    f"{CREDENTIAL_TOKEN_PREFIX}_{CREDENTIAL_ID_PREFIX}_{'0' * 32}_{'A' * CREDENTIAL_SECRET_LENGTH}"
)
_DUMMY_DIGEST = bytes(CREDENTIAL_DIGEST_BYTES)


@dataclass(frozen=True, slots=True)
class ParsedCredentialToken:
    """Strictly parsed token parts with a redacted secret representation."""

    credential_id: str
    secret: SecretStr


@dataclass(frozen=True, slots=True)
class IssuedCredential:
    """One-time credential material returned by operator provisioning."""

    credential_id: str
    token: SecretStr
    digest: bytes = field(repr=False)


class CredentialHasher:
    """Create and verify peppered HMAC-SHA256 credential digests."""

    __slots__ = ("_pepper",)

    def __init__(self, pepper: SecretStr | str | bytes) -> None:
        pepper_bytes = _secret_bytes(pepper)
        if len(pepper_bytes) < MINIMUM_PEPPER_BYTES:
            raise ValueError("credential pepper must contain at least 32 bytes")
        self._pepper = pepper_bytes

    def __repr__(self) -> str:
        return f"{type(self).__name__}(pepper=<redacted>)"

    def digest(self, token: SecretStr | str) -> bytes:
        """Return the persisted digest for a strictly valid credential token."""

        raw_token = _secret_value(token)
        parse_credential_token(raw_token)
        return self._digest_raw(raw_token)

    def verify(self, token: SecretStr | str, expected_digest: bytes | None) -> bool:
        """Verify without skipping the HMAC/constant-time comparison path on invalid input."""

        raw_token = _secret_value(token)
        valid_token = _match_token(raw_token) is not None
        if isinstance(expected_digest, bytes) and len(expected_digest) == CREDENTIAL_DIGEST_BYTES:
            expected = expected_digest
            digest_is_valid = True
        else:
            expected = _DUMMY_DIGEST
            digest_is_valid = False
        candidate = self._digest_raw(raw_token if valid_token else _DUMMY_TOKEN)
        matches = hmac.compare_digest(candidate, expected)
        return valid_token and digest_is_valid and matches

    def _digest_raw(self, token: str) -> bytes:
        return hmac.new(self._pepper, token.encode("ascii"), hashlib.sha256).digest()


def generate_tenant_id() -> str:
    """Generate a cryptographically random canonical tenant ID."""

    return validate_tenant_id(f"tnt_{secrets.token_hex(16)}")


def generate_principal_id() -> str:
    """Generate a cryptographically random canonical principal ID."""

    return validate_principal_id(f"prn_{secrets.token_hex(16)}")


def generate_credential_id() -> str:
    """Generate a cryptographically random canonical credential ID."""

    return validate_credential_id(f"{CREDENTIAL_ID_PREFIX}_{secrets.token_hex(16)}")


def parse_credential_token(token: SecretStr | str) -> ParsedCredentialToken:
    """Strictly parse a credential token without reflecting malformed input in errors."""

    raw_token = _secret_value(token)
    match = _match_token(raw_token)
    if match is None:
        raise ValueError("invalid NanoLoop credential token")
    return ParsedCredentialToken(
        credential_id=validate_credential_id(match.group("credential_id")),
        secret=SecretStr(match.group("secret")),
    )


def issue_credential(pepper: SecretStr | str | bytes) -> IssuedCredential:
    """Generate a token and its storage digest; the raw token must be displayed only once."""

    credential_id = generate_credential_id()
    secret = secrets.token_urlsafe(CREDENTIAL_SECRET_BYTES)
    if len(secret) != CREDENTIAL_SECRET_LENGTH:
        raise RuntimeError("credential secret generator returned an unexpected length")
    token = f"{CREDENTIAL_TOKEN_PREFIX}_{credential_id}_{secret}"
    return IssuedCredential(
        credential_id=credential_id,
        token=SecretStr(token),
        digest=CredentialHasher(pepper).digest(token),
    )


def legacy_principal_context(auth_mode: AuthMode) -> PrincipalContext:
    """Return the fixed service principal used only by shared-key/disabled compatibility."""

    if auth_mode is AuthMode.PRINCIPAL:
        raise ValueError("principal mode requires a persisted principal and credential")
    return PrincipalContext(
        tenant_id=LEGACY_TENANT_ID,
        principal_id=LEGACY_PRINCIPAL_ID,
        credential_id=None,
        kind=PrincipalKind.SERVICE,
        role=PrincipalRole.TENANT_ADMIN,
        auth_mode=auth_mode,
    )


def _secret_value(value: SecretStr | str) -> str:
    return value.get_secret_value() if isinstance(value, SecretStr) else value


def _secret_bytes(value: SecretStr | str | bytes) -> bytes:
    if isinstance(value, bytes):
        return bytes(value)
    return _secret_value(value).encode("utf-8")


def _match_token(token: str) -> re.Match[str] | None:
    match = _TOKEN_PATTERN.fullmatch(token)
    if match is None:
        return None
    encoded_secret = match.group("secret")
    try:
        decoded_secret = base64.b64decode(f"{encoded_secret}=", altchars=b"-_", validate=True)
    except (binascii.Error, ValueError):
        return None
    canonical_secret = base64.urlsafe_b64encode(decoded_secret).rstrip(b"=").decode("ascii")
    if len(decoded_secret) != CREDENTIAL_SECRET_BYTES or canonical_secret != encoded_secret:
        return None
    return match
