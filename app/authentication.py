"""Authentication-mode assembly for the HTTP boundary.

Raw credentials live only for the duration of one request.  Principal mode never falls back to
the compatibility shared key, even when ``NANOLOOP_API_KEY`` remains present in the environment.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Literal

from starlette.concurrency import run_in_threadpool

from app.contracts.identity import AuthMode, PrincipalContext
from app.core.config import Settings
from app.core.identity import CredentialHasher, legacy_principal_context, parse_credential_token
from app.core.rate_limit import RateLimitBucket
from app.core.security import ApiKeyVerifier
from app.db.identity import AuthenticationResult, IdentityService
from app.db.session import Database

AUTHENTICATION_VERIFIED_STATE_KEY = "_nanoloop_authentication_verified"
AUTH_MODE_STATE_KEY = "_nanoloop_auth_mode"
AUTH_OUTCOME_STATE_KEY = "_nanoloop_auth_outcome"
AUTH_REASON_STATE_KEY = "_nanoloop_auth_reason"

AuthenticationOutcome = Literal["authenticated", "rejected", "unavailable"]


@dataclass(frozen=True, slots=True)
class AuthenticationDecision:
    """Secret-free result consumed by the HTTP middleware."""

    outcome: AuthenticationOutcome
    reason: str
    principal: PrincipalContext | None = None

    @property
    def authenticated(self) -> bool:
        return self.outcome == "authenticated" and self.principal is not None


class RequestAuthenticator:
    """Authenticate one X-API-Key header under one explicitly resolved mode."""

    __slots__ = ("_clock", "_database", "_hasher", "_shared_key_verifier", "mode")

    def __init__(
        self,
        mode: AuthMode,
        *,
        database: Database | None = None,
        shared_key_verifier: ApiKeyVerifier | None = None,
        hasher: CredentialHasher | None = None,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self.mode = AuthMode(mode)
        self._database = database
        self._shared_key_verifier = shared_key_verifier or ApiKeyVerifier(None)
        self._hasher = hasher
        self._clock = clock or (lambda: datetime.now(UTC))
        if self.mode is AuthMode.SHARED_KEY and not self._shared_key_verifier.enabled:
            raise ValueError("shared-key authentication requires a configured verifier")
        if self.mode is AuthMode.PRINCIPAL and (database is None or hasher is None):
            raise ValueError("principal authentication requires a database and credential hasher")

    @classmethod
    def from_settings(
        cls,
        settings: Settings,
        database: Database,
        *,
        clock: Callable[[], datetime] | None = None,
    ) -> RequestAuthenticator:
        """Build only the credential mechanism selected by validated settings."""

        mode = settings.effective_auth_mode
        if mode is AuthMode.SHARED_KEY:
            return cls(
                mode,
                shared_key_verifier=ApiKeyVerifier(settings.nanoloop_api_key),
                clock=clock,
            )
        if mode is AuthMode.PRINCIPAL:
            pepper = settings.credential_pepper
            if pepper is None:  # Settings validation is the primary fail-fast boundary.
                raise ValueError("principal authentication requires a credential pepper")
            return cls(
                mode,
                database=database,
                hasher=CredentialHasher(pepper),
                clock=clock,
            )
        return cls(AuthMode.DISABLED, clock=clock)

    @classmethod
    def from_legacy_verifier(cls, verifier: ApiKeyVerifier) -> RequestAuthenticator:
        """Retain the old middleware construction surface for downstream integrations."""

        if verifier.enabled:
            return cls(AuthMode.SHARED_KEY, shared_key_verifier=verifier)
        return cls(AuthMode.DISABLED)

    @property
    def shared_key_verifier(self) -> ApiKeyVerifier:
        """Expose the compatibility verifier without exposing its secret or digest."""

        return self._shared_key_verifier

    async def authenticate(self, values: Sequence[str]) -> AuthenticationDecision:
        """Return one safe decision; principal persistence failures never downgrade mode."""

        if self.mode is AuthMode.DISABLED:
            return AuthenticationDecision(
                outcome="authenticated",
                reason="authentication_disabled",
                principal=legacy_principal_context(AuthMode.DISABLED),
            )
        if self.mode is AuthMode.SHARED_KEY:
            if self._shared_key_verifier.matches(values):
                return AuthenticationDecision(
                    outcome="authenticated",
                    reason="shared_key_valid",
                    principal=legacy_principal_context(AuthMode.SHARED_KEY),
                )
            return AuthenticationDecision(outcome="rejected", reason="credential_rejected")

        if len(values) != 1:
            return AuthenticationDecision(outcome="rejected", reason="credential_rejected")
        token = values[0]
        hasher = self._hasher
        if hasher is None:  # Construction invariant; fail closed if wiring is corrupted.
            return AuthenticationDecision(outcome="unavailable", reason="invalid_wiring")
        try:
            parsed = parse_credential_token(token)
            candidate_digest = hasher.digest(token)
        except (TypeError, ValueError):
            # Keep malformed inputs on the same local HMAC/constant-time path promised by the
            # credential primitive.  No database query is needed because no canonical ID exists.
            hasher.verify(token, None)
            return AuthenticationDecision(outcome="rejected", reason="credential_rejected")

        try:
            result = await run_in_threadpool(
                self._authenticate_principal,
                parsed.credential_id,
                candidate_digest,
                self._clock(),
            )
        except Exception:
            return AuthenticationDecision(outcome="unavailable", reason="backend_unavailable")
        if result.authenticated and result.principal is not None:
            return AuthenticationDecision(
                outcome="authenticated",
                reason="credential_active",
                principal=result.principal,
            )
        return AuthenticationDecision(outcome="rejected", reason=result.status.value)

    def rate_limit_bucket(self, values: Sequence[str]) -> RateLimitBucket:
        """Classify without database access so rate limiting stays ahead of authentication."""

        if self.mode is AuthMode.DISABLED:
            return "service"
        if self.mode is AuthMode.SHARED_KEY:
            return "authenticated" if self._shared_key_verifier.matches(values) else "anonymous"
        # A syntactically valid principal token is trivial for an unauthenticated caller to forge.
        # Promoting token-shaped input into the shared authenticated bucket would let an attacker
        # consume the real principals' capacity before the database check runs.  Principal mode
        # therefore remains in the pre-authentication bucket; a future two-stage limiter can add
        # bounded per-principal buckets after the single authentication query has succeeded.
        return "anonymous"

    def _authenticate_principal(
        self,
        credential_id: str,
        candidate_digest: bytes,
        now: datetime,
    ) -> AuthenticationResult:
        database = self._database
        if database is None:  # Construction invariant.
            raise RuntimeError("principal authentication database is unavailable")
        with database.session() as session:
            return IdentityService.from_session(session).authenticate(
                credential_id,
                candidate_digest,
                now=now,
            )

    def __repr__(self) -> str:
        return f"{type(self).__name__}(mode={self.mode.value!r}, credentials=<redacted>)"
