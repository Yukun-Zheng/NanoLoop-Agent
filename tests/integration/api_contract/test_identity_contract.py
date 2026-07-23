from __future__ import annotations

import logging
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import event, text

from app.contracts.identity import PrincipalKind, PrincipalRole
from app.core.config import Settings
from app.core.identity import IssuedCredential, issue_credential
from app.db.base import Base
from app.db.identity import IdentityService
from app.db.models import ApiCredential
from app.db.session import Database
from app.main import create_app

_NOW = datetime(2026, 7, 18, 12, 0, tzinfo=UTC)
_PEPPER = "principal-http-test-pepper-material-32-bytes"
_SHARED_KEY = "shared_key_that_must_not_be_a_fallback"
_TENANT_ID = f"tnt_{'a' * 32}"
_PRINCIPAL_ID = f"prn_{'b' * 32}"


class _Gateway:
    def health(self) -> list[object]:
        return []


@dataclass(slots=True)
class IdentityHarness:
    app: FastAPI
    client: TestClient
    database: Database
    issued: IssuedCredential


@pytest.fixture
def identity_harness(tmp_path: Path) -> Iterator[IdentityHarness]:
    settings = Settings(
        app_env="test",
        auth_mode="principal",
        nanoloop_api_key=_SHARED_KEY,
        credential_pepper=_PEPPER,
        database_url=f"sqlite:///{tmp_path / 'identity-http.db'}",
        output_root=tmp_path / "outputs",
        model_registry_path=tmp_path / "registry.yaml",
        faiss_index_path=tmp_path / "faiss.index",
        log_level="WARNING",
        api_rate_limit_requests=20,
        api_principal_preauth_rate_limit_requests=20,
        api_rate_limit_max_buckets=16,
    )
    database = Database(settings)
    Base.metadata.create_all(database.engine)
    issued = issue_credential(_PEPPER)
    with database.session() as session:
        service = IdentityService.from_session(session)
        service.create_tenant(
            tenant_id=_TENANT_ID,
            slug="http-contract",
            display_name="HTTP contract tenant",
            now=_NOW,
        )
        service.create_principal(
            principal_id=_PRINCIPAL_ID,
            tenant_id=_TENANT_ID,
            handle="http-tester",
            display_name="HTTP tester",
            kind=PrincipalKind.USER,
            role=PrincipalRole.ANALYST,
            now=_NOW,
        )
        service.issue_credential(
            credential_id=issued.credential_id,
            principal_id=_PRINCIPAL_ID,
            token_digest=issued.digest,
            label="contract test",
            now=_NOW,
        )
    app = create_app(settings=settings, database=database, inference_gateway=_Gateway())
    client = TestClient(app, raise_server_exceptions=False)
    try:
        yield IdentityHarness(app=app, client=client, database=database, issued=issued)
    finally:
        client.close()
        database.dispose()


def _credential_headers(token: str) -> dict[str, str]:
    return {"X-API-Key": token, "X-Request-ID": "req_identity_contract"}


def _disable_state(harness: IdentityHarness, state: str) -> None:
    with harness.database.session() as session:
        service = IdentityService.from_session(session)
        if state == "revoked":
            assert service.revoke_credential(harness.issued.credential_id, now=_NOW)
        elif state == "credential_disabled":
            assert service.set_credential_enabled(
                harness.issued.credential_id,
                enabled=False,
                now=_NOW,
            )
        elif state == "expired":
            credential = session.get(ApiCredential, harness.issued.credential_id)
            assert credential is not None
            credential.expires_at = datetime(2000, 1, 1, tzinfo=UTC)
        elif state == "principal_disabled":
            assert service.set_principal_enabled(_PRINCIPAL_ID, enabled=False, now=_NOW)
        elif state == "tenant_disabled":
            assert service.set_tenant_enabled(_TENANT_ID, enabled=False, now=_NOW)
        else:  # pragma: no cover - test parameter invariant
            raise AssertionError(state)


def test_principal_mode_accepts_the_persisted_credential_and_never_shared_fallbacks(
    identity_harness: IdentityHarness,
) -> None:
    token = identity_harness.issued.token.get_secret_value()

    accepted = identity_harness.client.get(
        "/api/v1/health",
        headers=_credential_headers(token),
    )
    shared_fallback = identity_harness.client.get(
        "/api/v1/health",
        headers=_credential_headers(_SHARED_KEY),
    )

    assert accepted.status_code == 200
    assert shared_fallback.status_code == 401
    assert shared_fallback.json()["error"]["code"] == "AUTHENTICATION_REQUIRED"


def test_principal_request_performs_one_identity_lookup_without_dependency_recheck(
    identity_harness: IdentityHarness,
) -> None:
    statements: list[str] = []

    def capture_statement(
        _connection: object,
        _cursor: object,
        statement: str,
        _parameters: object,
        _context: object,
        _executemany: bool,
    ) -> None:
        statements.append(" ".join(statement.split()))

    event.listen(identity_harness.database.engine, "before_cursor_execute", capture_statement)
    try:
        response = identity_harness.client.get(
            "/api/v1/health",
            headers=_credential_headers(identity_harness.issued.token.get_secret_value()),
        )
    finally:
        event.remove(
            identity_harness.database.engine,
            "before_cursor_execute",
            capture_statement,
        )

    assert response.status_code == 200
    identity_reads = [
        statement
        for statement in statements
        if "FROM api_credentials JOIN principals" in statement
    ]
    assert len(identity_reads) == 1
    assert identity_harness.app.state.principal_rate_limiter.bucket_count == 1


def test_only_exact_public_routes_remain_anonymous(identity_harness: IdentityHarness) -> None:
    assert identity_harness.client.get("/health").status_code == 200
    assert identity_harness.client.get("/docs").status_code == 200
    assert identity_harness.client.get("/openapi.json").status_code == 200
    assert identity_harness.app.state.principal_preauth_rate_limiter.bucket_count == 0
    assert identity_harness.client.get("/api/v1/health").status_code == 401
    assert identity_harness.client.get("/health/extra").status_code == 401


def test_cors_preflight_bypasses_both_stages_but_plain_options_is_protected(
    identity_harness: IdentityHarness,
) -> None:
    preflight = identity_harness.client.options(
        "/api/v1/health",
        headers={
            "Origin": "http://localhost:3000",
            "Access-Control-Request-Method": "GET",
            "Access-Control-Request-Headers": "X-API-Key",
        },
    )

    assert preflight.status_code == 200
    assert preflight.headers["access-control-allow-origin"] == "http://localhost:3000"
    assert identity_harness.app.state.principal_preauth_rate_limiter.bucket_count == 0
    assert identity_harness.app.state.principal_rate_limiter.bucket_count == 0

    plain_options = identity_harness.client.options("/api/v1/health")

    assert plain_options.status_code == 401
    assert identity_harness.app.state.principal_preauth_rate_limiter.bucket_count == 1
    assert identity_harness.app.state.principal_rate_limiter.bucket_count == 0


@pytest.mark.parametrize(
    "state",
    ["revoked", "credential_disabled", "expired", "principal_disabled", "tenant_disabled"],
)
def test_inactive_identity_states_share_one_http_failure(
    identity_harness: IdentityHarness,
    state: str,
) -> None:
    _disable_state(identity_harness, state)
    response = identity_harness.client.get(
        "/api/v1/health",
        headers=_credential_headers(identity_harness.issued.token.get_secret_value()),
    )

    assert response.status_code == 401
    assert response.headers["cache-control"] == "no-store"
    assert response.headers["www-authenticate"] == 'ApiKey realm="nanoloop"'
    assert response.json()["error"] == {
        "code": "AUTHENTICATION_REQUIRED",
        "message": "需要有效的 API Key",
        "details": {},
        "retryable": False,
    }
    assert identity_harness.app.state.principal_rate_limiter.bucket_count == 0


def test_unknown_malformed_missing_and_duplicate_credentials_are_indistinguishable(
    identity_harness: IdentityHarness,
) -> None:
    unknown = issue_credential(_PEPPER).token.get_secret_value()
    cases: list[Any] = [
        _credential_headers(unknown),
        _credential_headers("malformed"),
        {"X-Request-ID": "req_identity_contract"},
        [
            ("X-API-Key", identity_harness.issued.token.get_secret_value()),
            ("X-API-Key", identity_harness.issued.token.get_secret_value()),
            ("X-Request-ID", "req_identity_contract"),
        ],
    ]
    responses = [
        identity_harness.client.get("/api/v1/health", headers=headers)
        for headers in cases
    ]

    assert {response.status_code for response in responses} == {401}
    assert {response.headers["cache-control"] for response in responses} == {"no-store"}
    assert {response.headers["www-authenticate"] for response in responses} == {
        'ApiKey realm="nanoloop"'
    }
    assert len({str(response.json()["error"]) for response in responses}) == 1
    assert identity_harness.app.state.principal_rate_limiter.bucket_count == 0


def test_principal_database_or_schema_failure_is_a_safe_503(
    identity_harness: IdentityHarness,
) -> None:
    with identity_harness.database.engine.begin() as connection:
        connection.execute(text("ALTER TABLE api_credentials RENAME TO unavailable_credentials"))

    response = identity_harness.client.get(
        "/api/v1/health",
        headers=_credential_headers(identity_harness.issued.token.get_secret_value()),
    )

    assert response.status_code == 503
    assert response.headers["cache-control"] == "no-store"
    assert "www-authenticate" not in response.headers
    assert response.json()["error"] == {
        "code": "AUTHENTICATION_UNAVAILABLE",
        "message": "认证服务暂时不可用",
        "details": {},
        "retryable": True,
    }
    assert identity_harness.app.state.principal_rate_limiter.bucket_count == 0


def test_untrusted_source_cannot_exhaust_other_peers_or_the_principal_bucket(
    identity_harness: IdentityHarness,
) -> None:
    settings = identity_harness.app.state.settings.model_copy(
        update={
            "api_rate_limit_requests": 1,
            "api_principal_preauth_rate_limit_requests": 1,
            "api_rate_limit_max_buckets": 8,
        }
    )
    app = create_app(
        settings=settings,
        database=identity_harness.database,
        inference_gateway=_Gateway(),
    )
    attacker = TestClient(
        app,
        raise_server_exceptions=False,
        client=("192.0.2.1", 1000),
    )
    other_peer = TestClient(
        app,
        raise_server_exceptions=False,
        client=("192.0.2.2", 1000),
    )
    third_peer = TestClient(
        app,
        raise_server_exceptions=False,
        client=("192.0.2.3", 1000),
    )
    token = identity_harness.issued.token.get_secret_value()
    try:
        rejected = attacker.get(
            "/api/v1/health",
            headers={"X-API-Key": "wrong", "X-Forwarded-For": "198.51.100.1"},
        )
        spoofed_retry = attacker.get(
            "/api/v1/health",
            headers={"X-API-Key": token, "X-Forwarded-For": "203.0.113.1"},
        )
        accepted = other_peer.get(
            "/api/v1/health",
            headers={"X-API-Key": token, "X-Forwarded-For": "198.51.100.1"},
        )
        principal_limited = third_peer.get(
            "/api/v1/health",
            headers={"X-API-Key": token},
        )
    finally:
        attacker.close()
        other_peer.close()
        third_peer.close()

    assert rejected.status_code == 401
    assert spoofed_retry.status_code == 429
    assert accepted.status_code == 200
    assert principal_limited.status_code == 429
    assert app.state.principal_preauth_rate_limiter.bucket_count == 3
    assert app.state.principal_rate_limiter.bucket_count == 1


def test_access_log_recovers_safe_principal_state_without_recording_token(
    identity_harness: IdentityHarness,
    caplog: pytest.LogCaptureFixture,
) -> None:
    token = identity_harness.issued.token.get_secret_value()
    with caplog.at_level(logging.INFO, logger="app.api.middleware"):
        response = identity_harness.client.get(
            "/api/v1/health",
            headers=_credential_headers(token),
        )

    assert response.status_code == 200
    completed: Any = next(
        record for record in caplog.records if record.message == "request_completed"
    )
    assert completed.tenant_id == _TENANT_ID
    assert completed.principal_id == _PRINCIPAL_ID
    assert completed.credential_id == identity_harness.issued.credential_id
    assert completed.auth_mode == "principal"
    assert completed.auth_outcome == "authenticated"
    assert completed.auth_reason == "credential_active"
    assert token not in caplog.text
