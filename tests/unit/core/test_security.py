from __future__ import annotations

import asyncio
from pathlib import Path

import pytest
from pydantic import SecretStr
from sqlalchemy import event

from app.authentication import RequestAuthenticator
from app.core.config import Settings
from app.core.identity import CredentialHasher, issue_credential
from app.core.security import ApiKeyVerifier, cors_origins, normalize_http_origin, trusted_hosts
from app.db.session import Database


def test_local_security_defaults_are_narrow() -> None:
    settings = Settings(app_env="test")

    assert trusted_hosts(settings) == ["localhost", "127.0.0.1", "testserver"]
    assert cors_origins(settings) == [
        "http://127.0.0.1:3000",
        "http://localhost:3000",
    ]
    assert cors_origins(Settings(app_env="production")) == []


def test_security_allowlists_are_config_driven_and_canonicalized() -> None:
    settings = Settings(
        app_env="production",
        trusted_hosts="API.EXAMPLE.test,localhost,api.example.test.",
        cors_allow_origins="HTTPS://UI.EXAMPLE.test:443/, http://127.0.0.1:3000",
    )

    assert trusted_hosts(settings) == ["api.example.test", "localhost"]
    assert cors_origins(settings) == [
        "https://ui.example.test",
        "http://127.0.0.1:3000",
    ]


@pytest.mark.parametrize(
    "value",
    ["null", "https://example.test/path", "https://user@example.test", "*"],
)
def test_non_origin_values_are_rejected(value: str) -> None:
    assert normalize_http_origin(value) is None
    with pytest.raises(ValueError, match=r"HTTP\(S\) origins"):
        cors_origins(Settings(app_env="production", cors_allow_origins=value))


def test_empty_trusted_host_configuration_fails_closed() -> None:
    with pytest.raises(ValueError, match="TRUSTED_HOSTS"):
        trusted_hosts(Settings(trusted_hosts=" , "))


def test_api_key_verifier_is_explicitly_disabled_without_a_secret() -> None:
    verifier = ApiKeyVerifier(None)

    assert verifier.enabled is False
    assert verifier.matches([]) is False
    assert verifier.matches(["unused"]) is False


def test_api_key_verifier_accepts_one_exact_value() -> None:
    secret = "valid_api_key_0123456789_ABCDEFG"
    verifier = ApiKeyVerifier(SecretStr(secret))

    assert verifier.enabled is True
    assert verifier.matches([secret]) is True
    assert verifier.matches([]) is False
    assert verifier.matches(["wrong_api_key_0123456789_ABCDEFG"]) is False
    assert verifier.matches([secret, secret]) is False
    assert secret not in repr(verifier)


def test_api_key_verifier_hashes_both_values_before_comparison(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: list[tuple[bytes, bytes]] = []

    def record_comparison(candidate: bytes, expected: bytes) -> bool:
        observed.append((candidate, expected))
        return candidate == expected

    monkeypatch.setattr("app.core.security.compare_digest", record_comparison)
    secret = "valid_api_key_0123456789_ABCDEFG"
    verifier = ApiKeyVerifier(secret)

    assert verifier.matches([secret]) is True
    assert len(observed) == 1
    assert len(observed[0][0]) == 32
    assert len(observed[0][1]) == 32
    assert secret.encode() not in observed[0]


def test_principal_rate_limit_classification_is_anonymous_and_never_queries_database(
    tmp_path: Path,
) -> None:
    pepper = "rate-limit-principal-pepper-material-32bytes"
    settings = Settings(
        auth_mode="principal",
        credential_pepper=pepper,
        database_url=f"sqlite:///{tmp_path / 'classification.db'}",
    )
    database = Database(settings)
    statements: list[str] = []

    def record_statement(*_args: object) -> None:
        statements.append("executed")

    event.listen(database.engine, "before_cursor_execute", record_statement)
    try:
        authenticator = RequestAuthenticator.from_settings(settings, database)
        token = issue_credential(pepper).token.get_secret_value()
        assert authenticator.rate_limit_bucket([token]) == "anonymous"
        assert authenticator.rate_limit_bucket([]) == "anonymous"
        assert authenticator.rate_limit_bucket(["malformed"]) == "anonymous"
        assert authenticator.rate_limit_bucket([token, token]) == "anonymous"
        assert statements == []
    finally:
        event.remove(database.engine, "before_cursor_execute", record_statement)
        database.dispose()


def test_malformed_principal_token_runs_dummy_verification_without_database_query(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = Settings(
        auth_mode="principal",
        credential_pepper="dummy-verification-pepper-material-32bytes",
        database_url=f"sqlite:///{tmp_path / 'dummy-verification.db'}",
    )
    database = Database(settings)
    verifications: list[tuple[str, bytes | None]] = []
    statements: list[str] = []

    def record_verification(
        _hasher: CredentialHasher,
        token: SecretStr | str,
        expected_digest: bytes | None,
    ) -> bool:
        raw_token = token.get_secret_value() if isinstance(token, SecretStr) else token
        verifications.append((raw_token, expected_digest))
        return False

    def record_statement(*_args: object) -> None:
        statements.append("executed")

    monkeypatch.setattr(CredentialHasher, "verify", record_verification)
    event.listen(database.engine, "before_cursor_execute", record_statement)
    try:
        authenticator = RequestAuthenticator.from_settings(settings, database)
        decision = asyncio.run(authenticator.authenticate(["malformed-token"]))
        assert decision.outcome == "rejected"
        assert decision.reason == "credential_rejected"
        assert verifications == [("malformed-token", None)]
        assert statements == []
    finally:
        event.remove(database.engine, "before_cursor_execute", record_statement)
        database.dispose()


def test_disabled_and_shared_key_rate_limit_compatibility() -> None:
    disabled = RequestAuthenticator.from_legacy_verifier(ApiKeyVerifier(None))
    shared = RequestAuthenticator.from_legacy_verifier(ApiKeyVerifier("k" * 32))

    assert disabled.rate_limit_bucket([]) == "service"
    assert disabled.rate_limit_bucket(["anything", "duplicated"]) == "service"
    assert shared.rate_limit_bucket(["k" * 32]) == "authenticated"
    assert shared.rate_limit_bucket([]) == "anonymous"
    assert shared.rate_limit_bucket(["k" * 32, "k" * 32]) == "anonymous"
