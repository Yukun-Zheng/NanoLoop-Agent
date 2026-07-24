import pytest
from pydantic import SecretStr, ValidationError

from app.contracts.identity import AuthMode
from app.core.config import Settings


def test_total_request_limit_must_cover_a_single_file_limit() -> None:
    with pytest.raises(ValidationError, match="MAX_REQUEST_MB"):
        Settings(max_upload_mb=10, max_request_mb=9)

    settings = Settings(max_upload_mb=10, max_request_mb=10)
    assert settings.max_request_mb == 10


def test_llm_generation_and_history_limits_are_bounded() -> None:
    settings = Settings(
        llm_timeout_seconds=90,
        llm_max_tokens=1024,
        llm_temperature=0,
        llm_history_turns=8,
        llm_history_max_chars=12_000,
    )

    assert settings.llm_timeout_seconds == 90
    assert settings.llm_max_tokens == 1024
    assert settings.llm_history_turns == 8
    with pytest.raises(ValidationError):
        Settings(llm_history_turns=0)
    with pytest.raises(ValidationError):
        Settings(llm_history_max_chars=100)


def test_api_security_defaults_are_disabled() -> None:
    settings = Settings()

    assert settings.auth_mode == "auto"
    assert settings.effective_auth_mode is AuthMode.DISABLED
    assert settings.nanoloop_api_key is None
    assert settings.credential_pepper is None
    assert settings.api_rate_limit_requests == 0
    assert settings.api_rate_limit_window_seconds == 60.0
    assert settings.api_principal_preauth_rate_limit_requests == 600
    assert settings.api_principal_preauth_rate_limit_window_seconds == 60.0
    assert settings.api_rate_limit_max_buckets == 4_096


def test_api_key_is_secret_and_empty_string_disables_it() -> None:
    assert Settings(nanoloop_api_key="").nanoloop_api_key is None
    assert Settings(nanoloop_api_key=SecretStr("")).nanoloop_api_key is None

    settings = Settings(nanoloop_api_key="valid_api_key_0123456789_ABCDEFG")
    assert settings.nanoloop_api_key is not None
    assert settings.nanoloop_api_key.get_secret_value() == "valid_api_key_0123456789_ABCDEFG"
    assert "valid_api_key_0123456789_ABCDEFG" not in repr(settings)


def test_auto_mode_preserves_legacy_shared_key_behavior() -> None:
    settings = Settings(nanoloop_api_key="valid_api_key_0123456789_ABCDEFG")

    assert settings.effective_auth_mode is AuthMode.SHARED_KEY


def test_explicit_authentication_modes_fail_fast_without_required_secrets() -> None:
    with pytest.raises(ValidationError, match="NANOLOOP_API_KEY"):
        Settings(auth_mode="shared_key")
    with pytest.raises(ValidationError, match="CREDENTIAL_PEPPER"):
        Settings(auth_mode="principal")

    short_pepper = "not-long-enough"
    with pytest.raises(ValidationError, match="at least 32 bytes") as exc_info:
        Settings(auth_mode="principal", credential_pepper=short_pepper)
    assert short_pepper not in str(exc_info.value)


def test_principal_mode_never_resolves_to_shared_key_fallback() -> None:
    key = "valid_api_key_0123456789_ABCDEFG"
    pepper = "p" * 32
    settings = Settings(
        auth_mode="principal",
        nanoloop_api_key=key,
        credential_pepper=pepper,
    )

    assert settings.effective_auth_mode is AuthMode.PRINCIPAL
    assert settings.credential_pepper is not None
    assert pepper not in repr(settings)


def test_explicit_disabled_mode_ignores_a_configured_shared_key() -> None:
    settings = Settings(
        auth_mode="disabled",
        nanoloop_api_key="valid_api_key_0123456789_ABCDEFG",
    )

    assert settings.effective_auth_mode is AuthMode.DISABLED


@pytest.mark.parametrize(
    "value",
    [
        "too-short",
        "a" * 129,
        "contains spaces but is definitely long enough",
        "contains.periods.and.is.definitely.long.enough",
    ],
)
def test_api_key_rejects_values_outside_the_frozen_format(value: str) -> None:
    with pytest.raises(ValidationError, match="NANOLOOP_API_KEY") as exc_info:
        Settings(nanoloop_api_key=value)

    assert value not in str(exc_info.value)
    assert value not in repr(exc_info.value)


def test_api_rate_limit_configuration_is_bounded_at_zero() -> None:
    with pytest.raises(ValidationError):
        Settings(api_rate_limit_requests=-1)
    with pytest.raises(ValidationError):
        Settings(api_rate_limit_requests=1_000_001)
    with pytest.raises(ValidationError):
        Settings(api_rate_limit_window_seconds=0)
    with pytest.raises(ValidationError):
        Settings(api_rate_limit_window_seconds=3_601)
    with pytest.raises(ValidationError):
        Settings(api_principal_preauth_rate_limit_requests=-1)
    with pytest.raises(ValidationError):
        Settings(api_principal_preauth_rate_limit_requests=1_000_001)
    with pytest.raises(ValidationError):
        Settings(api_principal_preauth_rate_limit_window_seconds=0)
    with pytest.raises(ValidationError):
        Settings(api_rate_limit_max_buckets=0)
    with pytest.raises(ValidationError):
        Settings(api_rate_limit_max_buckets=1_000_001)

    settings = Settings(
        api_rate_limit_requests=25,
        api_rate_limit_window_seconds=10.5,
        api_principal_preauth_rate_limit_requests=100,
        api_principal_preauth_rate_limit_window_seconds=12.5,
        api_rate_limit_max_buckets=512,
    )
    assert settings.api_rate_limit_requests == 25
    assert settings.api_rate_limit_window_seconds == 10.5
    assert settings.api_principal_preauth_rate_limit_requests == 100
    assert settings.api_principal_preauth_rate_limit_window_seconds == 12.5
    assert settings.api_rate_limit_max_buckets == 512
