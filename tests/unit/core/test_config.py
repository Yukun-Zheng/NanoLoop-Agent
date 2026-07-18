import pytest
from pydantic import SecretStr, ValidationError

from app.core.config import Settings


def test_total_request_limit_must_cover_a_single_file_limit() -> None:
    with pytest.raises(ValidationError, match="MAX_REQUEST_MB"):
        Settings(max_upload_mb=10, max_request_mb=9)

    settings = Settings(max_upload_mb=10, max_request_mb=10)
    assert settings.max_request_mb == 10


def test_api_security_defaults_are_disabled() -> None:
    settings = Settings()

    assert settings.nanoloop_api_key is None
    assert settings.api_rate_limit_requests == 0
    assert settings.api_rate_limit_window_seconds == 60.0


def test_api_key_is_secret_and_empty_string_disables_it() -> None:
    assert Settings(nanoloop_api_key="").nanoloop_api_key is None
    assert Settings(nanoloop_api_key=SecretStr("")).nanoloop_api_key is None

    settings = Settings(nanoloop_api_key="valid_api_key_0123456789_ABCDEFG")
    assert settings.nanoloop_api_key is not None
    assert settings.nanoloop_api_key.get_secret_value() == "valid_api_key_0123456789_ABCDEFG"
    assert "valid_api_key_0123456789_ABCDEFG" not in repr(settings)


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

    settings = Settings(api_rate_limit_requests=25, api_rate_limit_window_seconds=10.5)
    assert settings.api_rate_limit_requests == 25
    assert settings.api_rate_limit_window_seconds == 10.5
