from __future__ import annotations

import pytest

from app.core.config import Settings
from app.core.security import cors_origins, normalize_http_origin, trusted_hosts


def test_local_security_defaults_are_narrow() -> None:
    settings = Settings(app_env="test")

    assert trusted_hosts(settings) == ["localhost", "127.0.0.1", "testserver"]
    assert cors_origins(settings) == [
        "http://127.0.0.1:8501",
        "http://localhost:8501",
    ]
    assert cors_origins(Settings(app_env="production")) == []


def test_security_allowlists_are_config_driven_and_canonicalized() -> None:
    settings = Settings(
        app_env="production",
        trusted_hosts="API.EXAMPLE.test,localhost,api.example.test.",
        cors_allow_origins="HTTPS://UI.EXAMPLE.test:443/, http://127.0.0.1:8501",
    )

    assert trusted_hosts(settings) == ["api.example.test", "localhost"]
    assert cors_origins(settings) == [
        "https://ui.example.test",
        "http://127.0.0.1:8501",
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
