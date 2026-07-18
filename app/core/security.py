"""Small, explicit HTTP security defaults for the competition MVP."""

from __future__ import annotations

import os
import re
from collections.abc import Sequence
from urllib.parse import urlsplit

from app.core.config import Settings

_REQUEST_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,99}$")
_LOCAL_FRONTEND_ORIGINS = (
    "http://127.0.0.1:8501",
    "http://localhost:8501",
)


def trusted_request_id(value: str | None) -> str | None:
    """Accept only log-safe request IDs before reflecting them to clients."""

    if value is None or not _REQUEST_ID_PATTERN.fullmatch(value):
        return None
    return value


def cors_origins(settings: Settings) -> list[str]:
    """Resolve an explicit allowlist; never use wildcard credentials."""

    origins = [
        _require_http_origin(item)
        for item in settings.cors_allow_origins.split(",")
        if item.strip()
    ]
    if origins:
        return _deduplicate(origins)
    if settings.app_env in {"development", "test"}:
        return list(_LOCAL_FRONTEND_ORIGINS)
    return []


def trusted_hosts(settings: Settings) -> list[str]:
    """Return the configured Host-header allowlist with fail-closed empty handling."""

    hosts = [item.strip().casefold().rstrip(".") for item in settings.trusted_hosts.split(",")]
    hosts = [host for host in hosts if host]
    if not hosts:
        raise ValueError("TRUSTED_HOSTS must contain at least one host")
    return _deduplicate(hosts)


def normalize_http_origin(value: str) -> str | None:
    """Canonicalize an HTTP(S) origin, rejecting paths, credentials, and opaque origins."""

    try:
        parsed = urlsplit(value.strip())
        port = parsed.port
    except ValueError:
        return None
    if (
        parsed.scheme.casefold() not in {"http", "https"}
        or parsed.hostname is None
        or parsed.username is not None
        or parsed.password is not None
        or parsed.path not in {"", "/"}
        or parsed.query
        or parsed.fragment
    ):
        return None
    scheme = parsed.scheme.casefold()
    hostname = parsed.hostname.casefold().rstrip(".")
    if ":" in hostname:
        hostname = f"[{hostname}]"
    default_port = 80 if scheme == "http" else 443
    port_suffix = "" if port in {None, default_port} else f":{port}"
    return f"{scheme}://{hostname}{port_suffix}"


def file_token_secret() -> str | None:
    """Return an optional stable secret without ever logging its value."""

    value = os.environ.get("NANOLOOP_FILE_TOKEN_SECRET")
    return value if value else None


def _require_http_origin(value: str) -> str:
    normalized = normalize_http_origin(value)
    if normalized is None:
        raise ValueError("CORS_ALLOW_ORIGINS entries must be HTTP(S) origins without paths")
    return normalized


def _deduplicate(values: Sequence[str]) -> list[str]:
    return list(dict.fromkeys(values))
