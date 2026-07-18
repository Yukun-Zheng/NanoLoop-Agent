from __future__ import annotations

import base64
import hashlib
import hmac
import inspect
import json
from collections.abc import Mapping
from dataclasses import replace
from typing import Any

import pytest
from pydantic import SecretStr

import app.storage.file_tokens_v2 as file_tokens_v2
from app.storage.file_tokens_v2 import (
    FILE_TOKEN_V2_MAX_UNIX_TIMESTAMP,
    FileTokenV2Audience,
    FileTokenV2Claims,
    FileTokenV2Error,
    FileTokenV2KeyRing,
    FileTokenV2Purpose,
)

_TENANT_ID = f"tnt_{'1' * 32}"
_OTHER_TENANT_ID = f"tnt_{'2' * 32}"
_PRINCIPAL_ID = f"prn_{'3' * 32}"
_OTHER_PRINCIPAL_ID = f"prn_{'4' * 32}"
_JOB_ID = f"job_{'5' * 32}"
_ARTIFACT_ID = f"art_{'6' * 32}"
_SHA256 = "7" * 64
_OLD_KEY = b"old-file-token-key-material-32!!"
_NEW_KEY = b"new-file-token-key-material-32!!"


def _ring(
    *,
    keys: Mapping[str, bytes | str | SecretStr] | None = None,
    active_kid: str = "key-old",
    max_ttl_seconds: int = 100,
    clock_skew_seconds: int = 0,
) -> FileTokenV2KeyRing:
    return FileTokenV2KeyRing(
        {"key-old": _OLD_KEY} if keys is None else keys,
        active_kid=active_kid,
        max_ttl_seconds=max_ttl_seconds,
        clock_skew_seconds=clock_skew_seconds,
    )


def _claims(
    ring: FileTokenV2KeyRing | None = None,
    *,
    purpose: FileTokenV2Purpose | str = FileTokenV2Purpose.DOWNLOAD_RUN_ARTIFACT,
    ttl_seconds: int = 20,
    now: int = 100,
    not_before: int | None = None,
) -> FileTokenV2Claims:
    issuer = _ring() if ring is None else ring
    return issuer.create_claims(
        tenant_id=_TENANT_ID,
        principal_id=_PRINCIPAL_ID,
        job_id=_JOB_ID,
        artifact_id=_ARTIFACT_ID,
        purpose=purpose,
        sha256=_SHA256,
        ttl_seconds=ttl_seconds,
        now=now,
        not_before=not_before,
    )


def _base64url(value: bytes) -> str:
    return base64.urlsafe_b64encode(value).rstrip(b"=").decode("ascii")


def _decode_payload(token: str) -> dict[str, Any]:
    encoded = token.split(".")[2]
    decoded = base64.urlsafe_b64decode(encoded + "=" * (-len(encoded) % 4))
    payload = json.loads(decoded)
    assert isinstance(payload, dict)
    return payload


def _signed_raw_payload(
    payload_bytes: bytes,
    *,
    kid: str = "key-old",
    key: bytes = _OLD_KEY,
    encoded_payload: str | None = None,
) -> str:
    payload_part = _base64url(payload_bytes) if encoded_payload is None else encoded_payload
    signed = f"v2.{kid}.{payload_part}".encode("ascii")
    signature = _base64url(hmac.new(key, signed, hashlib.sha256).digest())
    return f"{signed.decode('ascii')}.{signature}"


def _assert_invalid(ring: FileTokenV2KeyRing, token: object, *, now: int = 100) -> None:
    with pytest.raises(FileTokenV2Error) as error:
        ring.verify(token, now=now)  # type: ignore[arg-type]
    assert str(error.value) == "invalid file token"


def test_issue_uses_exact_canonical_envelope_and_claim_set() -> None:
    ring = _ring()
    claims = _claims(ring)

    token = ring.issue(claims)

    version, kid, encoded_payload, encoded_signature = token.split(".")
    assert (version, kid) == ("v2", "key-old")
    assert "=" not in encoded_payload
    assert "=" not in encoded_signature
    payload_bytes = base64.urlsafe_b64decode(encoded_payload + "=" * (-len(encoded_payload) % 4))
    payload = _decode_payload(token)
    assert set(payload) == {
        "v",
        "tid",
        "sub",
        "jid",
        "aid",
        "pur",
        "aud",
        "sha256",
        "iat",
        "nbf",
        "exp",
        "jti",
    }
    assert "path" not in payload
    assert "credential" not in payload
    assert "credential_id" not in payload
    assert payload["aud"] == "nanoloop-api:file-download"
    assert payload_bytes == json.dumps(payload, separators=(",", ":"), sort_keys=True).encode()
    expected_signature = hmac.new(
        _OLD_KEY,
        f"v2.key-old.{encoded_payload}".encode("ascii"),
        hashlib.sha256,
    ).digest()
    assert hmac.compare_digest(
        base64.urlsafe_b64decode(encoded_signature + "="), expected_signature
    )
    assert len(base64.urlsafe_b64decode(f"{payload['jti']}==")) == 16


def test_verify_returns_frozen_claims_and_checks_the_complete_expected_context() -> None:
    ring = _ring()
    issued = _claims(ring)
    token = ring.issue(issued)

    observed = ring.verify(
        token,
        now=101,
        expected_tenant_id=_TENANT_ID,
        expected_principal_id=_PRINCIPAL_ID,
        expected_audience="nanoloop-api:file-download",
        expected_purpose="download.run_artifact",
    )

    assert observed == issued
    assert observed.tenant_id == _TENANT_ID
    assert observed.principal_id == _PRINCIPAL_ID
    assert observed.job_id == _JOB_ID
    assert observed.artifact_id == _ARTIFACT_ID
    assert observed.purpose is FileTokenV2Purpose.DOWNLOAD_RUN_ARTIFACT
    assert observed.audience is FileTokenV2Audience.FILE_DOWNLOAD
    assert observed.issued_at == 100
    assert observed.not_before == 100
    assert observed.expires_at == 120
    assert observed.token_id == observed.jti
    with pytest.raises((AttributeError, TypeError)):
        observed.tid = _OTHER_TENANT_ID  # type: ignore[misc]


@pytest.mark.parametrize(
    ("purpose", "audience"),
    [
        (FileTokenV2Purpose.DOWNLOAD_ORIGINAL_IMAGE, FileTokenV2Audience.FILE_DOWNLOAD),
        (FileTokenV2Purpose.DOWNLOAD_RUN_ARTIFACT, FileTokenV2Audience.FILE_DOWNLOAD),
        (FileTokenV2Purpose.DOWNLOAD_ANALYSIS_EXPORT, FileTokenV2Audience.FILE_DOWNLOAD),
        (
            FileTokenV2Purpose.REVIEW_CORRECTED_MASK,
            FileTokenV2Audience.REVIEW_CORRECTED_MASK,
        ),
    ],
)
def test_create_claims_derives_the_only_allowed_audience_for_each_purpose(
    purpose: FileTokenV2Purpose,
    audience: FileTokenV2Audience,
) -> None:
    claims = _claims(purpose=purpose)
    assert claims.pur is purpose
    assert claims.aud is audience


def test_audience_values_are_namespaced_to_the_nanoloop_api_consumer() -> None:
    assert FileTokenV2Audience.FILE_DOWNLOAD.value == "nanoloop-api:file-download"
    assert FileTokenV2Audience.REVIEW_CORRECTED_MASK.value == "nanoloop-api:review-corrected-mask"


def test_claims_generate_distinct_canonical_128_bit_jtis() -> None:
    ring = _ring()
    first = _claims(ring)
    second = _claims(ring)
    assert first.jti != second.jti
    for claims in (first, second):
        assert len(claims.jti) == 22
        assert "=" not in claims.jti
        assert len(base64.urlsafe_b64decode(f"{claims.jti}==")) == 16


def test_key_rotation_issues_with_active_key_and_retains_old_tokens_for_verification() -> None:
    old_ring = _ring()
    old_token = old_ring.issue(_claims(old_ring))
    rotating_ring = _ring(
        keys={"key-old": _OLD_KEY, "key-new": _NEW_KEY},
        active_kid="key-new",
    )

    assert rotating_ring.verify(old_token, now=101).aid == _ARTIFACT_ID
    new_token = rotating_ring.issue(_claims(rotating_ring))
    assert new_token.startswith("v2.key-new.")
    assert rotating_ring.verify(new_token, now=101).aid == _ARTIFACT_ID

    retired_ring = _ring(keys={"key-new": _NEW_KEY}, active_kid="key-new")
    _assert_invalid(retired_ring, old_token, now=101)


def test_kid_is_covered_by_the_signature_even_when_retained_keys_share_material() -> None:
    ring = _ring(keys={"key-old": _OLD_KEY, "key-new": _OLD_KEY})
    token = ring.issue(_claims(ring))
    version, _, payload, signature = token.split(".")
    _assert_invalid(ring, f"{version}.key-new.{payload}.{signature}", now=101)


def test_unknown_kid_and_tampering_share_one_error_and_both_execute_hmac(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ring = _ring()
    token = ring.issue(_claims(ring))
    _, _, payload, signature = token.split(".")
    unknown = f"v2.key-missing.{payload}.{signature}"
    tampered = f"v2.key-old.{payload[:-1]}A.{signature}"
    real_hmac_new = hmac.new
    calls: list[bytes] = []

    def observed_hmac_new(
        key: bytes,
        msg: bytes | None = None,
        digestmod: Any = None,
    ) -> hmac.HMAC:
        calls.append(key)
        return real_hmac_new(key, msg, digestmod)

    monkeypatch.setattr(hmac, "new", observed_hmac_new)
    _assert_invalid(ring, unknown, now=101)
    _assert_invalid(ring, tampered, now=101)
    assert len(calls) == 2
    assert calls[0] != _OLD_KEY
    assert calls[1] == _OLD_KEY


@pytest.mark.parametrize(
    "expected",
    [
        {"expected_tenant_id": _OTHER_TENANT_ID},
        {"expected_principal_id": _OTHER_PRINCIPAL_ID},
        {"expected_audience": FileTokenV2Audience.REVIEW_CORRECTED_MASK},
        {"expected_purpose": FileTokenV2Purpose.DOWNLOAD_ORIGINAL_IMAGE},
        {"expected_tenant_id": "not-a-tenant"},
        {"expected_principal_id": "not-a-principal"},
        {"expected_audience": "not-an-audience"},
        {"expected_purpose": "not-a-purpose"},
    ],
)
def test_expected_context_mismatches_are_uniformly_rejected(expected: dict[str, Any]) -> None:
    ring = _ring()
    token = ring.issue(_claims(ring))
    with pytest.raises(FileTokenV2Error, match=r"^invalid file token$"):
        ring.verify(token, now=101, **expected)


def test_time_window_applies_bounded_clock_skew_at_both_edges() -> None:
    ring = _ring(clock_skew_seconds=5)
    claims = _claims(ring, now=100, not_before=110, ttl_seconds=20)
    token = ring.issue(claims)

    _assert_invalid(ring, token, now=104)
    assert ring.verify(token, now=105) == claims
    assert ring.verify(token, now=124) == claims
    _assert_invalid(ring, token, now=125)


def test_zero_skew_uses_a_half_open_not_before_expiration_window() -> None:
    ring = _ring(clock_skew_seconds=0)
    claims = _claims(ring, now=100, not_before=105, ttl_seconds=10)
    token = ring.issue(claims)
    _assert_invalid(ring, token, now=104)
    assert ring.verify(token, now=105) == claims
    assert ring.verify(token, now=109) == claims
    _assert_invalid(ring, token, now=110)


def test_ttl_is_bounded_during_creation_issuance_and_verification() -> None:
    ring = _ring(max_ttl_seconds=20)
    exact = _claims(ring, ttl_seconds=20)
    assert ring.verify(ring.issue(exact), now=100) == exact
    with pytest.raises(ValueError, match="ttl_seconds"):
        _claims(ring, ttl_seconds=21)

    overly_long = replace(exact, exp=121)
    with pytest.raises(ValueError, match="maximum TTL"):
        ring.issue(overly_long)

    permissive_ring = _ring(max_ttl_seconds=30)
    long_token = permissive_ring.issue(_claims(permissive_ring, ttl_seconds=30))
    _assert_invalid(ring, long_token)


def test_timestamp_upper_bound_and_explicit_not_before_are_strict() -> None:
    ring = _ring(max_ttl_seconds=10)
    edge = _claims(ring, ttl_seconds=10, now=FILE_TOKEN_V2_MAX_UNIX_TIMESTAMP - 10)
    assert edge.exp == FILE_TOKEN_V2_MAX_UNIX_TIMESTAMP
    with pytest.raises(ValueError, match="expiration"):
        _claims(ring, ttl_seconds=10, now=FILE_TOKEN_V2_MAX_UNIX_TIMESTAMP - 9)
    with pytest.raises(ValueError, match="iat <= nbf <= exp"):
        _claims(ring, ttl_seconds=10, now=100, not_before=99)
    with pytest.raises(ValueError, match="iat <= nbf <= exp"):
        _claims(ring, ttl_seconds=10, now=100, not_before=111)
    with pytest.raises(ValueError, match="not_before"):
        _claims(ring, ttl_seconds=10, now=100, not_before=True)


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"v": True}, "version"),
        ({"v": 1}, "version"),
        ({"v": 2.0}, "version"),
        ({"tid": f"tnt_{'A' * 32}"}, "tenant ID"),
        ({"sub": f"prn_{'A' * 32}"}, "principal ID"),
        ({"jid": ""}, "job ID"),
        ({"jid": "job/unsafe"}, "job ID"),
        ({"jid": "j" * 65}, "job ID"),
        ({"jid": "job\nunsafe"}, "job ID"),
        ({"aid": f"img_{'6' * 32}"}, "artifact ID"),
        ({"aid": f"art_{'A' * 32}"}, "artifact ID"),
        ({"pur": "download.anything"}, "purpose"),
        ({"aud": "nanoloop-api:file-download"}, "audience"),
        ({"aud": FileTokenV2Audience.REVIEW_CORRECTED_MASK}, "purpose"),
        ({"sha256": "A" * 64}, "SHA-256"),
        ({"iat": True}, "iat"),
        ({"nbf": -1}, "nbf"),
        ({"exp": FILE_TOKEN_V2_MAX_UNIX_TIMESTAMP + 1}, "exp"),
        ({"nbf": 99}, "iat <= nbf <= exp"),
        ({"exp": 99}, "iat <= nbf <= exp"),
        ({"jti": _base64url(bytes(15))}, "token ID"),
        ({"jti": f"{_base64url(bytes(16))}="}, "token ID"),
    ],
)
def test_claim_fields_have_strict_canonical_shapes(
    changes: dict[str, object],
    message: str,
) -> None:
    claims = _claims()
    with pytest.raises(ValueError, match=message):
        replace(claims, **changes)  # type: ignore[arg-type]


def test_job_id_accepts_the_bounded_log_safe_boundary() -> None:
    claims = replace(_claims(), jid=f"j{'._-' * 20}abc")
    assert len(claims.jid) == 64


@pytest.mark.parametrize(
    "mutation",
    [
        lambda payload: {**payload, "path": "jobs/secret/file.tif"},
        lambda payload: {**payload, "credential_id": f"crd_{'8' * 32}"},
        lambda payload: {key: value for key, value in payload.items() if key != "aid"},
    ],
)
def test_payload_requires_exact_keys(mutation: Any) -> None:
    ring = _ring()
    payload = _claims(ring).as_payload()
    mutated = mutation(payload)
    token = _signed_raw_payload(json.dumps(mutated, separators=(",", ":"), sort_keys=True).encode())
    _assert_invalid(ring, token)


def test_payload_rejects_duplicate_keys_and_noncanonical_json() -> None:
    ring = _ring()
    payload = _claims(ring).as_payload()
    canonical = json.dumps(payload, separators=(",", ":"), sort_keys=True)
    duplicate = canonical[:-1] + ',"v":2}'
    noncanonical = json.dumps(payload, sort_keys=False).encode()

    _assert_invalid(ring, _signed_raw_payload(duplicate.encode()))
    _assert_invalid(ring, _signed_raw_payload(noncanonical))


def test_payload_and_signature_require_canonical_unpadded_base64url() -> None:
    ring = _ring()
    payload_bytes = json.dumps(
        _claims(ring).as_payload(), separators=(",", ":"), sort_keys=True
    ).encode()
    encoded_payload = _base64url(payload_bytes)
    padded_payload = _signed_raw_payload(
        payload_bytes,
        encoded_payload=f"{encoded_payload}=",
    )
    _assert_invalid(ring, padded_payload)

    token = ring.issue(_claims(ring))
    version, kid, payload, signature = token.split(".")
    _assert_invalid(ring, f"{version}.{kid}.{payload}.{signature}=")
    _assert_invalid(ring, _signed_raw_payload(b"\x00", encoded_payload="AB"))


@pytest.mark.parametrize(
    "token",
    [
        "",
        "v2.only-three.parts",
        "v1.key-old.payload.signature",
        "v2.key.old.payload.signature",
        "v2.\N{SNOWMAN}.payload.signature",
        True,
    ],
)
def test_malformed_envelopes_are_uniformly_rejected(token: object) -> None:
    _assert_invalid(_ring(), token)


def test_token_length_is_bounded_for_issue_and_verify(monkeypatch: pytest.MonkeyPatch) -> None:
    ring = _ring()
    token = ring.issue(_claims(ring))
    monkeypatch.setattr(file_tokens_v2, "FILE_TOKEN_V2_MAX_LENGTH", len(token) - 1)
    with pytest.raises(ValueError, match="supported length"):
        ring.issue(_claims(ring))
    _assert_invalid(ring, token)


@pytest.mark.parametrize(
    ("keys", "active_kid", "message"),
    [
        ({"key-old": b"short"}, "key-old", "at least 32"),
        ({"key-old": _OLD_KEY}, "key-missing", "not present"),
        ({"Key-Old": _OLD_KEY}, "Key-Old", "key ID"),
        ({"key.old": _OLD_KEY}, "key.old", "key ID"),
        ({"key-": _OLD_KEY}, "key-", "key ID"),
        ({"k" * 33: _OLD_KEY}, "k" * 33, "key ID"),
    ],
)
def test_keyring_rejects_weak_keys_and_noncanonical_kids(
    keys: dict[str, bytes],
    active_kid: str,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        _ring(keys=keys, active_kid=active_kid)


def test_keyring_accepts_secret_text_but_never_exposes_key_material() -> None:
    secret = "a-secret-signing-key-that-is-long-enough"
    ring = _ring(keys={"key-old": SecretStr(secret)})
    representation = repr(ring)
    assert ring.active_kid == "key-old"
    assert ring.retained_kids == ("key-old",)
    assert ring.max_ttl_seconds == 100
    assert ring.clock_skew_seconds == 0
    assert secret not in representation
    assert "redacted" in representation


@pytest.mark.parametrize(
    ("max_ttl", "clock_skew"),
    [
        (True, 0),
        (0, 0),
        (100, True),
        (100, -1),
        (100, 301),
    ],
)
def test_keyring_time_policy_rejects_boolean_and_out_of_range_values(
    max_ttl: object,
    clock_skew: object,
) -> None:
    with pytest.raises(ValueError):
        FileTokenV2KeyRing(
            {"key-old": _OLD_KEY},
            active_kid="key-old",
            max_ttl_seconds=max_ttl,  # type: ignore[arg-type]
            clock_skew_seconds=clock_skew,  # type: ignore[arg-type]
        )


def test_codec_surface_cannot_accept_a_path_or_credential_claim() -> None:
    parameters = inspect.signature(FileTokenV2KeyRing.create_claims).parameters
    assert "path" not in parameters
    assert "credential" not in parameters
    assert "credential_id" not in parameters
    with pytest.raises(TypeError):
        _ring().create_claims(  # type: ignore[call-arg]
            tenant_id=_TENANT_ID,
            principal_id=_PRINCIPAL_ID,
            job_id=_JOB_ID,
            artifact_id=_ARTIFACT_ID,
            purpose=FileTokenV2Purpose.DOWNLOAD_RUN_ARTIFACT,
            sha256=_SHA256,
            ttl_seconds=20,
            now=100,
            credential_id=f"crd_{'8' * 32}",
        )
