from __future__ import annotations

import hashlib
import hmac
import re
from collections.abc import Callable

import pytest
from pydantic import SecretStr, ValidationError

from app.contracts.identity import (
    LEGACY_PRINCIPAL_ID,
    LEGACY_TENANT_ID,
    AuthMode,
    PrincipalContext,
    PrincipalKind,
    PrincipalRole,
    validate_credential_id,
    validate_principal_handle,
    validate_principal_id,
    validate_tenant_id,
    validate_tenant_slug,
)
from app.core.identity import (
    CREDENTIAL_DIGEST_BYTES,
    CREDENTIAL_SECRET_LENGTH,
    CredentialHasher,
    generate_credential_id,
    generate_principal_id,
    generate_tenant_id,
    issue_credential,
    legacy_principal_context,
    parse_credential_token,
)

_TOKEN_PATTERN = re.compile(r"\Anlk_v1_crd_[0-9a-f]{32}_[A-Za-z0-9_-]{43}\Z")
_PEPPER = b"p" * 32


def test_identity_enum_values_are_stable() -> None:
    assert [mode.value for mode in AuthMode] == ["disabled", "shared_key", "principal"]
    assert [kind.value for kind in PrincipalKind] == ["user", "service"]
    assert [role.value for role in PrincipalRole] == [
        "tenant_admin",
        "analyst",
        "viewer",
    ]


def test_principal_context_is_immutable_and_requires_persisted_ids() -> None:
    context = PrincipalContext(
        tenant_id="tnt_11111111111111111111111111111111",
        principal_id="prn_22222222222222222222222222222222",
        credential_id="crd_33333333333333333333333333333333",
        kind=PrincipalKind.USER,
        role=PrincipalRole.ANALYST,
        auth_mode=AuthMode.PRINCIPAL,
    )

    with pytest.raises(ValidationError, match="frozen"):
        context.__setattr__("role", PrincipalRole.VIEWER)
    with pytest.raises(ValidationError, match="requires tenant, principal, and credential IDs"):
        PrincipalContext(
            tenant_id=context.tenant_id,
            principal_id=context.principal_id,
            credential_id=None,
            kind=context.kind,
            role=context.role,
            auth_mode=AuthMode.PRINCIPAL,
        )


@pytest.mark.parametrize("mode", [AuthMode.DISABLED, AuthMode.SHARED_KEY])
def test_compatibility_modes_only_allow_the_fixed_legacy_identity(mode: AuthMode) -> None:
    invalid_shapes = (
        {},
        {"tenant_id": LEGACY_TENANT_ID},
        {
            "tenant_id": LEGACY_TENANT_ID,
            "principal_id": LEGACY_PRINCIPAL_ID,
            "credential_id": "crd_33333333333333333333333333333333",
        },
        {
            "tenant_id": "tnt_11111111111111111111111111111111",
            "principal_id": "prn_22222222222222222222222222222222",
        },
    )

    for values in invalid_shapes:
        with pytest.raises(ValidationError, match="fixed legacy principal"):
            PrincipalContext(
                **values,
                kind=PrincipalKind.SERVICE,
                role=PrincipalRole.TENANT_ADMIN,
                auth_mode=mode,
            )

    for kind, role in (
        (PrincipalKind.USER, PrincipalRole.TENANT_ADMIN),
        (PrincipalKind.SERVICE, PrincipalRole.VIEWER),
    ):
        with pytest.raises(ValidationError, match="fixed legacy principal"):
            PrincipalContext(
                tenant_id=LEGACY_TENANT_ID,
                principal_id=LEGACY_PRINCIPAL_ID,
                credential_id=None,
                kind=kind,
                role=role,
                auth_mode=mode,
            )


@pytest.mark.parametrize("mode", [AuthMode.DISABLED, AuthMode.SHARED_KEY])
def test_legacy_context_uses_fixed_non_credential_identity(mode: AuthMode) -> None:
    context = legacy_principal_context(mode)

    assert context == PrincipalContext(
        tenant_id=LEGACY_TENANT_ID,
        principal_id=LEGACY_PRINCIPAL_ID,
        credential_id=None,
        kind=PrincipalKind.SERVICE,
        role=PrincipalRole.TENANT_ADMIN,
        auth_mode=mode,
    )


def test_legacy_context_rejects_principal_mode() -> None:
    with pytest.raises(ValueError, match="persisted principal"):
        legacy_principal_context(AuthMode.PRINCIPAL)


def test_slug_handle_and_entity_id_validation_are_canonical() -> None:
    assert validate_tenant_slug("lab-7") == "lab-7"
    assert validate_tenant_slug("a" * 63) == "a" * 63
    assert validate_principal_handle("service.reader-1") == "service.reader-1"
    assert validate_principal_handle("a" * 64) == "a" * 64
    assert validate_tenant_id("tnt_" + "a" * 32) == "tnt_" + "a" * 32
    assert validate_principal_id("prn_" + "b" * 32) == "prn_" + "b" * 32
    assert validate_credential_id("crd_" + "c" * 32) == "crd_" + "c" * 32


@pytest.mark.parametrize(
    ("validator", "value"),
    [
        (validate_tenant_slug, "-lab"),
        (validate_tenant_slug, "Lab"),
        (validate_tenant_slug, "a" * 64),
        (validate_principal_handle, ".operator"),
        (validate_principal_handle, "operator@lab"),
        (validate_principal_handle, "a" * 65),
        (validate_tenant_id, "tnt_" + "A" * 32),
        (validate_principal_id, "principal_" + "1" * 32),
        (validate_credential_id, "crd_" + "1" * 31),
    ],
)
def test_identifiers_reject_noncanonical_values_without_reflection(
    validator: Callable[[str], str], value: str
) -> None:
    with pytest.raises(ValueError) as exc_info:
        validator(value)
    assert value not in str(exc_info.value)


def test_generated_ids_are_random_and_validate() -> None:
    tenant_ids = {generate_tenant_id() for _ in range(4)}
    principal_ids = {generate_principal_id() for _ in range(4)}
    credential_ids = {generate_credential_id() for _ in range(4)}

    assert len(tenant_ids) == len(principal_ids) == len(credential_ids) == 4
    assert all(validate_tenant_id(value) == value for value in tenant_ids)
    assert all(validate_principal_id(value) == value for value in principal_ids)
    assert all(validate_credential_id(value) == value for value in credential_ids)


def test_issued_credential_has_exact_format_and_peppered_digest() -> None:
    issued = issue_credential(_PEPPER)
    token = issued.token.get_secret_value()

    assert _TOKEN_PATTERN.fullmatch(token) is not None
    assert len(parse_credential_token(token).secret.get_secret_value()) == CREDENTIAL_SECRET_LENGTH
    assert parse_credential_token(token).credential_id == issued.credential_id
    assert issued.digest == hmac.new(_PEPPER, token.encode("ascii"), hashlib.sha256).digest()
    assert len(issued.digest) == CREDENTIAL_DIGEST_BYTES


def test_issued_credentials_use_fresh_id_and_secret_randomness() -> None:
    issued = [issue_credential(_PEPPER) for _ in range(4)]

    assert len({item.credential_id for item in issued}) == 4
    assert len({item.token.get_secret_value() for item in issued}) == 4
    assert len({item.digest for item in issued}) == 4


@pytest.mark.parametrize(
    "token",
    [
        "",
        "nlk_v2_crd_" + "1" * 32 + "_" + "A" * 43,
        "nlk_v1_crd_" + "A" * 32 + "_" + "A" * 43,
        "nlk_v1_crd_" + "1" * 31 + "_" + "A" * 43,
        "nlk_v1_crd_" + "1" * 32 + "_" + "A" * 42,
        "nlk_v1_crd_" + "1" * 32 + "_" + "A" * 42 + "=",
        "nlk_v1_crd_" + "1" * 32 + "_" + "A" * 42 + "B",
        " nlk_v1_crd_" + "1" * 32 + "_" + "A" * 43,
        "nlk_v1_crd_" + "1" * 32 + "_" + "A" * 42 + "é",
    ],
)
def test_token_parser_strictly_rejects_malformed_or_noncanonical_input(token: str) -> None:
    with pytest.raises(ValueError, match="invalid NanoLoop credential token") as exc_info:
        parse_credential_token(token)
    if token:
        assert token not in str(exc_info.value)


def test_hasher_requires_32_pepper_bytes_and_does_not_reveal_pepper() -> None:
    pepper = "private-pepper-value-1234567890"
    assert len(pepper.encode()) == 31

    with pytest.raises(ValueError, match="at least 32 bytes") as exc_info:
        CredentialHasher(pepper)
    assert pepper not in str(exc_info.value)

    multibyte_pepper = "密" * 11
    assert len(multibyte_pepper.encode("utf-8")) == 33
    assert "密" not in repr(CredentialHasher(SecretStr(multibyte_pepper)))


def test_hasher_verifies_exact_token_digest_and_pepper() -> None:
    issued = issue_credential(_PEPPER)
    token = issued.token.get_secret_value()

    assert CredentialHasher(_PEPPER).verify(SecretStr(token), issued.digest) is True
    assert CredentialHasher(b"q" * 32).verify(token, issued.digest) is False
    other = issue_credential(_PEPPER)
    assert CredentialHasher(_PEPPER).verify(other.token, issued.digest) is False


def test_verifier_keeps_constant_time_comparison_on_all_failure_paths(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    comparisons: list[tuple[bytes, bytes]] = []

    def record_comparison(candidate: bytes, expected: bytes) -> bool:
        comparisons.append((candidate, expected))
        return candidate == expected

    monkeypatch.setattr("app.core.identity.hmac.compare_digest", record_comparison)
    issued = issue_credential(_PEPPER)
    hasher = CredentialHasher(_PEPPER)

    assert hasher.verify(issued.token, issued.digest) is True
    assert hasher.verify(issued.token, None) is False
    assert hasher.verify("malformed-secret-token", issued.digest) is False
    assert hasher.verify(issued.token, b"too-short") is False
    assert len(comparisons) == 4
    assert all(
        len(candidate) == len(expected) == CREDENTIAL_DIGEST_BYTES
        for candidate, expected in comparisons
    )


def test_secret_representations_and_errors_do_not_expose_token_or_pepper() -> None:
    pepper = "p" * 32
    issued = issue_credential(pepper)
    token = issued.token.get_secret_value()
    parsed = parse_credential_token(token)

    assert token not in repr(issued)
    assert repr(issued.digest) not in repr(issued)
    assert parsed.secret.get_secret_value() not in repr(parsed)
    assert pepper not in repr(CredentialHasher(pepper))
    with pytest.raises(ValueError) as exc_info:
        CredentialHasher(pepper).digest(f"{token}x")
    assert token not in str(exc_info.value)
    assert pepper not in str(exc_info.value)
