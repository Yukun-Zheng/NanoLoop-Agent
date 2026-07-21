"""principal identity and revocable credentials

Revision ID: f5c1d8a4b2e9
Revises: e2a7c4d8f1b3
Create Date: 2026-07-18
"""

from collections.abc import Sequence
from datetime import UTC, datetime

import sqlalchemy as sa
from alembic import op

revision: str = "f5c1d8a4b2e9"
down_revision: str | None = "e2a7c4d8f1b3"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_LEGACY_TENANT_ID = f"tnt_{'0' * 32}"
_LEGACY_PRINCIPAL_ID = f"prn_{'0' * 32}"
_HEX_32_GLOB = "[0-9a-f]" * 32


def upgrade() -> None:
    tenants = op.create_table(
        "tenants",
        sa.Column("tenant_id", sa.String(length=36), nullable=False),
        sa.Column("slug", sa.String(length=63), nullable=False),
        sa.Column("display_name", sa.String(length=255), nullable=False),
        sa.Column("enabled", sa.Boolean(), nullable=False),
        sa.Column("version", sa.Integer(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.CheckConstraint(
            "length(tenant_id) = 36",
            name=op.f("ck_tenants_tenant_id_length"),
        ),
        sa.CheckConstraint(
            f"tenant_id GLOB 'tnt_{_HEX_32_GLOB}'",
            name=op.f("ck_tenants_tenant_id_canonical"),
        ),
        sa.CheckConstraint(
            "length(slug) BETWEEN 1 AND 63",
            name=op.f("ck_tenants_slug_length"),
        ),
        sa.CheckConstraint("slug = lower(slug)", name=op.f("ck_tenants_slug_lowercase")),
        sa.CheckConstraint(
            "slug NOT GLOB '*[^a-z0-9-]*' "
            "AND substr(slug, 1, 1) GLOB '[a-z0-9]' "
            "AND substr(slug, -1, 1) GLOB '[a-z0-9]'",
            name=op.f("ck_tenants_slug_canonical"),
        ),
        sa.CheckConstraint(
            "length(trim(display_name)) BETWEEN 1 AND 255",
            name=op.f("ck_tenants_display_name_length"),
        ),
        sa.CheckConstraint(
            "enabled IN (0, 1)",
            name=op.f("ck_tenants_enabled_boolean"),
        ),
        sa.CheckConstraint("version >= 1", name=op.f("ck_tenants_version_positive")),
        sa.PrimaryKeyConstraint("tenant_id", name=op.f("pk_tenants")),
        sa.UniqueConstraint("slug", name=op.f("uq_tenants_slug")),
    )
    op.create_index("ix_tenants_enabled", "tenants", ["enabled"], unique=False)

    principals = op.create_table(
        "principals",
        sa.Column("principal_id", sa.String(length=36), nullable=False),
        sa.Column("tenant_id", sa.String(length=36), nullable=False),
        sa.Column("handle", sa.String(length=64), nullable=False),
        sa.Column("display_name", sa.String(length=255), nullable=False),
        sa.Column("kind", sa.String(length=20), nullable=False),
        sa.Column("role", sa.String(length=32), nullable=False),
        sa.Column("enabled", sa.Boolean(), nullable=False),
        sa.Column("version", sa.Integer(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.CheckConstraint(
            "length(principal_id) = 36",
            name=op.f("ck_principals_principal_id_length"),
        ),
        sa.CheckConstraint(
            f"principal_id GLOB 'prn_{_HEX_32_GLOB}'",
            name=op.f("ck_principals_principal_id_canonical"),
        ),
        sa.CheckConstraint(
            "length(handle) BETWEEN 1 AND 64",
            name=op.f("ck_principals_handle_length"),
        ),
        sa.CheckConstraint(
            "handle = lower(handle)",
            name=op.f("ck_principals_handle_lowercase"),
        ),
        sa.CheckConstraint(
            "handle NOT GLOB '*[^a-z0-9._-]*' "
            "AND substr(handle, 1, 1) GLOB '[a-z0-9]' "
            "AND substr(handle, -1, 1) GLOB '[a-z0-9]'",
            name=op.f("ck_principals_handle_canonical"),
        ),
        sa.CheckConstraint(
            "length(trim(display_name)) BETWEEN 1 AND 255",
            name=op.f("ck_principals_display_name_length"),
        ),
        sa.CheckConstraint(
            "kind IN ('user', 'service')",
            name=op.f("ck_principals_kind_known"),
        ),
        sa.CheckConstraint(
            "role IN ('tenant_admin', 'analyst', 'viewer')",
            name=op.f("ck_principals_role_known"),
        ),
        sa.CheckConstraint(
            "enabled IN (0, 1)",
            name=op.f("ck_principals_enabled_boolean"),
        ),
        sa.CheckConstraint("version >= 1", name=op.f("ck_principals_version_positive")),
        sa.ForeignKeyConstraint(
            ["tenant_id"],
            ["tenants.tenant_id"],
            name=op.f("fk_principals_tenant_id_tenants"),
            ondelete="RESTRICT",
        ),
        sa.PrimaryKeyConstraint("principal_id", name=op.f("pk_principals")),
        sa.UniqueConstraint(
            "tenant_id",
            "handle",
            name="uq_principals_tenant_handle",
        ),
    )
    op.create_index(
        "ix_principals_tenant_enabled",
        "principals",
        ["tenant_id", "enabled"],
        unique=False,
    )

    op.create_table(
        "api_credentials",
        sa.Column("credential_id", sa.String(length=36), nullable=False),
        sa.Column("principal_id", sa.String(length=36), nullable=False),
        sa.Column("label", sa.String(length=120), nullable=False),
        sa.Column("token_digest", sa.LargeBinary(length=32), nullable=False),
        sa.Column("enabled", sa.Boolean(), nullable=False),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("revoked_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("version", sa.Integer(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.CheckConstraint(
            "length(credential_id) = 36",
            name=op.f("ck_api_credentials_credential_id_length"),
        ),
        sa.CheckConstraint(
            f"credential_id GLOB 'crd_{_HEX_32_GLOB}'",
            name=op.f("ck_api_credentials_credential_id_canonical"),
        ),
        sa.CheckConstraint(
            "length(trim(label)) BETWEEN 1 AND 120",
            name=op.f("ck_api_credentials_label_length"),
        ),
        sa.CheckConstraint(
            "length(token_digest) = 32",
            name=op.f("ck_api_credentials_token_digest_32_bytes"),
        ),
        sa.CheckConstraint(
            "enabled IN (0, 1)",
            name=op.f("ck_api_credentials_enabled_boolean"),
        ),
        sa.CheckConstraint(
            "version >= 1",
            name=op.f("ck_api_credentials_version_positive"),
        ),
        sa.ForeignKeyConstraint(
            ["principal_id"],
            ["principals.principal_id"],
            name=op.f("fk_api_credentials_principal_id_principals"),
            ondelete="RESTRICT",
        ),
        sa.PrimaryKeyConstraint("credential_id", name=op.f("pk_api_credentials")),
        sa.UniqueConstraint(
            "token_digest",
            name="uq_api_credentials_token_digest",
        ),
    )
    op.create_index(
        "ix_api_credentials_principal_enabled",
        "api_credentials",
        ["principal_id", "enabled"],
        unique=False,
    )

    audit_events = op.create_table(
        "identity_audit_events",
        sa.Column("event_id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("tenant_id", sa.String(length=36), nullable=False),
        sa.Column("principal_id", sa.String(length=36), nullable=True),
        sa.Column("credential_id", sa.String(length=36), nullable=True),
        sa.Column("actor_principal_id", sa.String(length=36), nullable=True),
        sa.Column("actor_kind", sa.String(length=24), nullable=False),
        sa.Column("event_type", sa.String(length=40), nullable=False),
        sa.Column("metadata_json", sa.JSON(), nullable=False),
        sa.Column("occurred_at", sa.DateTime(timezone=True), nullable=False),
        sa.CheckConstraint(
            "event_type IN ("
            "'tenant.created', 'tenant.enabled', 'tenant.disabled', "
            "'principal.created', 'principal.enabled', 'principal.disabled', "
            "'credential.issued', 'credential.enabled', 'credential.disabled', "
            "'credential.revoked')",
            name=op.f("ck_identity_audit_events_event_type_known"),
        ),
        sa.CheckConstraint(
            "actor_kind IN ('operator_cli', 'principal', 'migration', 'system')",
            name=op.f("ck_identity_audit_events_actor_kind_known"),
        ),
        sa.CheckConstraint(
            "(actor_kind = 'principal' AND actor_principal_id IS NOT NULL) OR "
            "(actor_kind <> 'principal' AND actor_principal_id IS NULL)",
            name=op.f("ck_identity_audit_events_actor_principal_shape"),
        ),
        sa.CheckConstraint(
            "credential_id IS NULL OR principal_id IS NOT NULL",
            name=op.f("ck_identity_audit_events_credential_requires_principal"),
        ),
        sa.ForeignKeyConstraint(
            ["actor_principal_id"],
            ["principals.principal_id"],
            name=op.f("fk_identity_audit_events_actor_principal_id_principals"),
            ondelete="RESTRICT",
        ),
        sa.ForeignKeyConstraint(
            ["credential_id"],
            ["api_credentials.credential_id"],
            name=op.f("fk_identity_audit_events_credential_id_api_credentials"),
            ondelete="RESTRICT",
        ),
        sa.ForeignKeyConstraint(
            ["principal_id"],
            ["principals.principal_id"],
            name=op.f("fk_identity_audit_events_principal_id_principals"),
            ondelete="RESTRICT",
        ),
        sa.ForeignKeyConstraint(
            ["tenant_id"],
            ["tenants.tenant_id"],
            name=op.f("fk_identity_audit_events_tenant_id_tenants"),
            ondelete="RESTRICT",
        ),
        sa.PrimaryKeyConstraint("event_id", name=op.f("pk_identity_audit_events")),
    )
    op.create_index(
        "ix_identity_audit_credential_occurred",
        "identity_audit_events",
        ["credential_id", "occurred_at"],
        unique=False,
    )
    op.create_index(
        "ix_identity_audit_principal_occurred",
        "identity_audit_events",
        ["principal_id", "occurred_at"],
        unique=False,
    )
    op.create_index(
        "ix_identity_audit_tenant_occurred",
        "identity_audit_events",
        ["tenant_id", "occurred_at"],
        unique=False,
    )
    if op.get_bind().dialect.name == "sqlite":
        op.execute(
            sa.text(
                """
                CREATE TRIGGER trg_identity_audit_events_no_update
                BEFORE UPDATE ON identity_audit_events
                BEGIN
                    SELECT RAISE(ABORT, 'identity audit events are append-only');
                END
                """
            )
        )
        op.execute(
            sa.text(
                """
                CREATE TRIGGER trg_identity_audit_events_no_delete
                BEFORE DELETE ON identity_audit_events
                BEGIN
                    SELECT RAISE(ABORT, 'identity audit events are append-only');
                END
                """
            )
        )

    bootstrap_at = datetime.now(UTC)
    op.bulk_insert(
        tenants,
        [
            {
                "tenant_id": _LEGACY_TENANT_ID,
                "slug": "legacy-local",
                "display_name": "Legacy local tenant",
                "enabled": True,
                "version": 1,
                "created_at": bootstrap_at,
                "updated_at": bootstrap_at,
            }
        ],
    )
    op.bulk_insert(
        principals,
        [
            {
                "principal_id": _LEGACY_PRINCIPAL_ID,
                "tenant_id": _LEGACY_TENANT_ID,
                "handle": "legacy-local",
                "display_name": "Legacy local service",
                "kind": "service",
                "role": "tenant_admin",
                "enabled": True,
                "version": 1,
                "created_at": bootstrap_at,
                "updated_at": bootstrap_at,
            }
        ],
    )
    op.bulk_insert(
        audit_events,
        [
            {
                "tenant_id": _LEGACY_TENANT_ID,
                "principal_id": None,
                "credential_id": None,
                "actor_principal_id": None,
                "actor_kind": "migration",
                "event_type": "tenant.created",
                "metadata_json": {"bootstrap": "legacy_local"},
                "occurred_at": bootstrap_at,
            },
            {
                "tenant_id": _LEGACY_TENANT_ID,
                "principal_id": _LEGACY_PRINCIPAL_ID,
                "credential_id": None,
                "actor_principal_id": None,
                "actor_kind": "migration",
                "event_type": "principal.created",
                "metadata_json": {"bootstrap": "legacy_local"},
                "occurred_at": bootstrap_at,
            },
        ],
    )


def downgrade() -> None:
    if op.get_bind().dialect.name == "sqlite":
        op.execute(sa.text("DROP TRIGGER IF EXISTS trg_identity_audit_events_no_delete"))
        op.execute(sa.text("DROP TRIGGER IF EXISTS trg_identity_audit_events_no_update"))
    op.drop_index(
        "ix_identity_audit_tenant_occurred",
        table_name="identity_audit_events",
    )
    op.drop_index(
        "ix_identity_audit_principal_occurred",
        table_name="identity_audit_events",
    )
    op.drop_index(
        "ix_identity_audit_credential_occurred",
        table_name="identity_audit_events",
    )
    op.drop_table("identity_audit_events")
    op.drop_index(
        "ix_api_credentials_principal_enabled",
        table_name="api_credentials",
    )
    op.drop_table("api_credentials")
    op.drop_index("ix_principals_tenant_enabled", table_name="principals")
    op.drop_table("principals")
    op.drop_index("ix_tenants_enabled", table_name="tenants")
    op.drop_table("tenants")
