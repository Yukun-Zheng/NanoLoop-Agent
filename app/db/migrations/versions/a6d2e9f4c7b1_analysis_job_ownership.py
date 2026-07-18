"""analysis job tenant ownership

Revision ID: a6d2e9f4c7b1
Revises: f5c1d8a4b2e9
Create Date: 2026-07-18
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "a6d2e9f4c7b1"
down_revision: str | None = "f5c1d8a4b2e9"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_LEGACY_TENANT_ID = f"tnt_{'0' * 32}"
_LEGACY_PRINCIPAL_ID = f"prn_{'0' * 32}"


def upgrade() -> None:
    _assert_legacy_identity_ready()

    # SQLite requires the referenced column tuple to be explicitly unique even though
    # principal_id alone is already the primary key. The pair lets the child FK prove that the
    # selected owner belongs to the selected tenant rather than validating both IDs separately.
    with op.batch_alter_table("principals", recreate="always") as batch_op:
        batch_op.create_unique_constraint(
            "uq_principals_principal_tenant",
            ["principal_id", "tenant_id"],
        )

    # Add nullable first so existing v2 jobs survive the schema transition, then backfill every
    # row to the fixed compatibility identity created by the preceding identity migration.
    op.add_column(
        "analysis_jobs",
        sa.Column("tenant_id", sa.String(length=36), nullable=True),
    )
    op.add_column(
        "analysis_jobs",
        sa.Column("owner_principal_id", sa.String(length=36), nullable=True),
    )
    op.execute(
        sa.text(
            """
            UPDATE analysis_jobs
            SET tenant_id = :tenant_id,
                owner_principal_id = :owner_principal_id
            WHERE tenant_id IS NULL OR owner_principal_id IS NULL
            """
        ).bindparams(
            tenant_id=_LEGACY_TENANT_ID,
            owner_principal_id=_LEGACY_PRINCIPAL_ID,
        )
    )
    missing_ownership = int(
        op.get_bind().scalar(
            sa.text(
                """
                SELECT count(*) FROM analysis_jobs
                WHERE tenant_id IS NULL OR owner_principal_id IS NULL
                """
            )
        )
        or 0
    )
    if missing_ownership:
        raise RuntimeError("analysis job ownership backfill was incomplete")

    with op.batch_alter_table("analysis_jobs", recreate="always") as batch_op:
        batch_op.alter_column(
            "tenant_id",
            existing_type=sa.String(length=36),
            nullable=False,
        )
        batch_op.alter_column(
            "owner_principal_id",
            existing_type=sa.String(length=36),
            nullable=False,
        )
        batch_op.create_foreign_key(
            "fk_analysis_jobs_tenant_id_tenants",
            "tenants",
            ["tenant_id"],
            ["tenant_id"],
            ondelete="RESTRICT",
        )
        batch_op.create_foreign_key(
            "fk_analysis_jobs_owner_principal_tenant",
            "principals",
            ["owner_principal_id", "tenant_id"],
            ["principal_id", "tenant_id"],
            ondelete="RESTRICT",
        )

    op.create_index(
        "ix_analysis_jobs_tenant_created",
        "analysis_jobs",
        ["tenant_id", "created_at"],
        unique=False,
    )
    op.create_index(
        "ix_analysis_jobs_tenant_owner_created",
        "analysis_jobs",
        ["tenant_id", "owner_principal_id", "created_at"],
        unique=False,
    )
    _assert_sqlite_foreign_keys_clean()


def downgrade() -> None:
    _assert_only_legacy_ownership()

    op.drop_index(
        "ix_analysis_jobs_tenant_owner_created",
        table_name="analysis_jobs",
    )
    op.drop_index(
        "ix_analysis_jobs_tenant_created",
        table_name="analysis_jobs",
    )
    with op.batch_alter_table("analysis_jobs", recreate="always") as batch_op:
        batch_op.drop_constraint(
            "fk_analysis_jobs_owner_principal_tenant",
            type_="foreignkey",
        )
        batch_op.drop_constraint(
            "fk_analysis_jobs_tenant_id_tenants",
            type_="foreignkey",
        )
        batch_op.drop_column("owner_principal_id")
        batch_op.drop_column("tenant_id")

    with op.batch_alter_table("principals", recreate="always") as batch_op:
        batch_op.drop_constraint(
            "uq_principals_principal_tenant",
            type_="unique",
        )


def _assert_legacy_identity_ready() -> None:
    """Fail before schema mutation unless the fixed backfill identity is intact."""

    bind = op.get_bind()
    tenant_count = int(
        bind.scalar(
            sa.text("SELECT count(*) FROM tenants WHERE tenant_id = :tenant_id").bindparams(
                tenant_id=_LEGACY_TENANT_ID
            )
        )
        or 0
    )
    principal = bind.execute(
        sa.text(
            """
            SELECT tenant_id, kind, role
            FROM principals
            WHERE principal_id = :principal_id
            """
        ).bindparams(principal_id=_LEGACY_PRINCIPAL_ID)
    ).one_or_none()
    if tenant_count != 1 or principal is None:
        raise RuntimeError("fixed legacy ownership identity is missing")
    if (
        principal.tenant_id != _LEGACY_TENANT_ID
        or principal.kind != "service"
        or principal.role != "tenant_admin"
    ):
        raise RuntimeError("fixed legacy ownership identity is invalid")


def _assert_only_legacy_ownership() -> None:
    """Refuse a lossy downgrade before dropping any ownership schema."""

    nonlegacy_count = int(
        op.get_bind().scalar(
            sa.text(
                """
                SELECT count(*)
                FROM analysis_jobs
                WHERE tenant_id IS NULL
                   OR owner_principal_id IS NULL
                   OR tenant_id != :tenant_id
                   OR owner_principal_id != :principal_id
                """
            ).bindparams(
                tenant_id=_LEGACY_TENANT_ID,
                principal_id=_LEGACY_PRINCIPAL_ID,
            )
        )
        or 0
    )
    if nonlegacy_count:
        raise RuntimeError(
            "analysis ownership downgrade refused because non-legacy ownership exists"
        )


def _assert_sqlite_foreign_keys_clean() -> None:
    bind = op.get_bind()
    if bind.dialect.name != "sqlite":
        return
    violations = bind.exec_driver_sql("PRAGMA foreign_key_check").fetchall()
    if violations:
        raise RuntimeError("SQLite foreign key check failed after ownership migration")
