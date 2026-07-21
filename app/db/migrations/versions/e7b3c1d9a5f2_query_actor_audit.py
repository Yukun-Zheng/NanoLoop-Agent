"""query actor audit

Revision ID: e7b3c1d9a5f2
Revises: c9a4e7b2d6f1
Create Date: 2026-07-18
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "e7b3c1d9a5f2"
down_revision: str | None = "c9a4e7b2d6f1"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_LEGACY_TENANT_ID = f"tnt_{'0' * 32}"
_LEGACY_PRINCIPAL_ID = f"prn_{'0' * 32}"


def upgrade() -> None:
    _assert_sqlite_foreign_keys_clean(boundary="before")
    _assert_legacy_actor_ready()
    _assert_historical_queries_are_legacy_owned()

    with op.batch_alter_table("analysis_jobs", recreate="always") as batch_op:
        batch_op.create_unique_constraint(
            "uq_analysis_jobs_job_tenant",
            ["job_id", "tenant_id"],
        )

    with op.batch_alter_table("api_credentials", recreate="always") as batch_op:
        batch_op.create_unique_constraint(
            "uq_api_credentials_credential_principal",
            ["credential_id", "principal_id"],
        )

    op.add_column(
        "query_logs",
        sa.Column("actor_tenant_id", sa.String(length=36), nullable=True),
    )
    op.add_column(
        "query_logs",
        sa.Column("actor_principal_id", sa.String(length=36), nullable=True),
    )
    op.add_column(
        "query_logs",
        sa.Column("actor_credential_id", sa.String(length=36), nullable=True),
    )
    op.add_column(
        "query_logs",
        sa.Column("actor_role", sa.String(length=32), nullable=True),
    )
    op.add_column(
        "query_logs",
        sa.Column("actor_auth_mode", sa.String(length=24), nullable=True),
    )
    op.execute(
        sa.text(
            """
            UPDATE query_logs
            SET actor_tenant_id = :tenant_id,
                actor_principal_id = :principal_id,
                actor_credential_id = NULL,
                actor_role = 'tenant_admin',
                actor_auth_mode = 'legacy_unknown'
            """
        ).bindparams(
            tenant_id=_LEGACY_TENANT_ID,
            principal_id=_LEGACY_PRINCIPAL_ID,
        )
    )
    _assert_actor_backfill_complete()

    with op.batch_alter_table("query_logs", recreate="always") as batch_op:
        batch_op.alter_column(
            "actor_tenant_id",
            existing_type=sa.String(length=36),
            nullable=False,
        )
        batch_op.alter_column(
            "actor_principal_id",
            existing_type=sa.String(length=36),
            nullable=False,
        )
        batch_op.alter_column(
            "actor_role",
            existing_type=sa.String(length=32),
            nullable=False,
        )
        batch_op.alter_column(
            "actor_auth_mode",
            existing_type=sa.String(length=24),
            nullable=False,
        )
        batch_op.drop_constraint(
            "fk_query_logs_job_id_analysis_jobs",
            type_="foreignkey",
        )
        batch_op.create_foreign_key(
            "fk_query_logs_job_actor_tenant",
            "analysis_jobs",
            ["job_id", "actor_tenant_id"],
            ["job_id", "tenant_id"],
            ondelete="CASCADE",
        )
        batch_op.create_foreign_key(
            "fk_query_logs_actor_principal_tenant",
            "principals",
            ["actor_principal_id", "actor_tenant_id"],
            ["principal_id", "tenant_id"],
            ondelete="RESTRICT",
        )
        batch_op.create_foreign_key(
            "fk_query_logs_actor_credential_principal",
            "api_credentials",
            ["actor_credential_id", "actor_principal_id"],
            ["credential_id", "principal_id"],
            ondelete="RESTRICT",
        )
        batch_op.create_check_constraint(
            "ck_query_logs_actor_role_known",
            "actor_role IN ('tenant_admin', 'analyst', 'viewer')",
        )
        batch_op.create_check_constraint(
            "ck_query_logs_actor_auth_mode_known",
            "actor_auth_mode IN "
            "('disabled', 'shared_key', 'principal', 'legacy_unknown')",
        )
        batch_op.create_check_constraint(
            "ck_query_logs_actor_credential_shape",
            "(actor_auth_mode = 'principal' AND actor_credential_id IS NOT NULL) OR "
            "(actor_auth_mode IN ('disabled', 'shared_key', 'legacy_unknown') "
            "AND actor_credential_id IS NULL)",
        )
        batch_op.create_check_constraint(
            "ck_query_logs_compatibility_actor_shape",
            "actor_auth_mode = 'principal' OR "
            f"(actor_tenant_id = '{_LEGACY_TENANT_ID}' "
            f"AND actor_principal_id = '{_LEGACY_PRINCIPAL_ID}' "
            "AND actor_role = 'tenant_admin')",
        )

    op.create_index(
        "ix_query_logs_actor_created",
        "query_logs",
        ["actor_tenant_id", "actor_principal_id", "created_at"],
        unique=False,
    )
    _assert_sqlite_foreign_keys_clean(boundary="after")


def downgrade() -> None:
    _assert_sqlite_foreign_keys_clean(boundary="before")
    _assert_only_legacy_unknown_queries()

    op.drop_index("ix_query_logs_actor_created", table_name="query_logs")
    with op.batch_alter_table("query_logs", recreate="always") as batch_op:
        batch_op.drop_constraint(
            "fk_query_logs_actor_credential_principal",
            type_="foreignkey",
        )
        batch_op.drop_constraint(
            "fk_query_logs_actor_principal_tenant",
            type_="foreignkey",
        )
        batch_op.drop_constraint(
            "fk_query_logs_job_actor_tenant",
            type_="foreignkey",
        )
        batch_op.drop_constraint(
            "ck_query_logs_compatibility_actor_shape",
            type_="check",
        )
        batch_op.drop_constraint(
            "ck_query_logs_actor_credential_shape",
            type_="check",
        )
        batch_op.drop_constraint(
            "ck_query_logs_actor_auth_mode_known",
            type_="check",
        )
        batch_op.drop_constraint(
            "ck_query_logs_actor_role_known",
            type_="check",
        )
        batch_op.create_foreign_key(
            "fk_query_logs_job_id_analysis_jobs",
            "analysis_jobs",
            ["job_id"],
            ["job_id"],
            ondelete="CASCADE",
        )
        batch_op.drop_column("actor_auth_mode")
        batch_op.drop_column("actor_role")
        batch_op.drop_column("actor_credential_id")
        batch_op.drop_column("actor_principal_id")
        batch_op.drop_column("actor_tenant_id")

    with op.batch_alter_table("api_credentials", recreate="always") as batch_op:
        batch_op.drop_constraint(
            "uq_api_credentials_credential_principal",
            type_="unique",
        )

    with op.batch_alter_table("analysis_jobs", recreate="always") as batch_op:
        batch_op.drop_constraint(
            "uq_analysis_jobs_job_tenant",
            type_="unique",
        )

    _assert_sqlite_foreign_keys_clean(boundary="after")


def _assert_legacy_actor_ready() -> None:
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
        raise RuntimeError("fixed legacy query actor identity is missing")
    if (
        principal.tenant_id != _LEGACY_TENANT_ID
        or principal.kind != "service"
        or principal.role != "tenant_admin"
    ):
        raise RuntimeError("fixed legacy query actor identity is invalid")


def _assert_historical_queries_are_legacy_owned() -> None:
    invalid_count = int(
        op.get_bind().scalar(
            sa.text(
                """
                SELECT count(*)
                FROM query_logs AS query
                LEFT JOIN analysis_jobs AS job ON job.job_id = query.job_id
                WHERE job.job_id IS NULL OR job.tenant_id != :tenant_id
                """
            ).bindparams(tenant_id=_LEGACY_TENANT_ID)
        )
        or 0
    )
    if invalid_count:
        raise RuntimeError(
            "query actor migration refused: historical query actor cannot be reconstructed"
        )


def _assert_actor_backfill_complete() -> None:
    missing_count = int(
        op.get_bind().scalar(
            sa.text(
                """
                SELECT count(*)
                FROM query_logs
                WHERE actor_tenant_id IS NULL
                   OR actor_principal_id IS NULL
                   OR actor_role IS NULL
                   OR actor_auth_mode IS NULL
                """
            )
        )
        or 0
    )
    if missing_count:
        raise RuntimeError("query actor migration backfill was incomplete")


def _assert_only_legacy_unknown_queries() -> None:
    nonlegacy_count = int(
        op.get_bind().scalar(
            sa.text(
                """
                SELECT count(*)
                FROM query_logs
                WHERE actor_auth_mode IS NULL OR actor_auth_mode != 'legacy_unknown'
                """
            )
        )
        or 0
    )
    if nonlegacy_count:
        raise RuntimeError(
            "query actor downgrade refused because attributable query audit facts exist"
        )


def _assert_sqlite_foreign_keys_clean(*, boundary: str) -> None:
    bind = op.get_bind()
    if bind.dialect.name != "sqlite":
        return
    violations = bind.exec_driver_sql("PRAGMA foreign_key_check").fetchall()
    if violations:
        raise RuntimeError(
            f"SQLite foreign key check failed {boundary} query actor migration"
        )
