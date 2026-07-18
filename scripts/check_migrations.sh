#!/bin/sh
set -eu

script_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
repo_root=$(CDPATH= cd -- "${script_dir}/.." && pwd)
python_bin=${PYTHON_BIN:-python3}

if [ ! -x "${python_bin}" ] && ! command -v "${python_bin}" >/dev/null 2>&1; then
    echo "Python executable not found: ${python_bin}" >&2
    exit 1
fi

migration_tmp_dir=$(mktemp -d "${TMPDIR:-/tmp}/nanoloop-migrations.XXXXXX")
cleanup() {
    case "${migration_tmp_dir}" in
        "${TMPDIR:-/tmp}"/nanoloop-migrations.*)
            rm -rf -- "${migration_tmp_dir}"
            ;;
        *)
            echo "Refusing to remove unexpected temporary path: ${migration_tmp_dir}" >&2
            ;;
    esac
}
trap cleanup EXIT
trap 'exit 129' HUP
trap 'exit 130' INT
trap 'exit 143' TERM

cd "${repo_root}"
export APP_ENV=test
migration_db="${migration_tmp_dir}/nanoloop-ci.db"
export DATABASE_URL="sqlite:///${migration_db}"

head_count=$("${python_bin}" -m alembic -c alembic.ini heads | awk 'NF {count++} END {print count+0}')
if [ "${head_count}" -ne 1 ]; then
    echo "Expected exactly one Alembic head, found ${head_count}." >&2
    exit 1
fi

"${python_bin}" -m alembic -c alembic.ini upgrade head
"${python_bin}" -m alembic -c alembic.ini downgrade base
"${python_bin}" -m alembic -c alembic.ini upgrade head

# SQLite FTS5 virtual tables and triggers are intentionally managed as raw migration SQL rather
# than SQLAlchemy metadata. Remove only those known external objects from this disposable database
# before asking Alembic to detect ORM schema drift.
"${python_bin}" - "${migration_db}" <<'PY'
import sqlite3
import sys

database_path = sys.argv[1]
with sqlite3.connect(database_path) as connection:
    for trigger in (
        "knowledge_chunks_fts_insert",
        "knowledge_chunks_fts_delete",
        "knowledge_chunks_fts_update",
    ):
        connection.execute(f'DROP TRIGGER IF EXISTS "{trigger}"')
    connection.execute('DROP TABLE IF EXISTS "knowledge_chunks_fts"')
PY

"${python_bin}" -m alembic -c alembic.ini check

echo "Alembic upgrade/downgrade round trip and metadata drift checks passed."
