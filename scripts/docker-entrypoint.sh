#!/bin/sh
set -eu

umask 027

log_dir="${LOG_DIR:-/app/logs}"
model_snapshot_root="${MODEL_SNAPSHOT_ROOT:-/app/data/model-snapshots}"
mkdir -p \
    /app/data/tmp \
    "${model_snapshot_root}" \
    /app/outputs \
    "${log_dir}" \
    /app/knowledge_base/sources \
    /app/knowledge_base/index

for writable_dir in \
    /app/data \
    /app/data/tmp \
    "${model_snapshot_root}" \
    /app/outputs \
    "${log_dir}" \
    /app/knowledge_base/sources \
    /app/knowledge_base/index; do
    if [ ! -w "${writable_dir}" ]; then
        echo "NanoLoop runtime directory is not writable: ${writable_dir}" >&2
        exit 1
    fi
done

# Persist a generated signing secret with the data volume when operators do not inject one. The
# value is never printed and remains stable across single-container restarts.
if [ -z "${NANOLOOP_FILE_TOKEN_SECRET:-}" ]; then
    token_secret_file=${NANOLOOP_FILE_TOKEN_SECRET_FILE:-/app/data/.file_token_secret}
    if [ ! -s "${token_secret_file}" ]; then
        token_secret_temp="${token_secret_file}.tmp.$$"
        umask 077
        python -c 'import secrets; print(secrets.token_urlsafe(48))' >"${token_secret_temp}"
        chmod 0600 "${token_secret_temp}"
        mv "${token_secret_temp}" "${token_secret_file}"
        umask 027
    fi
    NANOLOOP_FILE_TOKEN_SECRET=$(sed -n '1p' "${token_secret_file}")
    export NANOLOOP_FILE_TOKEN_SECRET
fi
if [ "${#NANOLOOP_FILE_TOKEN_SECRET}" -lt 32 ]; then
    echo "NANOLOOP_FILE_TOKEN_SECRET must contain at least 32 characters." >&2
    exit 1
fi

if [ "${NANOLOOP_SKIP_MIGRATIONS:-0}" != "1" ]; then
    migration_log="${log_dir}/migrations.log"
    if {
        echo "$(date -u +'%Y-%m-%dT%H:%M:%SZ') starting Alembic upgrade"
        alembic -c /app/alembic.ini upgrade head
        echo "$(date -u +'%Y-%m-%dT%H:%M:%SZ') Alembic upgrade complete"
    } >>"${migration_log}" 2>&1; then
        echo "Database migrations are at head." >&2
    else
        echo "Database migration failed; recent migration output follows." >&2
        tail -n 200 "${migration_log}" >&2 || true
        exit 1
    fi
fi

if [ "$#" -eq 0 ]; then
    set -- uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 1
fi

exec "$@"
