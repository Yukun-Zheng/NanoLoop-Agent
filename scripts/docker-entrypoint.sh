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

# Production never lets the application invent process-local file-token v2 signing material.
# Load the protected persisted key ring, or initialize it once through the audited Python store
# when the path is genuinely absent. Existing unsafe or corrupt state fails closed, and neither
# the path nor any key material is written to startup logs.
if [ "${APP_ENV:-development}" = "production" ]; then
    file_token_v2_keyring_path=${FILE_TOKEN_V2_KEYRING_PATH:-${NANOLOOP_FILE_TOKEN_V2_KEYRING_PATH:-/app/data/.file_token_v2_keyring.json}}
    FILE_TOKEN_V2_KEYRING_PATH=${file_token_v2_keyring_path}
    export FILE_TOKEN_V2_KEYRING_PATH
    if ! python - "${FILE_TOKEN_V2_KEYRING_PATH}" <<'PY'
import sys

from app.storage.file_token_keyring_store import (
    FileTokenV2KeyRingStore,
    FileTokenV2KeyRingStoreError,
)

try:
    store = FileTokenV2KeyRingStore(sys.argv[1])
    try:
        store.load()
    except FileTokenV2KeyRingStoreError as error:
        if error.code != "missing":
            raise
        store.initialize(active_kid="initial")
except FileTokenV2KeyRingStoreError as error:
    print(f"File-token v2 key ring unavailable: {error.code}", file=sys.stderr)
    raise SystemExit(1) from None
except Exception:
    print("File-token v2 key ring unavailable: unexpected_error", file=sys.stderr)
    raise SystemExit(1) from None
PY
    then
        exit 1
    fi
fi

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
    set -- uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 1 --no-proxy-headers
fi

exec "$@"
