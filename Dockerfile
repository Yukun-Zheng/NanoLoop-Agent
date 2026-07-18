# syntax=docker/dockerfile:1

ARG PYTHON_VERSION=3.12
ARG API_EXTRAS=""

FROM python:${PYTHON_VERSION}-slim-bookworm AS builder
ARG API_EXTRAS

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /build

COPY pyproject.toml README.md LICENSES.md ./
COPY app ./app

# Build a complete wheelhouse so the runtime image never invokes a compiler or package index.
# The default stays lightweight; operators may explicitly build with API_EXTRAS=rag,
# API_EXTRAS=models, or API_EXTRAS=rag,models once those external assets are mounted.
RUN set -eu; \
    case "${API_EXTRAS}" in \
        ""|rag|models|rag,models|models,rag) ;; \
        *) echo "Unsupported API_EXTRAS: ${API_EXTRAS}" >&2; exit 2 ;; \
    esac; \
    project_spec="."; \
    if [ -n "${API_EXTRAS}" ]; then project_spec=".[${API_EXTRAS}]"; fi; \
    python -m pip wheel --wheel-dir /wheels "${project_spec}"


FROM python:${PYTHON_VERSION}-slim-bookworm AS runtime

ARG APP_UID=10001
ARG APP_GID=10001
ARG API_EXTRAS

LABEL org.opencontainers.image.title="NanoLoop Agent API" \
      org.opencontainers.image.description="CPU API runtime without model weights or RAG corpus" \
      org.opencontainers.image.licenses="LicenseRef-Proprietary"

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    APP_ENV=production \
    DATABASE_URL=sqlite:////app/data/nanoloop.db \
    OUTPUT_ROOT=/app/outputs \
    FILE_TOKEN_V2_KEYRING_PATH=/app/data/.file_token_v2_keyring.json \
    MODEL_REGISTRY_PATH=/app/model_artifacts/registry.yaml \
    MODEL_SNAPSHOT_ROOT=/app/data/model-snapshots \
    MODEL_DEVICE=cpu \
    KNOWLEDGE_SOURCE_DIR=/app/knowledge_base/sources \
    FAISS_INDEX_PATH=/app/knowledge_base/index/faiss.index \
    LLM_PROVIDER=extractive \
    LOG_DIR=/app/logs \
    LOG_LEVEL=INFO \
    TMPDIR=/app/data/tmp

RUN groupadd --gid "${APP_GID}" nanoloop \
    && useradd --uid "${APP_UID}" --gid "${APP_GID}" \
        --create-home --home-dir /home/nanoloop --shell /usr/sbin/nologin nanoloop \
    && mkdir -p \
        /app/app/db/migrations \
        /app/data/tmp \
        /app/data/model-snapshots \
        /app/logs \
        /app/outputs \
        /app/model_artifacts/weights \
        /app/knowledge_base/sources \
        /app/knowledge_base/index \
    && chown -R nanoloop:nanoloop /app /home/nanoloop

WORKDIR /app

COPY --from=builder /wheels /wheels
RUN set -eu; \
    package_spec="nanoloop-agent"; \
    if [ -n "${API_EXTRAS}" ]; then package_spec="nanoloop-agent[${API_EXTRAS}]"; fi; \
    python -m pip install --no-index --find-links=/wheels "${package_spec}" \
    && rm -rf /wheels

# Alembic's script_location is repository-relative, while application code comes from the wheel.
COPY --chown=nanoloop:nanoloop alembic.ini ./alembic.ini
COPY --chown=nanoloop:nanoloop app/db/migrations ./app/db/migrations
COPY --chown=nanoloop:nanoloop model_artifacts ./model_artifacts
COPY --chown=nanoloop:nanoloop LICENSES.md ./LICENSES.md
COPY --chown=nanoloop:nanoloop scripts/backup_restore.py ./scripts/backup_restore.py
COPY --chown=nanoloop:nanoloop scripts/manage_file_token_keyring.py ./scripts/manage_file_token_keyring.py
COPY --chown=nanoloop:nanoloop scripts/docker-entrypoint.sh /usr/local/bin/nanoloop-entrypoint

RUN chmod 0555 \
        /usr/local/bin/nanoloop-entrypoint \
        /app/scripts/backup_restore.py \
        /app/scripts/manage_file_token_keyring.py \
    && chown -R nanoloop:nanoloop /app/model_artifacts /app/knowledge_base

USER nanoloop:nanoloop

EXPOSE 8000
STOPSIGNAL SIGTERM

HEALTHCHECK --interval=15s --timeout=5s --start-period=20s --retries=5 \
    CMD ["python", "-c", "import json,urllib.request; p=json.load(urllib.request.urlopen('http://127.0.0.1:8000/health',timeout=3)); assert p['status']=='success' and p['data']['service']['status']=='healthy' and p['data']['database']['status']=='healthy'"]

ENTRYPOINT ["/usr/local/bin/nanoloop-entrypoint"]
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1", "--timeout-keep-alive", "5", "--no-proxy-headers"]
