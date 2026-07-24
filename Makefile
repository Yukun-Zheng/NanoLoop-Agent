PYTHON ?= python3
VENV_DIR ?= .venv
PYTHON_BIN ?= $(VENV_DIR)/bin/python
BACKUP_ARCHIVE ?=
BACKUP_CHECKSUM ?=
BACKUP_REPORT ?=
RESTORE_ROOT ?=
IDENTITY_ARGS ?=

.DEFAULT_GOAL := help

.PHONY: help install install-models lint typecheck test frontend-install frontend frontend-check frontend-build frontend-e2e \
	openapi migration-check check mvp-smoke serve db-upgrade \
	handoff-doc handoff-doc-v3 backup-create backup-verify backup-restore backup-drill docker-build compose-config compose-up \
	compose-up-models compose-down compose-logs identity-manage rag-guide-doc \
	compose-up-local-llm-models

help:
	@echo "NanoLoop Agent development commands"
	@echo "  make install          Create .venv and install backend development dependencies"
	@echo "  make install-models   Install development dependencies plus real model runtimes"
	@echo "  make check            Run Ruff, Mypy, Pytest, and fresh Alembic checks"
	@echo "  make mvp-smoke        Run the offline engineering-fixture backend loop"
	@echo "  make serve            Run the local API with reload"
	@echo "  make frontend-install Install the locked Next.js frontend dependencies"
	@echo "  make frontend         Run the Next.js command center in development mode"
	@echo "  make frontend-check   Run frontend API drift, lint, types, tests, and build"
	@echo "  make frontend-build   Build the production Next.js frontend"
	@echo "  make frontend-e2e     Run the Playwright frontend suite"
	@echo "  make db-upgrade       Upgrade the configured database to Alembic head"
	@echo "  make handoff-doc      Rebuild the paused historical v4.0 handoff (do not distribute)"
	@echo "  make handoff-doc-v3   Rebuild the archived v3 handoff (do not distribute)"
	@echo "  make rag-guide-doc    Regenerate the RAG development guide DOCX"
	@echo "  make backup-create    Create BACKUP_ARCHIVE (offline writers only)"
	@echo "  make backup-verify    Verify BACKUP_ARCHIVE and optional BACKUP_CHECKSUM"
	@echo "  make backup-restore   Restore BACKUP_ARCHIVE into fresh RESTORE_ROOT"
	@echo "  make backup-drill     Create, verify, and restore with a limited BACKUP_REPORT"
	@echo "  make identity-manage  Run the operator identity CLI with IDENTITY_ARGS"
	@echo "  make docker-build     Build the CPU API image"
	@echo "  make compose-up       Start the hardened local container stack"
	@echo "  make compose-up-models Build and start the stack with model runtimes"
	@echo "  make compose-up-local-llm-models Start models stack with host Ollama Qwen3"
	@echo "  make compose-down     Stop the local container stack"

$(PYTHON_BIN):
	$(PYTHON) -m venv $(VENV_DIR)

install: $(PYTHON_BIN)
	$(PYTHON_BIN) -m pip install --upgrade pip
	$(PYTHON_BIN) -m pip install -e '.[dev,analysis,docs]'

install-models: $(PYTHON_BIN)
	$(PYTHON_BIN) -m pip install --upgrade pip
	@if [ "$$(uname -s)" = "Linux" ]; then \
		$(PYTHON_BIN) -m pip install \
			--index-url https://download.pytorch.org/whl/cpu \
			--no-deps \
			--requirement docker-models-cpu-constraints.txt; \
	else \
		$(PYTHON_BIN) -m pip install \
			--no-deps \
			--requirement docker-models-cpu-constraints.txt; \
	fi
	$(PYTHON_BIN) -m pip install \
		--constraint docker-models-cpu-constraints.txt \
		-e '.[dev,analysis,docs,models]'

lint:
	$(PYTHON_BIN) -m ruff check .

typecheck:
	$(PYTHON_BIN) -m mypy app

test:
	$(PYTHON_BIN) -m pytest -q

frontend-install:
	corepack enable && cd frontend && pnpm install --frozen-lockfile

frontend:
	cd frontend && pnpm dev

frontend-check:
	cd frontend && pnpm check

frontend-build:
	cd frontend && pnpm build

frontend-e2e:
	cd frontend && pnpm test:e2e

openapi:
	$(PYTHON_BIN) scripts/generate_openapi.py

migration-check:
	PYTHON_BIN=$(PYTHON_BIN) ./scripts/check_migrations.sh

check:
	PYTHON_BIN=$(PYTHON_BIN) ./scripts/verify.sh

mvp-smoke:
	$(PYTHON_BIN) scripts/mvp_fixture_smoke.py

serve:
	$(PYTHON_BIN) -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload --no-proxy-headers

db-upgrade:
	$(PYTHON_BIN) -m alembic -c alembic.ini upgrade head

handoff-doc:
	$(PYTHON_BIN) -m scripts.build_v4_handoff_doc

handoff-doc-v3:
	$(PYTHON_BIN) scripts/build_v3_handoff_doc.py

rag-guide-doc:
	$(PYTHON_BIN) scripts/build_rag_guide_doc.py

backup-create:
	@test -n "$(BACKUP_ARCHIVE)" || { echo "BACKUP_ARCHIVE is required" >&2; exit 2; }
	$(PYTHON_BIN) scripts/backup_restore.py create "$(BACKUP_ARCHIVE)" --offline-confirmed

backup-verify:
	@test -n "$(BACKUP_ARCHIVE)" || { echo "BACKUP_ARCHIVE is required" >&2; exit 2; }
	$(PYTHON_BIN) scripts/backup_restore.py verify "$(BACKUP_ARCHIVE)" $(if $(BACKUP_CHECKSUM),--checksum-path "$(BACKUP_CHECKSUM)")

backup-restore:
	@test -n "$(BACKUP_ARCHIVE)" || { echo "BACKUP_ARCHIVE is required" >&2; exit 2; }
	@test -n "$(RESTORE_ROOT)" || { echo "RESTORE_ROOT is required" >&2; exit 2; }
	$(PYTHON_BIN) scripts/backup_restore.py restore "$(BACKUP_ARCHIVE)" "$(RESTORE_ROOT)" --offline-confirmed $(if $(BACKUP_CHECKSUM),--checksum-path "$(BACKUP_CHECKSUM)")

backup-drill:
	@test -n "$(BACKUP_ARCHIVE)" || { echo "BACKUP_ARCHIVE is required" >&2; exit 2; }
	@test -n "$(RESTORE_ROOT)" || { echo "RESTORE_ROOT is required" >&2; exit 2; }
	@test -n "$(BACKUP_REPORT)" || { echo "BACKUP_REPORT is required" >&2; exit 2; }
	$(PYTHON_BIN) scripts/backup_restore.py drill "$(BACKUP_ARCHIVE)" "$(RESTORE_ROOT)" "$(BACKUP_REPORT)" --offline-confirmed

identity-manage:
	@test -n "$(IDENTITY_ARGS)" || { echo "IDENTITY_ARGS is required" >&2; exit 2; }
	$(PYTHON_BIN) scripts/manage_identity.py $(IDENTITY_ARGS)

docker-build:
	docker build --tag nanoloop-agent:local .

compose-config:
	docker compose config --quiet

compose-up:
	docker compose up --build --detach

compose-up-models:
	COMPOSE_PARALLEL_LIMIT=1 NANOLOOP_API_EXTRAS=models docker compose build api
	COMPOSE_PARALLEL_LIMIT=1 docker compose build frontend
	NANOLOOP_API_EXTRAS=models docker compose up --detach --no-build

compose-up-local-llm-models:
	@test -n "$(LLM_MODEL)" || { echo "LLM_MODEL is required" >&2; exit 2; }
	LLM_MODEL="$(LLM_MODEL)" $(PYTHON) scripts/check_local_llm.py
	COMPOSE_PARALLEL_LIMIT=1 NANOLOOP_API_EXTRAS=models docker compose build api
	COMPOSE_PARALLEL_LIMIT=1 docker compose build frontend
	LLM_MODEL="$(LLM_MODEL)" NANOLOOP_API_EXTRAS=models docker compose \
		-f docker-compose.yml -f docker-compose.ollama.yml up --detach --no-build

compose-down:
	docker compose down

compose-logs:
	docker compose logs --follow api frontend
