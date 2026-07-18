PYTHON ?= python3
VENV_DIR ?= .venv
PYTHON_BIN ?= $(VENV_DIR)/bin/python
BACKUP_ARCHIVE ?=
BACKUP_CHECKSUM ?=
RESTORE_ROOT ?=

.DEFAULT_GOAL := help

.PHONY: help install lint typecheck test frontend-check openapi migration-check check serve frontend db-upgrade \
	handoff-doc backup-create backup-verify backup-restore docker-build compose-config compose-up \
	compose-down compose-logs

help:
	@echo "NanoLoop Agent development commands"
	@echo "  make install          Create .venv and install base + dev dependencies"
	@echo "  make check            Run Ruff, Mypy, Pytest, and fresh Alembic checks"
	@echo "  make serve            Run the local API with reload"
	@echo "  make frontend         Run the Streamlit workbench"
	@echo "  make db-upgrade       Upgrade the configured database to Alembic head"
	@echo "  make handoff-doc      Regenerate the v3 developer handoff DOCX"
	@echo "  make backup-create    Create BACKUP_ARCHIVE (offline writers only)"
	@echo "  make backup-verify    Verify BACKUP_ARCHIVE and optional BACKUP_CHECKSUM"
	@echo "  make backup-restore   Restore BACKUP_ARCHIVE into fresh RESTORE_ROOT"
	@echo "  make docker-build     Build the CPU API image"
	@echo "  make compose-up       Start the hardened local container stack"
	@echo "  make compose-down     Stop the local container stack"

$(PYTHON_BIN):
	$(PYTHON) -m venv $(VENV_DIR)

install: $(PYTHON_BIN)
	$(PYTHON_BIN) -m pip install --upgrade pip
	$(PYTHON_BIN) -m pip install -e '.[dev,analysis,frontend,docs]'

lint:
	$(PYTHON_BIN) -m ruff check .

typecheck:
	$(PYTHON_BIN) -m mypy app frontend

test:
	$(PYTHON_BIN) -m pytest -q

frontend-check:
	$(PYTHON_BIN) scripts/check_frontend.py

openapi:
	$(PYTHON_BIN) scripts/generate_openapi.py

migration-check:
	PYTHON_BIN=$(PYTHON_BIN) ./scripts/check_migrations.sh

check:
	PYTHON_BIN=$(PYTHON_BIN) ./scripts/verify.sh

serve:
	$(PYTHON_BIN) -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload

frontend:
	NANOLOOP_API_BASE_URL=http://127.0.0.1:8000 $(PYTHON_BIN) -m streamlit run frontend/app.py

db-upgrade:
	$(PYTHON_BIN) -m alembic -c alembic.ini upgrade head

handoff-doc:
	$(PYTHON_BIN) scripts/build_v3_handoff_doc.py

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

docker-build:
	docker build --tag nanoloop-agent:local .

compose-config:
	docker compose config --quiet

compose-up:
	docker compose up --build --detach

compose-down:
	docker compose down

compose-logs:
	docker compose logs --follow api
