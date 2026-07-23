#!/bin/sh
set -eu

script_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
repo_root=$(CDPATH= cd -- "${script_dir}/.." && pwd)
python_bin=${PYTHON_BIN:-${repo_root}/.venv/bin/python}

cd "${repo_root}"
"${python_bin}" -m ruff check .
"${python_bin}" -m mypy app
"${python_bin}" scripts/generate_openapi.py
"${python_bin}" -m pytest -q
git diff --exit-code -- docs/api/openapi-v1.json
PYTHON_BIN="${python_bin}" "${script_dir}/check_migrations.sh"
