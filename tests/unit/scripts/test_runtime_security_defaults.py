from __future__ import annotations

import stat
import subprocess
import sys
from pathlib import Path

import yaml

from app.storage.file_token_keyring_store import FileTokenV2KeyRingStore

_REPOSITORY_ROOT = Path(__file__).parents[3]


def _entrypoint_keyring_program() -> str:
    entrypoint = (_REPOSITORY_ROOT / "scripts" / "docker-entrypoint.sh").read_text(
        encoding="utf-8"
    )
    marker = 'if ! python - "${FILE_TOKEN_V2_KEYRING_PATH}" <<\'PY\'\n'
    return entrypoint.split(marker, maxsplit=1)[1].split("\nPY\n", maxsplit=1)[0]


def _run_entrypoint_keyring_program(path: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-", str(path)],
        input=_entrypoint_keyring_program(),
        text=True,
        capture_output=True,
        cwd=_REPOSITORY_ROOT,
        check=False,
    )


def test_bundled_uvicorn_commands_disable_proxy_header_rewriting() -> None:
    dockerfile = (_REPOSITORY_ROOT / "Dockerfile").read_text(encoding="utf-8")
    entrypoint = (_REPOSITORY_ROOT / "scripts" / "docker-entrypoint.sh").read_text(
        encoding="utf-8"
    )
    makefile = (_REPOSITORY_ROOT / "Makefile").read_text(encoding="utf-8")

    docker_command = next(line for line in dockerfile.splitlines() if line.startswith("CMD ["))
    entrypoint_command = next(
        line for line in entrypoint.splitlines() if "set -- uvicorn app.main:app" in line
    )
    make_serve_command = next(
        line for line in makefile.splitlines() if "-m uvicorn app.main:app" in line
    )

    assert '"--no-proxy-headers"' in docker_command
    assert "--no-proxy-headers" in entrypoint_command
    assert "--no-proxy-headers" in make_serve_command


def test_production_entrypoint_initializes_missing_v2_keyring_via_store(
    tmp_path: Path,
) -> None:
    keyring_path = tmp_path / "private-v2-keyring.json"

    completed = _run_entrypoint_keyring_program(keyring_path)

    assert completed.returncode == 0
    assert completed.stdout == completed.stderr == ""
    assert stat.S_IMODE(keyring_path.stat().st_mode) == 0o600
    loaded = FileTokenV2KeyRingStore(keyring_path).load()
    assert loaded.active_kid == "initial"
    assert loaded.retained_kids == ("initial",)


def test_production_entrypoint_rejects_existing_corrupt_or_symlink_keyring_without_leak(
    tmp_path: Path,
) -> None:
    corrupt = tmp_path / "do-not-log-corrupt-keyring.json"
    secret_payload = "secret-key-material-must-not-appear"
    corrupt.write_text(secret_payload, encoding="utf-8")
    corrupt.chmod(0o600)

    corrupt_result = _run_entrypoint_keyring_program(corrupt)

    assert corrupt_result.returncode == 1
    assert corrupt_result.stdout == ""
    assert "invalid_payload" in corrupt_result.stderr
    assert str(corrupt) not in corrupt_result.stderr
    assert secret_payload not in corrupt_result.stderr

    target = tmp_path / "target-keyring.json"
    FileTokenV2KeyRingStore(target).initialize(key=b"s" * 32)
    symlink = tmp_path / "do-not-log-symlink.json"
    symlink.symlink_to(target)

    symlink_result = _run_entrypoint_keyring_program(symlink)

    assert symlink_result.returncode == 1
    assert "unsafe_type" in symlink_result.stderr
    assert str(symlink) not in symlink_result.stderr
    assert str(target) not in symlink_result.stderr


def test_compose_and_example_use_the_settings_keyring_environment_name() -> None:
    compose = yaml.safe_load(
        (_REPOSITORY_ROOT / "docker-compose.yml").read_text(encoding="utf-8")
    )
    api_environment = compose["services"]["api"]["environment"]
    assert (
        api_environment["FILE_TOKEN_V2_KEYRING_PATH"]
        == "/app/data/.file_token_v2_keyring.json"
    )
    assert "NANOLOOP_FILE_TOKEN_V2_KEYRING_PATH" not in api_environment

    example = (_REPOSITORY_ROOT / ".env.example").read_text(encoding="utf-8")
    assert "FILE_TOKEN_V2_KEYRING_PATH=./data/.file_token_v2_keyring.json" in example


def test_base_compose_connects_to_host_qwen_by_default() -> None:
    compose = yaml.safe_load(
        (_REPOSITORY_ROOT / "docker-compose.yml").read_text(encoding="utf-8")
    )
    api = compose["services"]["api"]
    environment = api["environment"]

    assert (
        environment["LLM_PROVIDER"]
        == "${NANOLOOP_COMPOSE_LLM_PROVIDER:-openai_compatible}"
    )
    assert environment["LLM_BASE_URL"] == (
        "${NANOLOOP_COMPOSE_LLM_BASE_URL:-"
        "http://host.docker.internal:11434/v1}"
    )
    assert environment["LLM_MODEL"] == (
        "${NANOLOOP_COMPOSE_LLM_MODEL:-qwen3:4b-instruct-2507-q4_K_M}"
    )
    assert "host.docker.internal:host-gateway" in api["extra_hosts"]


def test_api_image_includes_the_non_secret_keyring_operator_cli() -> None:
    dockerfile = (_REPOSITORY_ROOT / "Dockerfile").read_text(encoding="utf-8")

    assert "FILE_TOKEN_V2_KEYRING_PATH=/app/data/.file_token_v2_keyring.json" in dockerfile
    assert (
        "COPY --chown=nanoloop:nanoloop scripts/manage_file_token_keyring.py "
        "./scripts/manage_file_token_keyring.py"
    ) in dockerfile
    assert "/app/scripts/manage_file_token_keyring.py" in dockerfile


def test_model_compose_build_is_cpu_only_and_serial() -> None:
    dockerfile = (_REPOSITORY_ROOT / "Dockerfile").read_text(encoding="utf-8")
    constraints = (
        _REPOSITORY_ROOT / "docker-models-cpu-constraints.txt"
    ).read_text(encoding="utf-8")
    makefile = (_REPOSITORY_ROOT / "Makefile").read_text(encoding="utf-8")

    assert "https://download.pytorch.org/whl/cpu" in dockerfile
    assert "--no-deps" in dockerfile
    assert "--requirement docker-models-cpu-constraints.txt" in dockerfile
    assert '--index-url "${PYPI_INDEX_URL}"' in dockerfile
    assert "torch==2.13.0" in constraints
    assert "torchvision==0.28.0" in constraints

    project = (_REPOSITORY_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert '"torch>=2.13,<3"' in project
    assert '"torchvision>=0.28,<1"' in project

    compose = yaml.safe_load(
        (_REPOSITORY_ROOT / "docker-compose.yml").read_text(encoding="utf-8")
    )
    assert (
        compose["services"]["api"]["build"]["args"]["PYPI_INDEX_URL"]
        == "${PYPI_INDEX_URL:-https://pypi.org/simple}"
    )
    frontend_healthcheck = compose["services"]["frontend"]["healthcheck"]["test"]
    assert frontend_healthcheck[:3] == ["CMD", "node", "-e"]

    recipe = makefile.split("compose-up-models:\n", maxsplit=1)[1].split(
        "\n\n", maxsplit=1
    )[0]
    assert "COMPOSE_PARALLEL_LIMIT=1" in recipe
    assert "docker compose build api" in recipe
    assert "docker compose build frontend" in recipe
    assert "docker compose up --detach --no-build" in recipe

    install_recipe = makefile.split("install-models:", maxsplit=1)[1].split(
        "\n\n", maxsplit=1
    )[0]
    assert "https://download.pytorch.org/whl/cpu" in install_recipe
    assert "--requirement docker-models-cpu-constraints.txt" in install_recipe
    assert "--constraint docker-models-cpu-constraints.txt" in install_recipe
