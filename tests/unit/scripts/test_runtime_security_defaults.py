from __future__ import annotations

import os
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


def test_model_compose_build_supports_portable_cpu_and_optional_cuda() -> None:
    dockerfile = (_REPOSITORY_ROOT / "Dockerfile").read_text(encoding="utf-8")
    constraints = (_REPOSITORY_ROOT / "docker-models-constraints.txt").read_text(
        encoding="utf-8"
    )
    makefile = (_REPOSITORY_ROOT / "Makefile").read_text(encoding="utf-8")

    assert "https://download.pytorch.org/whl/cpu" in dockerfile
    assert 'ARG PYTORCH_INDEX_URL="https://download.pytorch.org/whl/cpu"' in dockerfile
    assert "--no-deps" in dockerfile
    assert "--requirement docker-models-constraints.txt" in dockerfile
    assert '--index-url "${PYPI_INDEX_URL}"' in dockerfile
    assert "MODEL_DEVICE=auto" in dockerfile
    assert "torch==2.13.0" in constraints
    assert "torchvision==0.28.0" in constraints

    project = (_REPOSITORY_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert '"torch>=2.11,<3"' in project
    assert '"torchvision>=0.26,<1"' in project

    compose = yaml.safe_load(
        (_REPOSITORY_ROOT / "docker-compose.yml").read_text(encoding="utf-8")
    )
    assert (
        compose["services"]["api"]["build"]["args"]["PYPI_INDEX_URL"]
        == "${PYPI_INDEX_URL:-https://pypi.org/simple}"
    )
    assert compose["services"]["api"]["build"]["args"]["PYTORCH_INDEX_URL"] == (
        "${PYTORCH_INDEX_URL:-https://download.pytorch.org/whl/cpu}"
    )
    assert compose["services"]["api"]["environment"]["MODEL_DEVICE"] == (
        "${MODEL_DEVICE:-auto}"
    )
    frontend_healthcheck = compose["services"]["frontend"]["healthcheck"]["test"]
    assert frontend_healthcheck[:3] == ["CMD", "node", "-e"]

    gpu_compose = yaml.safe_load(
        (_REPOSITORY_ROOT / "docker-compose.gpu.yml").read_text(encoding="utf-8")
    )
    gpu_api = gpu_compose["services"]["api"]
    assert gpu_api["build"]["args"]["PYTORCH_INDEX_URL"] == (
        "${PYTORCH_GPU_INDEX_URL:-https://download.pytorch.org/whl/cu126}"
    )
    gpu_device = gpu_api["deploy"]["resources"]["reservations"]["devices"][0]
    assert gpu_device == {
        "driver": "nvidia",
        "count": "all",
        "capabilities": ["gpu"],
    }

    recipe = makefile.split("compose-up-models:\n", maxsplit=1)[1].split(
        "\n\n", maxsplit=1
    )[0]
    assert "./scripts/compose-up-auto.sh" in recipe

    install_recipe = makefile.split("install-models:", maxsplit=1)[1].split(
        "\n\n", maxsplit=1
    )[0]
    assert "https://download.pytorch.org/whl/cpu" in install_recipe
    assert "--requirement docker-models-constraints.txt" in install_recipe
    assert "--constraint docker-models-constraints.txt" in install_recipe


def test_cross_platform_compose_launchers_auto_detect_or_force_cuda() -> None:
    posix_launcher = _REPOSITORY_ROOT / "scripts" / "compose-up-auto.sh"
    powershell_launcher = _REPOSITORY_ROOT / "scripts" / "compose-up-auto.ps1"
    posix = posix_launcher.read_text(encoding="utf-8")
    powershell = powershell_launcher.read_text(encoding="utf-8")

    assert subprocess.run(
        ["sh", "-n", str(posix_launcher)],
        cwd=_REPOSITORY_ROOT,
        check=False,
    ).returncode == 0
    assert stat.S_IMODE(posix_launcher.stat().st_mode) == 0o755
    for launcher in (posix, powershell):
        assert "NANOLOOP_ACCELERATOR" in launcher
        assert "docker-compose.gpu.yml" in launcher
        assert "--gpus" in launcher
        assert "MODEL_DEVICE" in launcher
        assert "docker compose" in launcher or '"compose"' in launcher


def test_posix_compose_launcher_selects_gpu_overlay_only_after_probe(
    tmp_path: Path,
) -> None:
    launcher = _REPOSITORY_ROOT / "scripts" / "compose-up-auto.sh"
    docker = tmp_path / "docker"
    call_log = tmp_path / "docker-calls.log"
    docker.write_text(
        """#!/bin/sh
printf '%s\\n' "$*" >> "${NANOLOOP_DOCKER_CALL_LOG}"
if [ "${1:-}" = "run" ]; then
    exit "${NANOLOOP_TEST_GPU_PROBE_EXIT:-1}"
fi
exit 0
""",
        encoding="utf-8",
    )
    docker.chmod(0o755)

    base_environment = os.environ.copy()
    base_environment.update(
        {
            "PATH": f"{tmp_path}:{base_environment['PATH']}",
            "NANOLOOP_DOCKER_CALL_LOG": str(call_log),
            "NANOLOOP_GPU_PROBE_IMAGE": "nanoloop-test-probe",
        }
    )

    cpu_environment = base_environment | {
        "NANOLOOP_ACCELERATOR": "cpu",
        "NANOLOOP_TEST_GPU_PROBE_EXIT": "0",
    }
    cpu_run = subprocess.run(
        [str(launcher)],
        cwd=_REPOSITORY_ROOT,
        env=cpu_environment,
        text=True,
        capture_output=True,
        check=False,
    )
    assert cpu_run.returncode == 0
    cpu_calls = call_log.read_text(encoding="utf-8")
    assert "run --rm --gpus all" not in cpu_calls
    assert "docker-compose.gpu.yml" not in cpu_calls
    assert "NanoLoop accelerator: cpu" in cpu_run.stdout

    call_log.unlink()
    gpu_environment = base_environment | {
        "NANOLOOP_ACCELERATOR": "auto",
        "NANOLOOP_TEST_GPU_PROBE_EXIT": "0",
    }
    gpu_run = subprocess.run(
        [str(launcher)],
        cwd=_REPOSITORY_ROOT,
        env=gpu_environment,
        text=True,
        capture_output=True,
        check=False,
    )
    assert gpu_run.returncode == 0
    gpu_calls = call_log.read_text(encoding="utf-8")
    assert "run --rm --gpus all nanoloop-test-probe" in gpu_calls
    assert "-f docker-compose.gpu.yml" in gpu_calls
    assert "NanoLoop accelerator: cuda" in gpu_run.stdout

    call_log.unlink()
    unavailable_environment = base_environment | {
        "NANOLOOP_ACCELERATOR": "cuda",
        "NANOLOOP_TEST_GPU_PROBE_EXIT": "1",
    }
    unavailable_run = subprocess.run(
        [str(launcher)],
        cwd=_REPOSITORY_ROOT,
        env=unavailable_environment,
        text=True,
        capture_output=True,
        check=False,
    )
    assert unavailable_run.returncode == 1
    unavailable_calls = call_log.read_text(encoding="utf-8")
    assert "run --rm --gpus all nanoloop-test-probe" in unavailable_calls
    assert "compose -f" not in unavailable_calls
    assert "Docker cannot access an NVIDIA GPU" in unavailable_run.stderr
