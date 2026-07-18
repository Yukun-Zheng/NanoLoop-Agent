from __future__ import annotations

from pathlib import Path

_REPOSITORY_ROOT = Path(__file__).parents[3]


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
