#!/bin/sh
set -eu

script_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
repository_root=$(dirname -- "${script_dir}")
cd "${repository_root}"

accelerator=${NANOLOOP_ACCELERATOR:-auto}
probe_image=${NANOLOOP_GPU_PROBE_IMAGE:-busybox:1.37}

case "${accelerator}" in
    auto|cpu|cuda) ;;
    *)
        echo "NANOLOOP_ACCELERATOR must be auto, cpu, or cuda." >&2
        exit 2
        ;;
esac

if ! docker info >/dev/null 2>&1; then
    echo "Docker is unavailable. Start Docker Desktop or the Docker daemon first." >&2
    exit 1
fi
if ! docker compose version >/dev/null 2>&1; then
    echo "Docker Compose v2 is required." >&2
    exit 1
fi

docker_cuda_available() {
    docker run --rm --gpus all "${probe_image}" sh -c \
        'test -e /dev/nvidiactl || test -d /proc/driver/nvidia/gpus' \
        >/dev/null 2>&1
}

selected_accelerator=cpu
if [ "${accelerator}" = "cuda" ]; then
    if ! docker_cuda_available; then
        echo "CUDA was requested, but Docker cannot access an NVIDIA GPU." >&2
        exit 1
    fi
    selected_accelerator=cuda
elif [ "${accelerator}" = "auto" ] && docker_cuda_available; then
    selected_accelerator=cuda
fi

case "${NANOLOOP_API_EXTRAS:-}" in
    "") NANOLOOP_API_EXTRAS=models ;;
    rag) NANOLOOP_API_EXTRAS=rag,models ;;
    models|rag,models|models,rag) ;;
    *)
        echo "NANOLOOP_API_EXTRAS must be empty, rag, models, rag,models, or models,rag." >&2
        exit 2
        ;;
esac
export NANOLOOP_API_EXTRAS

MODEL_DEVICE=${MODEL_DEVICE:-auto}
export MODEL_DEVICE

set -- -f docker-compose.yml
if [ "${selected_accelerator}" = "cuda" ]; then
    set -- "$@" -f docker-compose.gpu.yml
fi

echo "NanoLoop accelerator: ${selected_accelerator} (MODEL_DEVICE=${MODEL_DEVICE})"
docker compose "$@" config --quiet
COMPOSE_PARALLEL_LIMIT=1 docker compose "$@" build api
COMPOSE_PARALLEL_LIMIT=1 docker compose "$@" build frontend
docker compose "$@" up --detach --no-build
