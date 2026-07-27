#!/bin/sh
set -eu

action=${1:-start}
script_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)

find_project_root() {
    for candidate in \
        "${script_dir}" \
        "${script_dir}/../.." \
        "${script_dir}/NanoLoop-Agent" \
        "${script_dir}/../NanoLoop-Agent"
    do
        if [ -f "${candidate}/docker-compose.yml" ]; then
            CDPATH= cd -- "${candidate}" && pwd
            return 0
        fi
    done
    return 1
}

project_root=$(find_project_root || true)
if [ -z "${project_root}" ]; then
    echo "没有找到 docker-compose.yml。请保持启动脚本与 NanoLoop-Agent 文件夹的原始位置。" >&2
    exit 1
fi
cd "${project_root}"

compose() {
    docker compose -f docker-compose.yml "$@"
}

require_docker() {
    if ! command -v docker >/dev/null 2>&1; then
        echo "没有找到 Docker。请先按《Docker 部署与使用手册》安装 Docker Desktop 或 Docker Engine。" >&2
        exit 1
    fi
    if ! docker info >/dev/null 2>&1; then
        echo "Docker 尚未启动。macOS 请打开 Docker Desktop；Linux 请启动 docker 服务。" >&2
        exit 1
    fi
    if ! docker compose version >/dev/null 2>&1; then
        echo "没有找到 Docker Compose v2。请安装 docker-compose-plugin。" >&2
        exit 1
    fi
}

service_ready() {
    service=$1
    container_id=$(compose ps -q "${service}" 2>/dev/null || true)
    [ -n "${container_id}" ] || return 1
    status=$(docker inspect --format '{{if .State.Health}}{{.State.Health.Status}}{{else}}{{.State.Status}}{{end}}' "${container_id}" 2>/dev/null || true)
    [ "${status}" = "healthy" ]
}

all_services_ready() {
    service_ready api && service_ready frontend
}

project_has_containers() {
    [ -n "$(compose ps -q 2>/dev/null || true)" ]
}

port_in_use() {
    port=$1
    if command -v lsof >/dev/null 2>&1; then
        lsof -nP -iTCP:"${port}" -sTCP:LISTEN >/dev/null 2>&1
        return $?
    fi
    if command -v ss >/dev/null 2>&1; then
        ss -ltn | awk '{print $4}' | grep -Eq "(^|:)${port}$"
        return $?
    fi
    return 1
}

open_browser() {
    if command -v open >/dev/null 2>&1; then
        open http://127.0.0.1:3000
    elif command -v xdg-open >/dev/null 2>&1; then
        xdg-open http://127.0.0.1:3000 >/dev/null 2>&1 || true
    fi
}

select_llm_mode() {
    if [ -n "${NANOLOOP_COMPOSE_LLM_PROVIDER:-}" ]; then
        return
    fi
    if [ "${NANOLOOP_ENABLE_LOCAL_QWEN:-0}" != "1" ]; then
        NANOLOOP_COMPOSE_LLM_PROVIDER=extractive
        export NANOLOOP_COMPOSE_LLM_PROVIDER
        echo "评委机默认不启动 Qwen，科研助手采用本地证据摘录模式；图像分析功能不受影响。"
        return
    fi
    model=${NANOLOOP_COMPOSE_LLM_MODEL:-qwen3:4b-instruct-2507-q4_K_M}
    if command -v curl >/dev/null 2>&1 \
        && curl --fail --silent --max-time 3 http://127.0.0.1:11434/api/tags \
            | grep -Fq "\"${model}\""
    then
        NANOLOOP_COMPOSE_LLM_PROVIDER=openai_compatible
        export NANOLOOP_COMPOSE_LLM_PROVIDER
        echo "已检测到本机 Ollama 模型 ${model}，科研助手将使用本地 Qwen。"
    else
        NANOLOOP_COMPOSE_LLM_PROVIDER=extractive
        export NANOLOOP_COMPOSE_LLM_PROVIDER
        echo "未检测到指定的本地 Qwen，科研助手采用证据摘录模式；图像分析功能不受影响。"
    fi
}

find_offline_archive() {
    for archive in \
        "${script_dir}/NanoLoop-Docker-linux-amd64.tar.gz" \
        "${script_dir}/NanoLoop-Docker-linux-amd64.tar" \
        "${project_root}/NanoLoop-Docker-linux-amd64.tar.gz" \
        "${project_root}/NanoLoop-Docker-linux-amd64.tar" \
        "${project_root}/../NanoLoop-Docker-linux-amd64.tar.gz" \
        "${project_root}/../NanoLoop-Docker-linux-amd64.tar"
    do
        if [ -f "${archive}" ]; then
            printf '%s\n' "${archive}"
            return 0
        fi
    done
    return 1
}

verify_model_bundle() {
    for required in \
        "${project_root}/model_artifacts/registry.yaml" \
        "${project_root}/model_artifacts/weights/msbi-instance-balanced-v1.pt" \
        "${project_root}/model_artifacts/weights/unet-large-optimized-v1.pt" \
        "${project_root}/model_artifacts/weights/unet-small-balanced-v1.pt"
    do
        if [ ! -r "${required}" ]; then
            echo "模型交付包不完整或不可读：${required}" >&2
            exit 1
        fi
    done
}

promote_offline_images() {
    for image_pair in \
        "nanoloop-agent:ai4s-amd64 nanoloop-agent:local" \
        "nanoloop-agent-frontend:ai4s-amd64 nanoloop-agent-frontend:local"
    do
        set -- ${image_pair}
        source_image=$1
        target_image=$2
        if docker image inspect "${source_image}" >/dev/null 2>&1; then
            architecture=$(docker image inspect --format '{{.Architecture}}' "${source_image}")
            if [ "${architecture}" != "amd64" ]; then
                echo "离线镜像架构错误：${source_image} 为 ${architecture}，应为 amd64。" >&2
                exit 1
            fi
            docker tag "${source_image}" "${target_image}"
        fi
    done
}

start_nanoloop() {
    require_docker
    if ! project_has_containers && { port_in_use 3000 || port_in_use 8000; }; then
        echo "端口 3000 或 8000 已被其他程序占用。请关闭占用程序后重新运行本脚本。" >&2
        exit 1
    fi

    verify_model_bundle
    select_llm_mode
    NANOLOOP_API_EXTRAS=models
    MODEL_DEVICE=cpu
    COMPOSE_PARALLEL_LIMIT=1
    NANOLOOP_MODEL_ARTIFACTS_DIR="${project_root}/model_artifacts"
    NANOLOOP_BIND_HOST=127.0.0.1
    NANOLOOP_PORT=8000
    NANOLOOP_FRONTEND_PORT=3000
    ONLINE_RESEARCH_ENABLED=false
    export NANOLOOP_API_EXTRAS MODEL_DEVICE COMPOSE_PARALLEL_LIMIT
    export NANOLOOP_MODEL_ARTIFACTS_DIR NANOLOOP_BIND_HOST NANOLOOP_PORT NANOLOOP_FRONTEND_PORT
    export ONLINE_RESEARCH_ENABLED

    if all_services_ready; then
        echo "NanoLoop 已在运行，正在同步当前 Qwen 与联网配置……"
        compose up --detach --no-build
        echo "配置已同步：http://127.0.0.1:3000"
        open_browser
        return
    fi

    archive=$(find_offline_archive || true)
    if [ -n "${archive}" ]; then
        echo "发现离线镜像，正在导入：${archive}"
        docker load --input "${archive}"
        promote_offline_images
    fi

    if ! docker image inspect nanoloop-agent:local >/dev/null 2>&1 \
        || ! docker image inspect nanoloop-agent-frontend:local >/dev/null 2>&1
    then
        echo "未找到完整离线镜像，开始从源码构建 CPU 版。首次构建需要联网，通常需要 10–30 分钟。"
        compose build api
        compose build frontend
    fi

    compose config --quiet
    compose up --detach --no-build

    echo "容器已经启动，正在等待健康检查（最长 5 分钟）……"
    attempt=0
    while [ "${attempt}" -lt 150 ]; do
        if all_services_ready; then
            echo "NanoLoop 启动成功：http://127.0.0.1:3000"
            compose ps
            open_browser
            return
        fi
        attempt=$((attempt + 1))
        sleep 2
    done

    echo "等待超时。下面是容器状态和最近日志：" >&2
    compose ps >&2 || true
    compose logs --tail 120 api frontend >&2 || true
    exit 1
}

stop_nanoloop() {
    require_docker
    compose stop
    echo "NanoLoop 已停止。实验数据保留在 Docker 命名卷中。"
}

status_nanoloop() {
    require_docker
    compose ps
    if all_services_ready; then
        echo "状态：健康"
        echo "前端：http://127.0.0.1:3000"
        echo "API：http://127.0.0.1:8000/docs"
    else
        echo "状态：未完全就绪。可运行本脚本的 start 操作，或查看日志："
        echo "docker compose -f \"${project_root}/docker-compose.yml\" logs --tail 120 api frontend"
        exit 1
    fi
}

case "${action}" in
    start) start_nanoloop ;;
    stop) stop_nanoloop ;;
    status) status_nanoloop ;;
    *)
        echo "用法：$0 start|stop|status" >&2
        exit 2
        ;;
esac
