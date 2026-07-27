param(
    [ValidateSet("start", "stop", "status")]
    [string]$Action = "start"
)

$ErrorActionPreference = "Stop"

function Find-ProjectRoot {
    $Candidates = @(
        $PSScriptRoot,
        (Join-Path $PSScriptRoot "..\.."),
        (Join-Path $PSScriptRoot "NanoLoop-Agent"),
        (Join-Path $PSScriptRoot "..\NanoLoop-Agent")
    )
    foreach ($Candidate in $Candidates) {
        $Resolved = [System.IO.Path]::GetFullPath($Candidate)
        if (Test-Path (Join-Path $Resolved "docker-compose.yml")) {
            return $Resolved
        }
    }
    throw "没有找到 docker-compose.yml。请保持启动脚本与 NanoLoop-Agent 文件夹的原始位置。"
}

function Invoke-Docker {
    param([Parameter(Mandatory = $true)][string[]]$DockerArguments)
    & docker @DockerArguments
    if ($LASTEXITCODE -ne 0) {
        throw "docker $($DockerArguments -join ' ') 执行失败，退出码为 $LASTEXITCODE。"
    }
}

function Invoke-Compose {
    param([Parameter(Mandatory = $true)][string[]]$ComposeArguments)
    Invoke-Docker -DockerArguments (@("compose", "-f", "docker-compose.yml") + $ComposeArguments)
}

function Test-ServiceHealthy {
    param([Parameter(Mandatory = $true)][string]$Service)
    $ContainerId = (& docker compose -f docker-compose.yml ps -q $Service 2>$null)
    if (-not $ContainerId) {
        return $false
    }
    $Status = (& docker inspect --format "{{if .State.Health}}{{.State.Health.Status}}{{else}}{{.State.Status}}{{end}}" $ContainerId 2>$null)
    return $LASTEXITCODE -eq 0 -and $Status -eq "healthy"
}

function Test-AllServicesHealthy {
    return (Test-ServiceHealthy "api") -and (Test-ServiceHealthy "frontend")
}

function Test-ProjectHasContainers {
    $ContainerIds = @(& docker compose -f docker-compose.yml ps -q 2>$null)
    return $ContainerIds.Count -gt 0
}

function Test-Port {
    param([Parameter(Mandatory = $true)][int]$Port)
    $Client = New-Object System.Net.Sockets.TcpClient
    try {
        $Task = $Client.ConnectAsync("127.0.0.1", $Port)
        if (-not $Task.Wait(500)) {
            return $false
        }
        return $Client.Connected
    } catch {
        return $false
    } finally {
        $Client.Dispose()
    }
}

function Require-Docker {
    if (-not (Get-Command docker -ErrorAction SilentlyContinue)) {
        throw "没有找到 Docker。请先按《Docker 部署与使用手册》安装 Docker Desktop。"
    }
    & docker info *> $null
    if ($LASTEXITCODE -ne 0) {
        throw "Docker 尚未启动。请从开始菜单打开 Docker Desktop，等待左下角显示 Engine running。"
    }
    & docker compose version *> $null
    if ($LASTEXITCODE -ne 0) {
        throw "没有找到 Docker Compose v2。请更新 Docker Desktop。"
    }
}

function Select-LlmMode {
    if ($env:NANOLOOP_COMPOSE_LLM_PROVIDER) {
        return
    }
    if ($env:NANOLOOP_ENABLE_LOCAL_QWEN -ne "1") {
        $env:NANOLOOP_COMPOSE_LLM_PROVIDER = "extractive"
        Write-Host "评委机默认不启动 Qwen，科研助手采用本地证据摘录模式；图像分析功能不受影响。"
        return
    }
    $Model = if ($env:NANOLOOP_COMPOSE_LLM_MODEL) {
        $env:NANOLOOP_COMPOSE_LLM_MODEL
    } else {
        "qwen3:4b-instruct-2507-q4_K_M"
    }
    try {
        $Tags = Invoke-RestMethod -Uri "http://127.0.0.1:11434/api/tags" -TimeoutSec 3
        $Names = @($Tags.models | ForEach-Object { $_.name })
        if ($Names -contains $Model) {
            $env:NANOLOOP_COMPOSE_LLM_PROVIDER = "openai_compatible"
            Write-Host "已检测到本机 Ollama 模型 $Model，科研助手将使用本地 Qwen。"
            return
        }
    } catch {
        # Absence of Ollama is expected on a clean judging computer.
    }
    $env:NANOLOOP_COMPOSE_LLM_PROVIDER = "extractive"
    Write-Host "未检测到指定的本地 Qwen，科研助手采用证据摘录模式；图像分析功能不受影响。"
}

function Find-OfflineArchive {
    $Candidates = @(
        (Join-Path $PSScriptRoot "NanoLoop-Docker-linux-amd64.tar.gz"),
        (Join-Path $PSScriptRoot "NanoLoop-Docker-linux-amd64.tar"),
        (Join-Path $ProjectRoot "NanoLoop-Docker-linux-amd64.tar.gz"),
        (Join-Path $ProjectRoot "NanoLoop-Docker-linux-amd64.tar"),
        (Join-Path $ProjectRoot "..\NanoLoop-Docker-linux-amd64.tar.gz"),
        (Join-Path $ProjectRoot "..\NanoLoop-Docker-linux-amd64.tar")
    )
    foreach ($Candidate in $Candidates) {
        if (Test-Path $Candidate) {
            return [System.IO.Path]::GetFullPath($Candidate)
        }
    }
    return $null
}

function Assert-ModelBundle {
    $RequiredFiles = @(
        (Join-Path $ProjectRoot "model_artifacts\registry.yaml"),
        (Join-Path $ProjectRoot "model_artifacts\weights\msbi-instance-balanced-v1.pt"),
        (Join-Path $ProjectRoot "model_artifacts\weights\unet-large-optimized-v1.pt"),
        (Join-Path $ProjectRoot "model_artifacts\weights\unet-small-balanced-v1.pt")
    )
    foreach ($RequiredFile in $RequiredFiles) {
        if (-not (Test-Path -PathType Leaf $RequiredFile)) {
            throw "模型交付包不完整或不可读：$RequiredFile"
        }
    }
}

function Promote-OfflineImages {
    $ImagePairs = @(
        @("nanoloop-agent:ai4s-amd64", "nanoloop-agent:local"),
        @("nanoloop-agent-frontend:ai4s-amd64", "nanoloop-agent-frontend:local")
    )
    foreach ($ImagePair in $ImagePairs) {
        $SourceImage = $ImagePair[0]
        $TargetImage = $ImagePair[1]
        & docker image inspect $SourceImage *> $null
        if ($LASTEXITCODE -eq 0) {
            $Architecture = (& docker image inspect --format "{{.Architecture}}" $SourceImage)
            if ($Architecture -ne "amd64") {
                throw "离线镜像架构错误：$SourceImage 为 $Architecture，应为 amd64。"
            }
            Invoke-Docker -DockerArguments @("tag", $SourceImage, $TargetImage)
        }
    }
}

function Start-NanoLoop {
    Require-Docker
    if ((-not (Test-ProjectHasContainers)) -and ((Test-Port 3000) -or (Test-Port 8000))) {
        throw "端口 3000 或 8000 已被其他程序占用。请关闭占用程序后重新双击启动脚本。"
    }

    Assert-ModelBundle
    Select-LlmMode
    $env:NANOLOOP_API_EXTRAS = "models"
    $env:MODEL_DEVICE = "cpu"
    $env:COMPOSE_PARALLEL_LIMIT = "1"
    $env:NANOLOOP_MODEL_ARTIFACTS_DIR = (Join-Path $ProjectRoot "model_artifacts")
    $env:NANOLOOP_BIND_HOST = "127.0.0.1"
    $env:NANOLOOP_PORT = "8000"
    $env:NANOLOOP_FRONTEND_PORT = "3000"
    $env:ONLINE_RESEARCH_ENABLED = "false"

    if (Test-AllServicesHealthy) {
        Write-Host "NanoLoop 已在运行，正在同步当前 Qwen 与联网配置……"
        Invoke-Compose -ComposeArguments @("up", "--detach", "--no-build")
        Write-Host "配置已同步：http://127.0.0.1:3000"
        Start-Process "http://127.0.0.1:3000"
        return
    }

    $Archive = Find-OfflineArchive
    if ($Archive) {
        Write-Host "发现离线镜像，正在导入：$Archive"
        Invoke-Docker -DockerArguments @("load", "--input", $Archive)
        Promote-OfflineImages
    }

    & docker image inspect "nanoloop-agent:local" *> $null
    $ApiImageReady = $LASTEXITCODE -eq 0
    & docker image inspect "nanoloop-agent-frontend:local" *> $null
    $FrontendImageReady = $LASTEXITCODE -eq 0
    if (-not ($ApiImageReady -and $FrontendImageReady)) {
        Write-Host "未找到完整离线镜像，开始从源码构建 CPU 版。首次构建需要联网，通常需要 10–30 分钟。"
        Invoke-Compose -ComposeArguments @("build", "api")
        Invoke-Compose -ComposeArguments @("build", "frontend")
    }

    Invoke-Compose -ComposeArguments @("config", "--quiet")
    Invoke-Compose -ComposeArguments @("up", "--detach", "--no-build")
    Write-Host "容器已经启动，正在等待健康检查（最长 5 分钟）……"

    for ($Attempt = 0; $Attempt -lt 150; $Attempt++) {
        if (Test-AllServicesHealthy) {
            Write-Host "NanoLoop 启动成功：http://127.0.0.1:3000"
            Invoke-Compose -ComposeArguments @("ps")
            Start-Process "http://127.0.0.1:3000"
            return
        }
        Start-Sleep -Seconds 2
    }

    Write-Host "等待超时。下面是容器状态和最近日志：" -ForegroundColor Red
    & docker compose -f docker-compose.yml ps
    & docker compose -f docker-compose.yml logs --tail 120 api frontend
    throw "NanoLoop 未在 5 分钟内通过健康检查。"
}

function Stop-NanoLoop {
    Require-Docker
    Invoke-Compose -ComposeArguments @("stop")
    Write-Host "NanoLoop 已停止。实验数据保留在 Docker 命名卷中。"
}

function Show-NanoLoopStatus {
    Require-Docker
    Invoke-Compose -ComposeArguments @("ps")
    if (Test-AllServicesHealthy) {
        Write-Host "状态：健康"
        Write-Host "前端：http://127.0.0.1:3000"
        Write-Host "API：http://127.0.0.1:8000/docs"
        return
    }
    throw "状态：未完全就绪。请运行启动脚本，或执行 docker compose logs --tail 120 api frontend。"
}

$ProjectRoot = Find-ProjectRoot
Push-Location $ProjectRoot
try {
    switch ($Action) {
        "start" { Start-NanoLoop }
        "stop" { Stop-NanoLoop }
        "status" { Show-NanoLoopStatus }
    }
} finally {
    Pop-Location
}
