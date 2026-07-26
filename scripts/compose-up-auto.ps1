$ErrorActionPreference = "Stop"

$RepositoryRoot = Split-Path -Parent $PSScriptRoot
$Accelerator = if ($env:NANOLOOP_ACCELERATOR) {
    $env:NANOLOOP_ACCELERATOR.ToLowerInvariant()
} else {
    "auto"
}
$ProbeImage = if ($env:NANOLOOP_GPU_PROBE_IMAGE) {
    $env:NANOLOOP_GPU_PROBE_IMAGE
} else {
    "busybox:1.37"
}

if ($Accelerator -notin @("auto", "cpu", "cuda")) {
    throw "NANOLOOP_ACCELERATOR must be auto, cpu, or cuda."
}

function Invoke-Docker {
    param([Parameter(Mandatory = $true)][string[]]$DockerArguments)

    & docker @DockerArguments
    if ($LASTEXITCODE -ne 0) {
        throw "docker $($DockerArguments -join ' ') failed with exit code $LASTEXITCODE."
    }
}

function Test-DockerCuda {
    & docker run --rm --gpus all $ProbeImage sh -c `
        "test -e /dev/nvidiactl || test -d /proc/driver/nvidia/gpus" *> $null
    return $LASTEXITCODE -eq 0
}

Push-Location $RepositoryRoot
try {
    Invoke-Docker -DockerArguments @("info")
    Invoke-Docker -DockerArguments @("compose", "version")

    $SelectedAccelerator = "cpu"
    $DockerCudaAvailable = $false
    if ($Accelerator -ne "cpu") {
        $DockerCudaAvailable = Test-DockerCuda
    }
    if ($Accelerator -eq "cuda" -and -not $DockerCudaAvailable) {
        throw "CUDA was requested, but Docker cannot access an NVIDIA GPU."
    }
    if ($Accelerator -eq "cuda" -or ($Accelerator -eq "auto" -and $DockerCudaAvailable)) {
        $SelectedAccelerator = "cuda"
    }

    switch ($env:NANOLOOP_API_EXTRAS) {
        { [string]::IsNullOrEmpty($_) } { $env:NANOLOOP_API_EXTRAS = "models"; break }
        "rag" { $env:NANOLOOP_API_EXTRAS = "rag,models"; break }
        "models" { break }
        "rag,models" { break }
        "models,rag" { break }
        default {
            throw (
                "NANOLOOP_API_EXTRAS must be empty, rag, models, " +
                "rag,models, or models,rag."
            )
        }
    }
    if (-not $env:MODEL_DEVICE) {
        $env:MODEL_DEVICE = "auto"
    }
    $env:COMPOSE_PARALLEL_LIMIT = "1"

    $ComposeFiles = @("-f", "docker-compose.yml")
    if ($SelectedAccelerator -eq "cuda") {
        $ComposeFiles += @("-f", "docker-compose.gpu.yml")
    }

    Write-Host (
        "NanoLoop accelerator: $SelectedAccelerator " +
        "(MODEL_DEVICE=$($env:MODEL_DEVICE))"
    )
    Invoke-Docker -DockerArguments (@("compose") + $ComposeFiles + @("config", "--quiet"))
    Invoke-Docker -DockerArguments (@("compose") + $ComposeFiles + @("build", "api"))
    Invoke-Docker -DockerArguments (@("compose") + $ComposeFiles + @("build", "frontend"))
    Invoke-Docker -DockerArguments (@("compose") + $ComposeFiles + @(
        "up",
        "--detach",
        "--no-build"
    ))
} finally {
    Pop-Location
}
