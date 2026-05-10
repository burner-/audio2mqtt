Push-Location "$PSScriptRoot\.."
try {
    $cid = docker compose ps -q audio2mqtt
    if (-not $cid) {
        throw "audio2mqtt container not found. Run docker compose up first."
    }
    while ($true) {
        Clear-Host
        docker exec $cid nvidia-smi --query-gpu=name,memory.used,memory.free,memory.total,utilization.gpu --format=csv
        Start-Sleep 1
    }
} finally {
    Pop-Location
}
