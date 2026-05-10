Push-Location "$PSScriptRoot\.."
try {
    docker compose exec audio2mqtt nvidia-smi
} finally {
    Pop-Location
}
