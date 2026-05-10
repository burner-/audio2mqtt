param(
    [Parameter(Mandatory=$true)]
    [string]$Device,

    [string]$HostName = "127.0.0.1",
    [int]$Port = 15000
)

$ffmpeg = Get-Command ffmpeg.exe -ErrorAction SilentlyContinue
if (-not $ffmpeg) {
    throw "ffmpeg.exe not found in PATH. Install FFmpeg for Windows and add it to PATH."
}

Write-Host "Streaming DirectShow audio device to tcp://$HostName`:$Port"
Write-Host "Device: $Device"
Write-Host "Stop with Ctrl+C."

& $ffmpeg.Source `
    -hide_banner `
    -loglevel info `
    -f dshow `
    -audio_buffer_size 100 `
    -i "audio=$Device" `
    -ac 1 `
    -ar 16000 `
    -sample_fmt s16 `
    -f s16le `
    "tcp://$HostName`:$Port"
