$ffmpeg = Get-Command ffmpeg.exe -ErrorAction SilentlyContinue
if (-not $ffmpeg) {
    throw "ffmpeg.exe not found in PATH. Install FFmpeg for Windows and add it to PATH."
}

Write-Host "Listing Windows DirectShow devices..."
Write-Host "Copy the exact audio device name from lines ending with (audio)."
Write-Host ""
cmd /c "`"$($ffmpeg.Source)`" -hide_banner -list_devices true -f dshow -i dummy 2>&1"
