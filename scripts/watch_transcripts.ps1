$path = Join-Path $PSScriptRoot "..\output\transcripts.jsonl"
if (-not (Test-Path $path)) {
    New-Item -ItemType Directory -Force -Path (Split-Path $path) | Out-Null
    New-Item -ItemType File -Force -Path $path | Out-Null
}
Get-Content $path -Wait
