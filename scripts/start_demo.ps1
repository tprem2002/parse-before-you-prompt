[CmdletBinding()]
param(
    [ValidateRange(1, 65535)][int]$ApiPort = 8080,
    [ValidateRange(1, 65535)][int]$UiPort = 8501
)

$ErrorActionPreference = "Stop"
$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location -LiteralPath $projectRoot

$python = Join-Path $projectRoot ".venv\Scripts\python.exe"
$useVenv = Test-Path -LiteralPath $python
if (-not $useVenv -and -not (Get-Command uv -ErrorAction SilentlyContinue)) {
    throw "No local .venv or uv executable was found. Run uv sync first."
}

if (Get-Command docker-compose -ErrorAction SilentlyContinue) {
    & docker-compose up -d
} elseif (Get-Command docker -ErrorAction SilentlyContinue) {
    & docker compose up -d
} else {
    throw "Docker Compose is required to start PostgreSQL and Chroma."
}
if ($LASTEXITCODE -ne 0) {
    throw "Docker Compose could not start PostgreSQL and Chroma."
}

if ($useVenv) { & $python -m alembic upgrade head } else { & uv run alembic upgrade head }
if ($LASTEXITCODE -ne 0) {
    throw "Alembic migration failed. Verify PostgreSQL and DATABASE_URL."
}

$shellPath = (Get-Process -Id $PID).Path
$apiScript = (Resolve-Path (Join-Path $PSScriptRoot "start_api.ps1")).Path
$uiScript = (Resolve-Path (Join-Path $PSScriptRoot "start_ui.ps1")).Path
$logRoot = Join-Path $projectRoot "data\logs"
New-Item -ItemType Directory -Path $logRoot -Force | Out-Null

$apiProcess = Start-Process -FilePath $shellPath -ArgumentList @(
    "-NoProfile", "-File", $apiScript, "-Port", $ApiPort
) -WorkingDirectory $projectRoot -WindowStyle Hidden -PassThru `
    -RedirectStandardOutput (Join-Path $logRoot "api.stdout.log") `
    -RedirectStandardError (Join-Path $logRoot "api.stderr.log")
$uiProcess = Start-Process -FilePath $shellPath -ArgumentList @(
    "-NoProfile", "-File", $uiScript, "-Port", $UiPort,
    "-ApiBaseUrl", "http://127.0.0.1:$ApiPort"
) -WorkingDirectory $projectRoot -WindowStyle Hidden -PassThru `
    -RedirectStandardOutput (Join-Path $logRoot "ui.stdout.log") `
    -RedirectStandardError (Join-Path $logRoot "ui.stderr.log")

Write-Host "FastAPI process: $($apiProcess.Id) · http://127.0.0.1:$ApiPort/docs"
Write-Host "Streamlit process: $($uiProcess.Id) · http://127.0.0.1:$UiPort"
Write-Host "Logs: data/logs"
