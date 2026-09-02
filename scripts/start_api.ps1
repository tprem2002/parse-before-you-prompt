[CmdletBinding()]
param(
    [string]$HostAddress = "127.0.0.1",
    [ValidateRange(1, 65535)][int]$Port = 8080
)

$ErrorActionPreference = "Stop"
$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location -LiteralPath $projectRoot

$python = Join-Path $projectRoot ".venv\Scripts\python.exe"
$useVenv = Test-Path -LiteralPath $python
if (-not $useVenv -and -not (Get-Command uv -ErrorAction SilentlyContinue)) {
    throw "No local .venv or uv executable was found. Run uv sync first."
}
if (-not (Test-Path -LiteralPath ".env")) {
    throw "Missing .env. Copy .env.example to .env and supply the server-side Azure settings."
}

Write-Host "Checking the database migration state..."
if ($useVenv) { & $python -m alembic current } else { & uv run alembic current }
if ($LASTEXITCODE -ne 0) {
    throw "Alembic could not connect. Start PostgreSQL and verify DATABASE_URL."
}

Write-Host "Starting FastAPI at http://${HostAddress}:$Port"
if ($useVenv) {
    & $python -m fastapi dev app/api/main.py --host $HostAddress --port $Port
} else {
    & uv run fastapi dev app/api/main.py --host $HostAddress --port $Port
}
exit $LASTEXITCODE
