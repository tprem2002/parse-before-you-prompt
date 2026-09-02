[CmdletBinding()]
param(
    [ValidateRange(1, 65535)][int]$Port = 8501,
    [string]$ApiBaseUrl = "http://127.0.0.1:8080"
)

$ErrorActionPreference = "Stop"
$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location -LiteralPath $projectRoot

$python = Join-Path $projectRoot ".venv\Scripts\python.exe"
$useVenv = Test-Path -LiteralPath $python
if (-not $useVenv -and -not (Get-Command uv -ErrorAction SilentlyContinue)) {
    throw "No local .venv or uv executable was found. Run uv sync first."
}
$env:PBTP_API_BASE_URL = $ApiBaseUrl.TrimEnd("/")
Write-Host "Starting Streamlit at http://127.0.0.1:$Port"
Write-Host "FastAPI boundary: $env:PBTP_API_BASE_URL"
if ($useVenv) {
    & $python -m streamlit run ui/Home.py --server.address 127.0.0.1 --server.port $Port
} else {
    & uv run streamlit run ui/Home.py --server.address 127.0.0.1 --server.port $Port
}
exit $LASTEXITCODE
