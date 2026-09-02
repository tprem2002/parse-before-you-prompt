[CmdletBinding()]
param(
    [switch]$Execute,
    [string]$Confirmation = ""
)

$ErrorActionPreference = "Stop"
$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location -LiteralPath $projectRoot
$python = Join-Path $projectRoot ".venv\Scripts\python.exe"
$useVenv = Test-Path -LiteralPath $python
if (-not $useVenv -and -not (Get-Command uv -ErrorAction SilentlyContinue)) {
    throw "No local .venv or uv executable was found. Run uv sync first."
}

$arguments = @()
if ($Execute) {
    if ($Confirmation -cne "RESET_PROJECT_AURORA_DEMO") {
        throw "Destructive reset requires -Confirmation RESET_PROJECT_AURORA_DEMO exactly."
    }
    $arguments += @("--execute", "--confirmation", $Confirmation)
}

if ($useVenv) {
    & $python scripts/reset_demo.py @arguments
} else {
    & uv run python scripts/reset_demo.py @arguments
}
exit $LASTEXITCODE
