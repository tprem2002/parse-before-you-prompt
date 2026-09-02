[CmdletBinding(DefaultParameterSetName = "DryRun")]
param(
    [Parameter(Mandatory = $true)][string]$DocumentId,
    [Parameter(Mandatory = $true)][string]$BaselineProcessingRunId,
    [Parameter(Mandatory = $true)][string]$DoclingProcessingRunId,
    [ValidateRange(1, 20)][int]$TopK = 5,
    [Parameter(ParameterSetName = "DryRun")][switch]$DryRun,
    [Parameter(ParameterSetName = "Execute", Mandatory = $true)][switch]$Execute,
    [switch]$ForceNew
)

$ErrorActionPreference = "Stop"
$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location -LiteralPath $projectRoot
$python = Join-Path $projectRoot ".venv\Scripts\python.exe"
$useVenv = Test-Path -LiteralPath $python
if (-not $useVenv -and -not (Get-Command uv -ErrorAction SilentlyContinue)) {
    throw "No local .venv or uv executable was found. Run uv sync first."
}

$arguments = @(
    "--document-id", $DocumentId,
    "--baseline-processing-run-id", $BaselineProcessingRunId,
    "--docling-processing-run-id", $DoclingProcessingRunId,
    "--top-k", $TopK
)
if ($Execute) { $arguments += "--execute" } else { $arguments += "--dry-run" }
if ($ForceNew) { $arguments += "--force-new" }

if ($useVenv) {
    & $python scripts/run_evaluation.py @arguments
} else {
    & uv run python scripts/run_evaluation.py @arguments
}
exit $LASTEXITCODE
