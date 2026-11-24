# Thin wrapper — prefers scripts/serve_llm.py backend switch.
$ErrorActionPreference = "Stop"
Set-Location (Split-Path -Parent $PSScriptRoot)

if (Test-Path ".venv\Scripts\python.exe") {
  $py = ".venv\Scripts\python.exe"
} else {
  $py = "python"
}

& $py scripts\serve_llm.py @args
exit $LASTEXITCODE