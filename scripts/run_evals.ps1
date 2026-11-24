$ErrorActionPreference = "Stop"

$suite = if ($args.Count -ge 1) { $args[0] } else { "smoke" }
$stamp = Get-Date -Format "yyyyMMdd-HHmmss"
$out = "results/${suite}-${stamp}.json"

New-Item -ItemType Directory -Force -Path results | Out-Null
python -m harness.runner --suite $suite --out $out
Write-Host "Wrote $out"