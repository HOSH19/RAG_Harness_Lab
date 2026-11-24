# Native (non-Docker) vLLM serve fallback for Windows.
# Prefer: docker compose --env-file .env up vllm

$ErrorActionPreference = "Stop"

if (Test-Path ".env") {
  Get-Content ".env" | ForEach-Object {
    if ($_ -match '^\s*#' -or $_ -match '^\s*$') { return }
    $name, $value = $_.Split('=', 2)
    [Environment]::SetEnvironmentVariable($name.Trim(), $value.Trim(), "Process")
  }
}

$Model = if ($env:MODEL_ID) { $env:MODEL_ID } else { "Qwen/Qwen2.5-7B-Instruct" }
$Port = if ($env:VLLM_PORT) { $env:VLLM_PORT } else { "8000" }
$MaxLen = if ($env:MAX_MODEL_LEN) { $env:MAX_MODEL_LEN } else { "4096" }
$GpuUtil = if ($env:GPU_MEMORY_UTILIZATION) { $env:GPU_MEMORY_UTILIZATION } else { "0.85" }

Write-Host "Starting vLLM: model=$Model port=$Port max_model_len=$MaxLen"

vllm serve $Model `
  --host 0.0.0.0 `
  --port $Port `
  --max-model-len $MaxLen `
  --gpu-memory-utilization $GpuUtil `
  --dtype auto