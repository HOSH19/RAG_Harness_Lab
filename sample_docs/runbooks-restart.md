# Runbook: Pod Keeps Restarting

Symptoms often look like `CrashLoopBackOff` in Kubernetes.

## Quick checks

1. `kubectl describe pod <name>` — look for OOMKilled vs Error
2. Confirm `GPU_MEMORY_UTILIZATION` is <= `0.90`
3. Confirm `MAX_MODEL_LEN` is not larger than VRAM allows
4. Check recent deploys for `--skip-migrations true`

## OOM path

If the container was OOMKilled:

- Reduce `--max-batch-tokens` from 2048 toward 1024
- Or switch to `Qwen/Qwen2.5-3B-Instruct`
- Or lower `gpu_memory_utilization` to `0.75`

## Auth path

If logs show `token_expired` or `AO-1002`, rotate credentials and remount the secret `acmeops-api-token`.

## Escalation

Page `@oncall-platform` if restarts continue for more than 15 minutes after the above steps.