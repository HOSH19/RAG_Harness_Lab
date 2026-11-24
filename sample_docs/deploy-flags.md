# AcmeOps Deploy Flags

The deploy CLI (`acmeops deploy`) accepts these flags:

| Flag | Default | Meaning |
| --- | --- | --- |
| `--env` | `staging` | Target environment: `staging`, `prod`, or `dr` |
| `--canary-percent` | `10` | Percentage of traffic for canary before full rollout |
| `--max-batch-tokens` | `2048` | Cap for batched inference jobs in the worker pool |
| `--skip-migrations` | `false` | Dangerous; only for emergency hotfixes |
| `--region` | `us-east-1` | Primary region for the release |

## Recommended production command

```bash
acmeops deploy --env prod --canary-percent 5 --region us-east-1
```

Never use `--skip-migrations` in `prod` unless an incident commander approves it in the #incidents channel.

## Region notes

- `us-east-1` is the default primary.
- `eu-west-1` requires the `EU_DATA_RESIDENCY=1` feature flag.
- `ap-southeast-1` is read-replica only and cannot accept writes.