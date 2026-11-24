# AcmeOps Error Codes

| Code | HTTP | Meaning | Remediation |
| --- | --- | --- | --- |
| `AO-1001` | 400 | Malformed JSON body | Validate request schema |
| `AO-1002` | 401 | Missing Authorization header | Attach Bearer token |
| `AO-1003` | 403 | Insufficient scope | Request `ops.write` |
| `AO-2001` | 409 | Deploy already in progress | Wait or cancel with `acmeops deploy cancel` |
| `AO-2002` | 422 | `--max-batch-tokens` above 8192 | Lower the flag value |
| `AO-3001` | 503 | Worker pool saturated | Retry with backoff; page on-call if >15m |
| `invalid_upstream_json` | 502 | Upstream model gateway returned non-JSON | Check vLLM health endpoint |

## Severity

- `AO-1xxx` — client errors, usually safe to retry after fixing the request
- `AO-2xxx` — deploy pipeline conflicts
- `AO-3xxx` — platform capacity issues