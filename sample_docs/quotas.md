# AcmeOps Quotas

Per-workspace soft limits:

- **Chat requests:** 120 requests / minute
- **Ingest jobs:** 5 concurrent
- **Stored chunks:** 250,000
- **Max upload size:** 25 MB per file
- **Max context passages:** 8 chunks per `/chat` call

Hard limit: exceeding 120 rpm returns HTTP 429 with header `Retry-After`.

Enterprise workspaces can raise chat RPM to 600 by filing ticket type `quota-increase` in the portal.