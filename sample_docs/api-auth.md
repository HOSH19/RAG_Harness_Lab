# AcmeOps Authentication API

AcmeOps uses bearer tokens for all private REST endpoints.

## Obtain a token

`POST /v1/auth/token`

Required JSON body fields:

- `client_id` — issued by Platform Ops
- `client_secret` — rotate every 90 days
- `grant_type` — must be `client_credentials`

Successful responses return:

```json
{
  "access_token": "...",
  "token_type": "Bearer",
  "expires_in": 3600
}
```

## Using the token

Send `Authorization: Bearer <access_token>` on every request.
Tokens expire after **3600 seconds**. Refresh by requesting a new token; there is no refresh_token grant.

## Common errors

- `invalid_client` — bad `client_id` / `client_secret`
- `unauthorized_scope` — token missing required scope `ops.read` or `ops.write`
- `token_expired` — reuse after expiry; mint a new token