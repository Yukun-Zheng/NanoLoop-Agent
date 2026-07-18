# ADR 0004: Shared API key and single-process rate limit

- Status: Accepted
- Date: 2026-07-18

## Context

NanoLoop Agent currently supports one API process on a trusted host. Host and browser-origin
checks reduce request forgery risk, but they do not establish a caller identity. A public user,
tenant, role, distributed quota, or external identity provider would require a larger data and
operations model than the current single-host deployment can honestly support.

The immediate need is a bounded service-level gate that works consistently for JSON, multipart,
file downloads, Streamlit, and the black-box smoke client without making external identity or
Redis infrastructure mandatory.

## Decision

When `NANOLOOP_API_KEY` is configured, every versioned API operation, including file downloads and
`/api/v1/health`, requires exactly one matching `X-API-Key` header. The comparison hashes the
candidate and configured secret to fixed-length SHA-256 values before constant-time comparison.
Missing, incorrect, and duplicate headers return the same `401 AUTHENTICATION_REQUIRED` envelope.

Exact unauthenticated paths are limited to the root `/health` probe and FastAPI documentation
paths. Valid CORS preflight is answered by the outer `CORSMiddleware` before authentication;
plain `OPTIONS` requests are still authenticated and rate limited. Similar prefixes are not exempt.

An optional in-process token bucket uses only three fixed caller classes:

- `authenticated`: the one valid shared key;
- `anonymous`: missing, duplicate, or invalid keys while authentication is enabled;
- `service`: all requests while authentication is disabled.

The limiter is placed before authentication, while the authenticated and anonymous buckets remain
separate. It uses a monotonic clock and a lock, and returns `429 RATE_LIMITED`, `Retry-After`, and
bounded rate headers. The middleware order remains:

```text
RequestContext
-> TrustedHost
-> BrowserMutationGuard
-> CORS
-> InMemoryRateLimit
-> ApiKeyAuth
-> RequestBodyLimit
-> Router
```

This ensures hostile Host and cross-site mutation failures occur before key validation, allowed
browser origins receive CORS headers on `401`/`429`, and unauthenticated large bodies are rejected
before parsing.

When the key is present in Streamlit, the configured backend destination is locked to the
normalized `NANOLOOP_API_BASE_URL`. A session-controlled destination mismatch fails before the
client receives the secret. Settings validation also hides raw inputs in rendered validation
errors so an invalid real key is not echoed into startup logs.

## Consequences

- Existing local development remains compatible when no key is configured.
- Streamlit and the smoke client read the same optional key and attach it to every trusted backend
  request. Smoke automation should prefer the environment variable because CLI arguments can be
  visible in shell history and process listings.
- OpenAPI declares `ApiKeyAuth`, and all versioned operations document `401`, `403`, and `429`.
- All legitimate callers share one identity and one bucket; one caller can exhaust the shared
  quota for the others.
- Counters reset on process restart and are not coordinated across workers or replicas. This is
  acceptable only because the supported topology remains one API process.
- `X-Forwarded-For` is not trusted. A reverse proxy must still provide TLS, connection controls,
  user authentication/authorization, edge rate limiting, and access audit for public deployment.
- Environment variables are visible to the host administrator and may be visible through container
  inspection. Secret-file support and overlapping-key rotation remain future work.
- The public root health endpoint is a readiness-style dependency check, not a cheap liveness-only
  probe. Public deployment must split or protect it before exposing the service beyond the supported
  loopback/trusted-host topology.

This decision does not change the conclusion that public multi-tenant deployment is unsupported.
