# ADR 0007: Bounded two-stage principal rate limiting

- Status: Accepted
- Date: 2026-07-18
- Supersedes in part: the principal rate-limit decision in
  [ADR 0005](0005-principal-credentials-and-legacy-compatibility.md).

## Context

ADR 0005 initially placed every principal-mode request in one anonymous fixed bucket before the
credential lookup. The fixed state was safe from attacker-controlled memory growth, but any caller
could exhaust that bucket and prevent every valid principal from reaching authentication. Merely
classifying token-shaped input as authenticated would not solve the problem because token syntax
and credential IDs can be forged.

The replacement must protect the identity database, isolate authenticated principals, retain a
strict memory bound, perform exactly one credential lookup, and preserve disabled/shared-key
compatibility. It must also define which network address is trusted before a reverse proxy can
rewrite the ASGI peer.

## Decision

Disabled and shared-key modes retain the existing three fixed buckets: `service`, `authenticated`,
and `anonymous`. Principal mode uses two independent `BoundedKeyedTokenBucketLimiter` instances:

1. Before authentication, a request consumes a bucket keyed only by the normalized direct socket
   peer from `scope.client`. The application does not read `Forwarded` or `X-Forwarded-For`.
   IPv4-mapped IPv6 addresses are normalized to the same IPv4 key. Missing or invalid peers share
   the bounded `peer:unknown` key.
2. Authentication strictly parses the bearer token and performs the existing single indexed
   credential/principal/tenant query. Rejected credentials and authentication-backend failures do
   not consume authenticated principal capacity.
3. After successful authentication, the middleware consumes a bucket keyed by the verified
   `PrincipalContext.principal_id`. It reuses the authentication result and must not parse the
   header again or query the database a second time.

Each keyed limiter stores an `OrderedDict` protected by one lock. Access refreshes LRU order. A new
key at capacity evicts exactly the least-recently-used state before insertion, so retained state
never exceeds `API_RATE_LIMIT_MAX_BUCKETS` per limiter. Eviction is intentionally fail-open for the
evicted key: an attacker with many genuine source addresses may weaken pre-authentication rate
enforcement, but cannot grow memory without bound or force unrelated callers into a shared overflow
bucket. Raw bearer tokens are never limiter keys or log fields.

`API_RATE_LIMIT_REQUESTS` and `API_RATE_LIMIT_WINDOW_SECONDS` remain the compatibility fixed-bucket
settings and become the post-authentication per-principal settings in principal mode.
`API_PRINCIPAL_PREAUTH_RATE_LIMIT_REQUESTS` and
`API_PRINCIPAL_PREAUTH_RATE_LIMIT_WINDOW_SECONDS` configure the direct-peer stage independently; a
zero capacity disables that stage so a trusted ingress can own pre-authentication throttling.

Rate-limit response headers reflect the stage that controls the request. A single active layer is
authoritative and overwrites downstream `X-RateLimit-Limit` and `X-RateLimit-Remaining` values.
Only when principal mode has both an outer peer layer and an inner principal layer may the outer
layer preserve the inner layer's authoritative headers. Public health/documentation paths and
valid CORS preflights continue to bypass both layers; ordinary `OPTIONS` requests do not.

Bundled local and container Uvicorn commands explicitly use `--no-proxy-headers`. Consequently,
untrusted forwarding headers cannot change the application key before middleware runs. A future
trusted-proxy design requires a separate decision covering direct-peer trust, bounded header-chain
parsing, and ingress behavior; it must not silently re-enable Uvicorn's defaults.

## Consequences and boundaries

- One unauthenticated source cannot consume any post-authentication principal bucket, and sources
  with different direct peers have independent pre-authentication capacity.
- Multiple credentials for one principal share one post-authentication bucket; changing source IP
  does not multiply authenticated capacity.
- Clients behind the same NAT or direct reverse-proxy peer share the pre-authentication bucket. The
  pre-authentication capacity must therefore be higher than a typical single-principal allowance.
  Deployments needing real-client source limits must enforce them at a trusted ingress and may
  disable the application pre-authentication stage.
- LRU eviction favors availability and memory safety, not strict global abuse accounting. A large
  botnet can churn the bounded map and reduce pre-authentication effectiveness.
- Both limiters are process-local and reset on restart. Multiple Uvicorn workers or API replicas
  multiply effective capacity and do not share LRU state. Cluster-wide enforcement requires an
  ingress or shared rate-limit service.
- This mechanism is request throttling, not a billing, storage, scientific-compute, or tenant quota.
  It does not complete resource authorization or make the service safe for public multi-tenant use.
