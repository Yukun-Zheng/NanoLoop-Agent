# ADR 0005: Principal credentials and legacy compatibility

- Status: Accepted
- Date: 2026-07-18

## Context

ADR 0004 added an optional shared deployment key for the trusted, single-process MVP. That key
answers only whether a caller knows one deployment secret. It cannot identify a person or service,
be revoked independently, express a tenant role, or establish ownership of an analysis, download,
knowledge document, or model operation.

The next backend slice needs stable identity values before database rows and middleware can share a
contract. It must also avoid a flag day: existing trusted deployments need explicit disabled and
shared-key compatibility modes while persisted principals are introduced. At the same time, this
compatibility must not be mistaken for completed multi-tenant authorization.

## Decision

Three authentication modes are represented in every principal context:

- `principal`: a persisted user or service with tenant, principal, and credential IDs;
- `shared_key`: the ADR 0004 deployment-key boundary; and
- `disabled`: trusted local compatibility with application authentication switched off.

A `PrincipalContext` is immutable. Principal mode requires canonical `tnt_`, `prn_`, and `crd_`
IDs, each followed by 32 lowercase hexadecimal characters. Compatibility contexts must use the
fixed `LEGACY_TENANT_ID` and `LEGACY_PRINCIPAL_ID`, a service kind, and the `tenant_admin` role;
arbitrary or partial identity shapes are rejected. These IDs make attribution deterministic; they
do not prove a real user or tenant. Compatibility contexts never claim a credential ID.

Tenant slugs are lowercase URL-safe labels of at most 63 characters. Principal handles are
lowercase, log-safe labels of at most 64 characters. IDs are generated from 128 bits of operating
system randomness and validated at contract boundaries.

Operator-provisioned bearer credentials have one strict representation:

```text
nlk_v1_crd_<32 lowercase hex>_<43 canonical base64url characters>
```

The final component is generated with `secrets.token_urlsafe(32)`. The complete token is returned
once as a `SecretStr`; normal object representations redact it. Persistence stores the credential
ID and an HMAC-SHA256 digest of the complete token under a separate deployment pepper. The pepper
must contain at least 32 bytes. Verification uses fixed-length digests and
`hmac.compare_digest`; malformed tokens and unknown credentials still execute a dummy
HMAC/comparison path and yield the same false result. Parser and configuration errors do not echo
the candidate token or pepper.

The raw token and pepper have different recovery boundaries:

- a raw token is not recoverable from its database digest and must be shown only at issuance;
- losing a raw token requires issuing another credential and revoking the old record;
- losing the pepper invalidates all stored credential digests even if the database is restored;
- restoring a pepper restores the ability to verify every still-active digest, so the pepper must
  be backed up, access-controlled, and rotated separately from the database; and
- backup archives, logs, exception details, API responses after issuance, and source control must
  never contain either secret.

The operator CLI writes a newly issued token before attempting the database commit, using
`O_CREAT | O_EXCL | O_NOFOLLOW`, mode `0600`, a complete-write loop, and file/directory `fsync`.
Portable POSIX APIs cannot atomically say “unlink this name only if it still references this open
inode.” On any uncommitted failure the CLI therefore truncates and fsyncs only the original open
file descriptor and never unlinks a raceable path name. If the original directory entry still
exists, a private zero-byte placeholder remains for the operator to remove explicitly; a
concurrently replaced directory entry is never deleted or truncated. If commit completion is
uncertain after the row was staged, the token is destroyed and the safe credential ID is returned
with an explicit list-then-revoke recovery action. This avoids leaving a usable unknown credential,
but can leave an unusable row that requires operator compensation.

Mode selection is an explicit deployment setting. A failed principal credential must never fall
back automatically to the shared key or disabled mode. Moving from compatibility to principal mode
requires provisioning identities, assigning legacy resource ownership, and changing the setting
only after middleware and resource checks are ready. Emergency rollback is an operator decision at
the deployment boundary; it reopens the limitations of ADR 0004 and must not silently relabel
legacy activity as individually authenticated activity.

Principal-mode requests remain in the anonymous pre-authentication rate-limit bucket. Token syntax
is forgeable and must not be used to consume the capacity reserved for verified callers. A future
two-stage limiter may add bounded per-principal buckets after the single credential query succeeds;
it must not perform a second authentication lookup.

## Consequences

- Database, middleware, logs, and services can share stable enum, ID, and principal-context shapes.
- Credentials can be individually stored, rotated, and revoked once the persistence slice consumes
  these primitives; plaintext recovery is intentionally impossible.
- Compatibility requests have deterministic attribution without inventing a user identity.
- The versioned token prefix leaves room for a future parsing or hashing migration without
  accepting ambiguous formats.
- The dummy verification path reduces credential-oracle differences inside this process, but it
  does not make database lookup, HTTP handling, or end-to-end response timing constant.
- Identity lifecycle audit rows are guarded against ORM mutation and, on SQLite, direct update or
  delete triggers. This is application-database integrity, not an external tamper-evident ledger.

This slice implements operator provisioning, credential persistence/revocation, request
authentication, principal log context, and identity lifecycle audit. It does **not** implement
interactive login, resource ownership, role-policy enforcement, tenant-scoped queries/downloads,
quotas, or retention. Until those slices are connected and tested, principal authentication alone
does not make the service safe for public or multi-tenant deployment.
