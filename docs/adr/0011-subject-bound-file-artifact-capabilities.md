# ADR 0011: Subject-bound file artifact capabilities and pinned streaming

- Status: Accepted
- Date: 2026-07-18

## Context

The v1 file token signed only a managed relative path, expiry, and nonce. Possession was sufficient
to download the path, so a leaked token was not bound to an authenticated tenant or principal. The
same token type was also accepted as a corrected-mask input, which did not distinguish download from
one-shot review intent. Finally, the download route validated one path and then passed that path to
`FileResponse`, allowing a second open after validation.

## Decision

New public file references use four coordinated boundaries:

1. `file_artifacts` stores immutable job/image/run relationships, storage path, display filename,
   media type, SHA-256 and byte size. A path is registered idempotently only for exactly identical
   facts. Facts cannot be updated, and lifecycle is one-way from `active` to `consumed` or `revoked`.
2. A canonical `v2.<kid>.<payload>.<signature>` HMAC capability contains only version, tenant,
   principal, job, artifact ID, purpose, audience, hash, bounded times and a 128-bit token ID. It
   contains neither a filesystem path nor credential ID. Purpose/audience pairs are exact.
3. Resolution verifies signature, tenant, principal and endpoint audience before a tenant-scoped
   active registry lookup. It then opens every path component with descriptor-relative
   `O_NOFOLLOW`, hashes and sizes the final regular file on that descriptor, rechecks registry state,
   and streams that same descriptor. The HTTP layer never reopens the path and closes the pinned
   descriptor from the response lifecycle on normal completion, cancellation, or client disconnect.
4. A corrected-mask capability is bound to one parent job/image/run and the review audience. The
   final child-creation transaction consumes its registry row with an `active -> consumed`
   compare-and-swap before inserting the child run. A failed child transaction rolls the grant back;
   successful consumption prevents replay even if best-effort byte cleanup fails.

Signing material is a canonical protected JSON keyring. One active `kid` signs new tokens; retained
keys verify outstanding tokens during rotation. The keyring file must be a current-user-owned regular
file with mode `0600`. Development and tests may initialize a missing on-disk keyring; production
must start with stable material supplied by the runtime entrypoint or operator. Downloads use a
15-minute TTL and the runtime keyring rejects claims beyond one hour.

## Authorization and compatibility order

- Principal mode rejects every `v1.` token before decoding it or touching the filesystem.
- Disabled/shared-key compatibility may resolve v1 only after extracting the signed relative path
  without filesystem access and proving its first component is a legacy-owned database job.
- Legacy corrected-mask v1 is narrower: `job/input/review_mask_*/original[.*]`, exact authenticated
  parent binding, and lazy registry registration followed by the same one-shot database CAS.
- All newly returned original-image, run-artifact, export and corrected-mask tokens are v2 in every
  authentication mode.
- Authentication failures remain `401`; invalid/stale/cross-context downloads are a uniform file
  `404`; invalid review grants remain the generic `INVALID_IMAGE` corrected-mask error.

## Consequences

- A URL copied to another principal no longer works, even inside the same tenant. Clients refresh a
  URL through the authorized analysis/run/export endpoint after expiry or principal changes.
- Artifact paths are lazy-registered, so the migration does not scan the filesystem or invent facts
  for historical outputs. An existing path whose bytes or immutable metadata differ fails closed.
- Revocation prevents subsequent resolution; an already resolved in-flight descriptor may finish.
- Descriptor pinning prevents symlink and path-replacement races. Exact streamed bytes additionally
  rely on NanoLoop writers publishing registered artifacts by atomic replacement rather than
  modifying the registered inode in place.
- Same-process registration is serialized because the supported topology remains one API process.
  A future multi-process deployment must add database-native conflict retry/coordination before that
  topology is declared supported.
- The current operator flow performs single-writer, maintenance-window rotation followed by an API
  restart. It retains old keys but does not hot-reload, coordinate concurrent rotations, or retire
  keys; the bounded eight-key ring fails closed until a later audited retirement workflow exists.
- The frontend continues to treat download URLs and corrected-mask tokens as opaque strings; no
  frontend source change is required for this backend contract change.
