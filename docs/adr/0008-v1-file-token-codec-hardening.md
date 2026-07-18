# ADR 0008: Harden the existing v1 file-token codec without changing its wire format

- Status: Accepted
- Date: 2026-07-18

## Context

NanoLoop uses compact HMAC-SHA256 file tokens to address managed local files. The deployed v1 wire
format is `v1.<payload>.<signature>`, where the payload is compact JSON and both binary segments use
unpadded base64url. Existing canonical tokens within the codec's supported Unix timestamp range and
the stable signing secret must remain usable.

The original decoder verified the HMAC, expiry and managed path, but accepted more representations
than the issuer produced. In particular, Python booleans can pass ordinary integer checks, padded
or non-canonical base64 encodings can decode to canonical bytes, unknown JSON keys were ignored,
and an explicit zero TTL fell back to the default because it was treated as a generic false value.
Those ambiguities make validation and future security review unnecessarily difficult.

## Decision

The v1 prefix, payload fields, HMAC algorithm and signing input remain unchanged. Issuance and
verification now apply one strict codec contract:

- only `ttl_seconds=None` selects the configured default; zero, negative, boolean and non-integer
  TTLs are rejected;
- each store has a positive `max_token_ttl_seconds`, defaulting to 86,400 seconds, and its default
  TTL and every explicitly issued TTL must not exceed that bound;
- caller-supplied `now` and payload `exp` are exact integers, never booleans, within the supported
  non-negative Unix timestamp range; version is the exact integer `1`;
- payload keys are exactly `exp`, `nonce`, `path` and `v`, with no missing, extra or duplicate keys;
- payload JSON is the same compact, sorted representation emitted by the issuer;
- payload and signature segments are unpadded canonical base64url, proven by decoding and
  re-encoding; signatures decode to exactly one SHA-256 digest;
- nonce is the canonical unpadded base64url encoding of exactly eight random bytes;
- path is a non-empty, bounded, canonical relative POSIX path without control characters,
  traversal components, redundant separators or Windows separators; the existing managed-path and
  regular-file checks still make the final storage decision;
- issuance reuses that path validation before signing, proves that the complete emitted token fits
  the same 4,096-character limit enforced by the parser, and preflights upload/export destinations
  before publishing bytes that would require an unrepresentable token; and
- token failures expose only fixed domain messages. They do not echo the token or payload path and
  do not retain input-bearing decoder or filesystem exceptions as public causes.

The configured maximum limits new issuance. Verification does not infer a historical TTL from
`exp`, because v1 has no issued-at field. This preserves already-issued canonical v1 tokens,
including tokens created under an earlier, longer configured TTL, until their signed expiry, as
long as `exp` is no later than Unix timestamp `253402300799` (the end of year 9999 UTC). A historical
canonical token with a larger integer `exp` is intentionally rejected by the new bounded parser.

## Consequences

- Existing canonical v1 tokens inside the documented timestamp and shape bounds remain valid with
  the same signing secret and require no migration; extreme `exp` values beyond year 9999 do not.
- Ambiguous encodings and payload extensions that the NanoLoop issuer never generated now fail
  closed even when their HMAC is otherwise valid.
- The default 86,400-second maximum continues to permit the existing 3,600-second integration
  tokens while placing an explicit bound on new issuance.
- Operators rotating the file-token secret still invalidate outstanding tokens, as before.
- This change does not add artifact identity, tenant/job ownership, token purpose, revocation or a
  v2 format. It also does not solve the separate path-open/FileResponse time-of-check-to-time-of-use
  boundary; those require a separately designed versioned protocol and download path.
