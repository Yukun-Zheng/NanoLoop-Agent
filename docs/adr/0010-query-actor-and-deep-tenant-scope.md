# ADR 0010: Attribute queries and repeat tenant scope at every data boundary

- Status: Accepted
- Date: 2026-07-18

## Context

ADR 0009 made the Analysis aggregate tenant-scoped, but the query use case still loaded jobs,
images, and runs through global identifiers. Its numeric data tool repeated those unscoped reads,
and `query_logs` recorded the question and answer without the authenticated actor. An HTTP-only
check would therefore leave a second data-access boundary able to cross tenants, while a successful
query could not later prove which principal, credential, role, or authentication mode issued it.

The knowledge corpus remains global. Until knowledge documents, FTS rows, and vector generations
carry an independently enforced tenant boundary, allowing a principal-mode knowledge or mixed
query would expose one tenant to another tenant's corpus even if the analysis job itself were
correctly scoped.

## Decision

The query route explicitly receives the middleware-cached `PrincipalContext`; it does not
reauthenticate. `QueryApplicationService` resolves the job, optional image, and requested runs with
SQL predicates or joins that include the principal's `tenant_id`, then applies the Analysis read
policy. Missing and cross-tenant identifiers both return `404 RESOURCE_NOT_FOUND`. Tenant
administrators, owning and peer analysts, and viewers may all read a same-tenant analysis and issue
an analysis-data query.

Tenant scope is deliberately repeated inside the numeric data-tool boundary. `DataQuery` carries a
required tenant ID, and `SqlAlchemyDataToolService` filters the job, optional image, and selected or
default runs through tenant-scoped SQL before it reads summaries or particle rows. It does not
trust route validation as proof for a later database session.

The database audit commit is authoritative. Immediately before inserting `query_logs`, the final
unit of work repeats the job policy and repository scope checks for the optional image and every
requested run. A `QueryActorDTO` is created only from the verified principal and freezes:

- tenant ID and principal ID;
- credential ID for principal mode;
- the role used by the request; and
- the effective authentication mode.

`query_logs` stores those five facts in required columns. Composite foreign keys prove that the
query job belongs to the actor tenant, the principal belongs to that tenant, and a non-null
credential belongs to that principal. Checks require a credential for principal mode, forbid one
for compatibility modes, and constrain disabled/shared-key/legacy migration actors to the fixed
legacy tenant administrator. The deepest runtime repository explicitly rejects the migration-only
`legacy_unknown` mode even when an internal caller constructs that DTO directly. Query-history and
RAG-citation files include the same actor DTO, but remain rebuildable
projections written only after the database commit; projection failure is logged as degradation and
does not roll back or disguise the committed query.

The migration refuses, before its first DDL statement, to guess an actor for any historical query
owned by a non-legacy tenant. Historical queries on valid legacy jobs are backfilled as the fixed
legacy administrator with `legacy_unknown`. Downgrade refuses before DDL if any attributable
runtime query exists, so real actor facts cannot be silently discarded. SQLite upgrades and
downgrades require a clean database-wide `PRAGMA foreign_key_check` before their first DDL statement
and repeat that check after all DDL, avoiding a partially rebuilt schema on SQLite's
non-transactional DDL path.

Principal mode fails closed with `503 SERVICE_UNAVAILABLE` and
`component=knowledge_tenant_scope` for explicit material-knowledge and mixed queries, and for AUTO
queries that classify to either path. This guard runs before FTS, vector retrieval, an answer
provider, database audit, or file projection. Disabled and shared-key compatibility modes retain
their existing global knowledge behavior for legacy deployments.

## Consequences

- Analysis-data queries have defense in depth: the HTTP/application boundary and the data tool each
  enforce tenant scope in their own database access path.
- Every newly committed query has a relationally constrained actor and an identical rebuildable
  projection; legacy historical rows remain explicitly distinguishable from attributable runtime
  rows.
- Query is a read operation, so same-tenant peer analysts and viewers remain allowed even though
  they cannot mutate another analyst's Analysis aggregate.
- Principal-mode material and mixed querying is intentionally unavailable until knowledge
  documents, keyword retrieval, and vector retrieval are tenant-scoped. A compatibility-mode
  success is not evidence that principal-mode knowledge isolation is complete.
- This decision does not bind download tokens to a tenant or principal, add knowledge ownership,
  implement quota or retention, or make multi-instance SQLite/rate-limit state safe. Those remain
  separate required batches before any public multi-tenant claim.
