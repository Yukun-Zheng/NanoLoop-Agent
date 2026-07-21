# ADR 0009: Authorize the analysis aggregate at tenant-scoped repository boundaries

- Status: Accepted
- Date: 2026-07-18

## Context

ADR 0005 established authenticated tenant and principal identities. ADR 0006 then made every
analysis job carry a required tenant and owner, but deliberately left reads and mutations
unscoped. A caller that knew another job, image, or run identifier could therefore still address
that aggregate. Applying a role check only in an HTTP route would also leave application services
and child-resource queries able to bypass the tenant boundary.

Images, ROI revisions, runs, run results, query audit rows, and exports inherit their scope from
`analysis_jobs`. That inheritance is safe only if a run cannot reference an image from another job,
a review child cannot reference a parent from another job, and a query image cannot disagree with
its job.

## Decision

Analysis authorization uses one fixed evaluation order:

1. a repository query includes the authenticated `tenant_id` in SQL and resolves an
   `AnalysisResourceScope` containing the job, tenant, and owner;
2. a missing row and a row owned by another tenant both raise `404 RESOURCE_NOT_FOUND`;
3. only after the tenant-scoped lookup succeeds does the pure policy evaluate role and ownership;
4. a same-tenant caller with insufficient permission receives `403 FORBIDDEN`.

The role matrix is:

| Operation | tenant_admin | owning analyst | peer analyst | viewer |
|---|---:|---:|---:|---:|
| Create analysis | allow | allow | allow | deny |
| Read analysis/image/boxes/run/export | allow | allow | allow | allow |
| Replace boxes/create run/review/corrected mask | allow | allow | deny | deny |

Disabled and shared-key modes do not bypass this policy. Their middleware-verified principal is
the fixed legacy tenant administrator, so they can address legacy-owned jobs and receive the same
404 for jobs outside the legacy tenant.

HTTP-facing code uses explicit scoped repository methods for job, image, box, run, artifact-path,
and export-query reads. Child identifiers are filtered with joins to `analysis_jobs`; code does not
load a global row and compare its tenant afterward. Existing unscoped methods remain only for
trusted dispatcher, recovery, and execution-worker paths whose work is selected from durable
server state rather than a caller-supplied tenant claim.

Mutation services receive the middleware-cached `PrincipalContext` explicitly. Box replacement
and the final run/review writes authorize inside the same unit of work as the mutation. Run
creation performs an early scoped check before model discovery, health probes, or bundle freezing,
then repeats the check in the write transaction. Review creation checks the parent both before
provider/file work and again before inserting the child. Corrected-mask staging checks access
before reading the upload stream or writing a file. Reusing the request principal never performs a
second credential lookup.

The relationship migration fails before its first DDL statement if it finds any of these legacy
states:

- a run whose image belongs to a different job or is missing;
- a query whose non-null image belongs to a different job or is missing; or
- a review child whose parent belongs to a different job or is missing.

It then adds composite image/job and run/job uniqueness plus foreign keys proving run-image,
review-parent, and query-image job agreement. Existing run-image cascade behavior and optional
parent/query-image `SET NULL` behavior remain intact. SQLite upgrades and downgrades end with a
database-wide `PRAGMA foreign_key_check`.

## Consequences

- Analysis aggregate reads and scientific workflow mutations have a consistent tenant, role, and
  owner policy at both HTTP and persistence boundaries.
- Cross-tenant and nonexistent identifiers have the same public status and error code; callers
  cannot use role-dependent 403 responses to enumerate foreign resources.
- Viewers remain able to inspect same-tenant results but cannot create or mutate scientific state.
- Tenant administrators can operate any analysis in their tenant, while analysts can mutate only
  analyses they own.
- Internal workers retain unscoped methods, so queued execution and startup recovery do not invent
  a user principal or silently impersonate the legacy tenant.
- This ADR is not a complete multi-tenant-production claim. Query actor attribution and deep
  query/data-tool scoping, tenant/job-bound file-token v2 downloads, and knowledge-document tenant
  ownership remain separate required batches. Existing v1 download tokens are still bearer
  capabilities and must not be treated as proof of tenant authorization.
