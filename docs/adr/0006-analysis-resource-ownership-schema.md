# ADR 0006: Analysis resource ownership schema before authorization enforcement

- Status: Accepted
- Date: 2026-07-18

## Context

ADR 0005 established authenticated tenant and principal identities but deliberately did not claim
resource authorization. Existing NanoLoop databases already contain analysis jobs created before
those identities existed, and disabled/shared-key deployments must continue to open those jobs
without a flag-day data migration outside Alembic.

The analysis domain is an aggregate: an `AnalysisJob` owns images, ROI revisions, segmentation
runs, particles, summaries, query logs, and generated files. Repeating tenant and owner columns on
every child would create mismatch states without improving the first authorization boundary.

## Decision

`analysis_jobs` stores two required ownership values:

- `tenant_id` identifies the tenant that may address the aggregate; and
- `owner_principal_id` identifies the principal that created or owns the mutable workflow.

A composite foreign key from `(owner_principal_id, tenant_id)` to
`principals(principal_id, tenant_id)` proves at the database boundary that the owner belongs to the
tenant. The referenced principal tuple is explicitly unique. Tenant/creation and
tenant/owner/creation indexes support the next slice's scoped listing and point checks.

Every job that predates this revision is assigned the fixed legacy tenant and service principal
from ADR 0005. Before any schema mutation, the migration verifies that both fixed rows exist and
that the legacy principal still belongs to the legacy tenant with `service` kind and
`tenant_admin` role. SQLite upgrades finish with a database-wide `PRAGMA foreign_key_check`; any
violation aborts the migration instead of leaving a deployable-looking but inconsistent schema.

There are no ORM or server defaults for ownership. Every creation call must explicitly provide a
validated tenant and principal ID. The HTTP route passes the middleware-verified, request-cached
`PrincipalContext` to the application service: disabled/shared-key modes naturally carry ADR
0005's fixed compatibility principal, while principal mode persists the real authenticated owner.
The repository validates the canonical ID shapes and the composite foreign key validates tenant
membership without a second authentication query.

Images, ROI rows and revisions, runs and status events, particle rows, image summaries, query logs,
and analysis files inherit their resource scope from the owning job. Direct child-resource routes
must eventually join or otherwise validate that aggregate instead of treating an opaque child ID
as authority.

This revision adds creation-time attribution but does not enable route authorization, owner-only
mutation, tenant-scoped repository reads, or a multi-tenant production-readiness claim. Until
those checks land, principal authentication remains insufficient for public multi-tenant
deployment.

Downgrade is deliberately fail-closed. Ownership columns may be removed only while every job is
still owned by the fixed legacy identity. If any non-legacy tenant or owner exists, the downgrade
raises before dropping indexes, constraints, or columns; operators must explicitly migrate or
remove those resources before accepting the lossy schema transition.

The follow-up policy is frozen as follows:

- a missing resource and a resource owned by another tenant both return `404 RESOURCE_NOT_FOUND`,
  so identifiers cannot be used for tenant enumeration;
- a verified caller in the same tenant whose role or ownership is insufficient receives
  `403 FORBIDDEN`;
- tenant administrators may manage analysis aggregates in their tenant;
- analysts may create jobs and mutate their own jobs; and
- viewers are read-only.

The next implementation batches must add, in order:

1. tenant-scoped analysis repositories and same-transaction route/application authorization;
2. query actor attribution and query/data-tool scope enforcement;
3. tenant/job-bound file-token issuance and authenticated download checks; and
4. independently migrated knowledge-document ownership plus tenant-scoped FTS/vector retrieval.

## Consequences

- Existing jobs have deterministic, queryable ownership after migration and legacy-only data can
  survive a full downgrade/upgrade round trip.
- The database rejects a principal/tenant mismatch even if application validation is bypassed.
- Child tables keep their current v2-compatible shape and derive scope from one authoritative
  aggregate root.
- Disabled and shared-key deployments explicitly create legacy-owned jobs.
- Principal-mode creation records the real verified tenant and principal without re-authentication.
- Non-legacy ownership makes downgrade unavailable until an operator resolves the data explicitly.
- Creation attribution alone must not be presented as completed authorization.
- Knowledge documents, query actors, and file downloads remain explicit follow-up security work.
