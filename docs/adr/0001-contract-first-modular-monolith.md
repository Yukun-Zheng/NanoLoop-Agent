# ADR-0001: Contract-first modular monolith

- Status: accepted
- Date: 2026-07-18
- Decision owner: integration

## Context

NanoLoop Agent must support parallel work on platform APIs, analysis, model adapters, retrieval,
and a later frontend while remaining demonstrable on one workstation. Model weights and the final
knowledge corpus are not yet available.

## Decision

Build one Python application with strict module boundaries and typed contracts. FastAPI is the HTTP
boundary, SQLAlchemy/Alembic own queryable state, and a filesystem store owns large artifacts.
Analysis, inference, and retrieval communicate through DTOs and Protocols rather than importing
route or ORM internals. SQLite runs in WAL mode for the MVP. Background execution stays in-process
behind an execution abstraction so a durable queue can replace it later without changing APIs.

Unavailable heavyweight integrations expose truthful health state. Deterministic fakes are allowed
only through dependency injection in tests and fixtures, never as production-ready model entries.

## Consequences

- Public contract changes require an explicit ADR plus synchronized schema, migration, OpenAPI, and
  fixture updates.
- Runtime artifacts remain outside the database and are addressed through opaque download tokens.
- The application can start without model weights, a GPU, an LLM key, or a vector index.
- A future worker service, PostgreSQL database, React UI, or equipment-control provider can replace
  an adapter without rewriting domain services.
