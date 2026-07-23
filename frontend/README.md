# NanoLoop Command Center

This directory is the Next.js replacement for the retired Python UI. It is a server-rendered
frontend/BFF, not a second scientific backend: FastAPI remains authoritative for analyses, ROI
revisions, model runs, statistics, quality, queries, knowledge documents and exports.

## Runtime

- Node.js 24
- pnpm 10 through Corepack
- Next.js 16, React 19, TypeScript 5 and Tailwind CSS 4
- TanStack Query, Zod, Zustand and React-Konva

Install and run from the repository root:

```bash
make frontend-install
make frontend
```

Open `http://127.0.0.1:3000`. The user-facing routes are:

- `/` — create or reopen an analysis;
- `/workspace/{job_id}` — project, ROI, models/runs, timeline, results/review/export and Agent query;
- `/knowledge` — knowledge ingestion, status and reindex actions.

## Server-only API boundary

The browser calls only same-origin `/api/nanoloop/*`. The route handler applies a strict path/method
allowlist and maps allowed requests to FastAPI `/api/v1` using:

```dotenv
NANOLOOP_API_INTERNAL_URL=http://127.0.0.1:8000
NANOLOOP_API_KEY=
NANOLOOP_FRONTEND_ALLOWED_ORIGINS=http://127.0.0.1:3000,http://localhost:3000
```

Both variables are server-only. Never expose them through `NEXT_PUBLIC_*`. Browser-provided Cookie,
Authorization and API Key headers are not forwarded. Signed artifact URLs are accepted only in the
FastAPI `/api/v1/files/{token}` form and are converted to a same-origin download route.

## Verification

```bash
make frontend-check
make frontend-e2e
```

`pnpm check` audits production dependencies, regenerates API types from
`../docs/api/openapi-v1.json`, checks drift, lints, type-checks, runs Vitest and builds the
standalone production bundle. Playwright currently uses same-origin API mocks for the scientific
workflow, ROI CAS recovery, responsive inspector, and knowledge lifecycle. It does not prove a live
FastAPI deployment, real model accuracy or a production RAG corpus. Those require separate
target-environment acceptance.
