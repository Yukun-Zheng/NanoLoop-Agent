# ADR-0002: Minimal resolutions for v2.0 contract ambiguities

- Status: accepted for implementation; scientific terminology remains subject to domain review
- Date: 2026-07-18
- Decision owner: integration

## Context

The v2.0 specification is the source of truth but leaves several cross-layer details ambiguous or
internally inconsistent. Implementing different interpretations would break OpenAPI, persistence,
and frontend handoff. This ADR makes the smallest decisions required for a runnable baseline.

## Decisions

1. Every public route uses `ApiResponse` with request ID, status, data, and error. Examples showing
   bare payloads are treated as abbreviated `data` examples. `INSUFFICIENT_EVIDENCE` is a successful
   HTTP 200 business outcome with citations/limitations and no API error.
2. The canonical health route is `/api/v1/health`; `/health` is a compatibility alias for Docker and
   smoke probes. Run and opaque file-download routes are added at `/api/v1/runs/{run_id}` and
   `/api/v1/files/{token}` because both are required elsewhere in v2.0.
3. `JobStatus` is reused for job and run machines, matching the document. Individual run status is
   authoritative. A job aggregates its runs: active work reports the least-advanced active stage;
   all-success is completed; any warning or mixed success/failure is completed-with-warnings; all
   failed is failed. Creating a new immutable run may requeue a terminal job without modifying old
   runs. Run state transitions themselves remain strictly forward-only.
4. Each image persists `box_revision`, including an empty set, to make optimistic locking possible.
   A run stores box revision plus an immutable box/config snapshot in its run configuration.
   Because a batch can contain multiple images, create-runs uses `box_revisions` keyed by image ID;
   the document's singular `box_revision` shorthand is accepted only for one-image requests. Runs
   are created for the image/model Cartesian product.
5. Backend box validation is strict. Out-of-bounds, undersized, or invalid-region overlap returns
   `INVALID_BOX`; the displayed-coordinate rounding formula is not permission to silently repair a
   submitted rectangle.
6. Images expose a versioned `analysis_roi` snapshot in `original_px`: one axis-aligned valid
   rectangle, zero or more invalid rectangles with reasons, source (`none/manual/detected`), and a
   revision. The effective area is the valid rectangle minus invalid rectangles, further intersected
   with the saved box union in boxes mode. A run freezes this snapshot in its configuration.
7. Physical scale is optional. Upload uses `nm_per_pixel` with a positive value or `pixel_only` with
   no value. Pixel metrics remain valid without scale; operations explicitly requesting physical
   units use `MISSING_SCALE`.
8. Summary machine names use `mean_equivalent_diameter_px/nm`; coverage is always a ratio in `[0,1]`
   and the UI may display percent. Perimeter-density fields are retained but labeled provisional
   until scientific review confirms terminology.
9. YAML is the authored model registry and the database is its queryable runtime mirror. Startup or
   an explicit seed command validates artifact/config/card health before syncing. Missing artifacts
   always force `unavailable` regardless of the authored status.
10. Model IDs use `{family-slug}-{variant}-{quality-tier}-v{major}`, with slugs `unet`, `yolo`, and
    `sam2`; SAM2 box support is metadata, not a variant named `box`.
11. Hybrid retrieval normalizes fused RRF scores to `[0,1]` before applying the documented `0.20`
    threshold. SQLite FTS remains available without embeddings; vector retrieval reports degraded
    health until the optional model/index exists.
12. Review uses JSON parameters and an optional previously uploaded `corrected_mask_token`; it
    always creates a child run. The original run remains immutable.

## Consequences

These fields and routes must appear consistently in Pydantic schemas, the initial migration,
OpenAPI, fixtures, and frontend code. The choices are intentionally reversible through a later ADR,
but existing runs and exports remain readable by schema version.
