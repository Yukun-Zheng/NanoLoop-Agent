import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import { RunTimeline } from "@/components/runs/run-timeline";
import type { Run } from "@/lib/api/types";

describe("RunTimeline", () => {
  it("renders authoritative backend failure details and audit history", () => {
    const run = {
      run_id: "run-failed",
      job_id: "job-1",
      image_id: "image-1",
      model_id: "unet-large",
      status: "FAILED",
      roi_mode: "full_image",
      created_at: "2026-07-23T07:59:00Z",
      updated_at: "2026-07-23T08:00:00Z",
      error_code: "MODEL_ASSET_UNAVAILABLE",
      error_message: "checkpoint digest mismatch",
      inference: {
        device: "cpu",
        exclude_border: true,
        min_area_px: 8,
        seed: 42,
        threshold: 0.5,
        watershed_enabled: false
      },
      configuration: {
        analysis_roi: {
          coordinate_space: "original_px",
          revision: 1,
          schema_version: 1,
          source: "none",
          valid_rect: { x1: 0, y1: 0, x2: 128, y2: 128 },
          invalid_rects: []
        },
        created_at: "2026-07-23T07:59:00Z",
        inference: {
          device: "cpu",
          exclude_border: true,
          min_area_px: 8,
          seed: 42,
          threshold: 0.5,
          watershed_enabled: false
        },
        model_id: "unet-large",
        model_version: "1.0.0",
        postprocess_profile: "large-particle-v1",
        preprocess_profile: "grayscale-v1",
        provenance_status: "complete",
        provenance_warnings: [],
        review_source: "model_inference",
        roi_context_px: 16,
        roi_mode: "full_image",
        schema_version: 3
      },
      status_history: [
        {
          event_id: 1,
          from_status: "SEGMENTING",
          to_status: "FAILED",
          created_at: "2026-07-23T08:00:00Z",
          error_code: "MODEL_ASSET_UNAVAILABLE",
          error_message: "checkpoint digest mismatch"
        }
      ]
    } satisfies Run;

    render(<RunTimeline run={run} />);

    expect(screen.getByText("MODEL_ASSET_UNAVAILABLE")).toBeVisible();
    expect(screen.getByText("checkpoint digest mismatch")).toBeVisible();
    expect(screen.getByText("查看 1 条状态审计记录")).toBeVisible();
  });
});
