import { describe, expect, it } from "vitest";

import { isModelSelectable } from "@/components/models/model-selector";
import type { ModelMetadata } from "@/lib/api/types";

const base: ModelMetadata = {
  model_id: "fixture",
  family: "unet",
  variant: "general",
  quality_tier: "accurate",
  version: "1.0.0",
  status: "ready",
  supports_box_prompt: true,
  preprocess_profile: "fixture",
  postprocess_profile: "fixture",
  inference_invalid_bottom_px: 0,
  notes: ""
};

describe("model selectability", () => {
  it("requires ready health and compatible ROI support", () => {
    expect(isModelSelectable(base, "full_image")).toBe(true);
    expect(isModelSelectable({ ...base, status: "unavailable" }, "full_image")).toBe(false);
    expect(isModelSelectable({ ...base, health_error: "checksum mismatch" }, "full_image")).toBe(
      false
    );
    expect(isModelSelectable({ ...base, supports_box_prompt: false }, "boxes")).toBe(false);
  });
});
