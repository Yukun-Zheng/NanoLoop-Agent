import { beforeEach, describe, expect, it, vi } from "vitest";

import { apiRequest } from "@/lib/api/client";
import type { ImageAsset, ModelList } from "@/lib/api/types";
import {
  recommendModelsForImages,
  runAssignmentPayload
} from "@/lib/models/recommendation";

vi.mock("@/lib/api/client", () => ({
  apiRequest: vi.fn()
}));

const images = [
  { image_id: "img_small", filename: "small.tif" },
  { image_id: "img_dense", filename: "dense.tif" }
] as ImageAsset[];

const models = [
  {
    model_id: "model-small",
    status: "ready",
    health_error: null
  },
  {
    model_id: "model-dense",
    status: "ready",
    health_error: null
  }
] as ModelList["models"];

describe("per-image model recommendation", () => {
  beforeEach(() => {
    vi.mocked(apiRequest).mockReset();
  });

  it("keeps an independent model assignment for every image", async () => {
    vi.mocked(apiRequest).mockImplementation(async (_path, options) => {
      const imageId = (options?.body as { image_id: string }).image_id;
      const modelId = imageId === "img_small" ? "model-small" : "model-dense";
      return {
        request_id: "req_test",
        status: "success" as const,
        data: {
          candidates: [{ model_id: modelId, score: 0.9, reasons: ["profile match"] }],
          requires_user_confirmation: true
        },
        error: null
      };
    });

    const assignments = await recommendModelsForImages({
      images,
      models,
      roiMode: "full_image",
      prefer: "accuracy",
      device: "auto"
    });

    expect(apiRequest).toHaveBeenCalledTimes(2);
    expect(assignments.map((item) => [item.imageId, item.modelId])).toEqual([
      ["img_small", "model-small"],
      ["img_dense", "model-dense"]
    ]);
    expect(runAssignmentPayload(assignments)).toEqual({
      model_ids: ["model-small", "model-dense"],
      model_assignments: {
        img_small: "model-small",
        img_dense: "model-dense"
      }
    });
  });

  it("falls back per image without discarding successful recommendations", async () => {
    vi.mocked(apiRequest)
      .mockResolvedValueOnce({
        request_id: "req_ok",
        status: "success",
        data: {
          candidates: [
            { model_id: "model-dense", score: 0.8, reasons: ["profile match"] }
          ],
          requires_user_confirmation: true
        },
        error: null
      })
      .mockRejectedValueOnce(new Error("recommendation unavailable"));

    const assignments = await recommendModelsForImages({
      images,
      models,
      roiMode: "full_image",
      prefer: "accuracy",
      device: "auto"
    });

    expect(assignments[0]).toMatchObject({
      imageId: "img_small",
      modelId: "model-dense",
      usedFallback: false
    });
    expect(assignments[1]).toMatchObject({
      imageId: "img_dense",
      modelId: "model-small",
      usedFallback: true
    });
  });
});
