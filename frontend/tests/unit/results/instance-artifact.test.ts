import { describe, expect, it } from "vitest";

import { parseInstanceArtifact } from "@/lib/results/instance-artifact";

describe("instance artifact", () => {
  it("maps authoritative bounding boxes to stable image percentages", () => {
    const artifact = parseInstanceArtifact({
      coordinate_space: "original_px",
      width: 200,
      height: 100,
      instances: [
        {
          instance_index: 17,
          bbox_xyxy: [20, 10, 60, 30],
          confidence: 0.875
        }
      ]
    });

    expect(artifact.labels).toEqual([
      {
        instanceIndex: 17,
        xPercent: 20,
        yPercent: 20,
        confidence: 0.875
      }
    ]);
  });

  it("rejects malformed records instead of guessing instance positions", () => {
    expect(() =>
      parseInstanceArtifact({
        width: 200,
        height: 100,
        instances: [{ instance_index: 1, bbox_xyxy: [1, 2], confidence: 0.5 }]
      })
    ).toThrow("实例记录缺少有效编号或边界框");
  });
});
