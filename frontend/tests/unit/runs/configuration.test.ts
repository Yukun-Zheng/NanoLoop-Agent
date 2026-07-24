import { describe, expect, it } from "vitest";

import {
  defaultAnalysisName,
  runParameterError
} from "@/lib/runs/configuration";

describe("defaultAnalysisName", () => {
  it("derives a useful single-image task name without user input", () => {
    expect(defaultAnalysisName(["sample-01.tif"])).toBe("图像分割 · sample-01");
  });

  it("uses a bounded batch label for multiple images", () => {
    expect(defaultAnalysisName(["a.png", "b.jpg", "c.tiff"])).toBe(
      "批量图像分割 · 3 张图像"
    );
  });
});

describe("runParameterError", () => {
  it("accepts empty fields so model defaults can be used", () => {
    expect(runParameterError("", "")).toBeNull();
  });

  it("rejects out-of-range thresholds and fractional areas before submission", () => {
    expect(runParameterError("1.2", "")).toContain("0 到 1");
    expect(runParameterError("", "2.5")).toContain("整数");
    expect(runParameterError("0.65", "16")).toBeNull();
  });
});
