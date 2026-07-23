import { describe, expect, it } from "vitest";

import {
  canvasTransform,
  displayToOriginal,
  originalToDisplay,
  rectIntersects,
  validateRoiRect
} from "@/lib/roi/geometry";

describe("ROI geometry", () => {
  it("letterboxes a landscape image", () => {
    expect(canvasTransform(800, 600, 1600, 800)).toEqual({
      scale: 0.5,
      offsetX: 0,
      offsetY: 100
    });
  });

  it("uses floor for top-left and ceil for bottom-right", () => {
    const transform = canvasTransform(600, 600, 1024, 1024);
    expect(
      displayToOriginal(
        { x1: 10.2, y1: 20.8, x2: 100.1, y2: 200.2 },
        transform,
        1024,
        1024
      )
    ).toEqual({ x1: 17, y1: 35, x2: 171, y2: 342 });
  });

  it("round trips within one display pixel", () => {
    const transform = canvasTransform(700, 440, 2048, 1536);
    const original = { x1: 123, y1: 210, x2: 999, y2: 1111 };
    const display = originalToDisplay(original, transform);
    const back = displayToOriginal(display, transform, 2048, 1536);
    expect(Math.abs(back.x1 - original.x1)).toBeLessThanOrEqual(1);
    expect(Math.abs(back.y2 - original.y2)).toBeLessThanOrEqual(1);
  });

  it("treats touching half-open rectangles as non-overlapping", () => {
    expect(
      rectIntersects(
        { x1: 0, y1: 0, x2: 32, y2: 32 },
        { x1: 32, y1: 0, x2: 64, y2: 32 }
      )
    ).toBe(false);
  });

  it("enforces 32 px, valid rect and invalid regions", () => {
    const valid = { x1: 0, y1: 0, x2: 100, y2: 100 };
    const invalid = [{ x1: 0, y1: 90, x2: 100, y2: 100 }];
    expect(validateRoiRect({ x1: 2, y1: 2, x2: 34, y2: 34 }, valid, invalid)).toBeNull();
    expect(validateRoiRect({ x1: 2, y1: 2, x2: 33, y2: 34 }, valid, invalid)).toContain(
      "至少"
    );
    expect(validateRoiRect({ x1: 70, y1: 70, x2: 105, y2: 105 }, valid, invalid)).toContain(
      "有效"
    );
    expect(validateRoiRect({ x1: 10, y1: 60, x2: 50, y2: 95 }, valid, invalid)).toContain(
      "无效"
    );
  });
});
