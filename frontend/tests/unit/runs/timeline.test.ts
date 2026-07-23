import { describe, expect, it } from "vitest";

import { TERMINAL_RUN_STATUSES, timelineFor } from "@/lib/runs/timeline";

describe("run timeline", () => {
  it("marks previous steps complete and the real status active", () => {
    expect(timelineFor("POSTPROCESSING").map((step) => step.state)).toEqual([
      "complete",
      "complete",
      "active",
      "pending",
      "pending",
      "pending"
    ]);
  });

  it.each(["COMPLETED", "COMPLETED_WITH_WARNINGS"])(
    "marks all steps complete for %s",
    (status) => {
      expect(timelineFor(status).every((step) => step.state === "complete")).toBe(true);
    }
  );

  it("has exactly the three backend terminal states", () => {
    expect([...TERMINAL_RUN_STATUSES]).toEqual([
      "COMPLETED",
      "COMPLETED_WITH_WARNINGS",
      "FAILED"
    ]);
  });
});
