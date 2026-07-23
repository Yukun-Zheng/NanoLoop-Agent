import { describe, expect, it } from "vitest";

import { buildQueryScopeKey, selectRunForImage } from "@/lib/runs/selection";

const runs = [
  { run_id: "run-image-a", image_id: "image-a", status: "COMPLETED" },
  { run_id: "run-image-b", image_id: "image-b", status: "COMPLETED" }
];

describe("selectRunForImage", () => {
  it("keeps a selected run only when it belongs to the active image", () => {
    expect(selectRunForImage(runs, "run-image-b", "image-b")?.run_id).toBe(
      "run-image-b"
    );
  });

  it("never falls back to a run from another image", () => {
    expect(selectRunForImage(runs, "run-image-a", "image-b")?.run_id).toBe(
      "run-image-b"
    );
    expect(selectRunForImage([runs[0]!], "run-image-a", "image-b")).toBeNull();
  });

  it("does not select a run when there is no active image", () => {
    expect(selectRunForImage(runs, "run-image-a", null)).toBeNull();
  });

  it("binds query answers to the exact job, image, and ordered run scope", () => {
    expect(buildQueryScopeKey("job-a", "image-a", ["run-a"])).not.toBe(
      buildQueryScopeKey("job-a", "image-b", ["run-a"])
    );
    expect(buildQueryScopeKey("job-a", "image-a", ["run-a"])).not.toBe(
      buildQueryScopeKey("job-a", "image-a", ["run-b"])
    );
  });
});
