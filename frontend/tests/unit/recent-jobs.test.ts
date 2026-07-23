import { describe, expect, it } from "vitest";

import { readRecentJobs, rememberJob } from "@/lib/recent-jobs";

describe("recent jobs", () => {
  it("keeps the most recent local job without storing backend payloads", () => {
    const values = new Map<string, string>();
    const storage = {
      getItem(key: string) {
        return values.get(key) ?? null;
      },
      setItem(key: string, value: string) {
        values.set(key, value);
      }
    };
    rememberJob(
      { jobId: "job-1", name: "First", openedAt: "2026-07-23T00:00:00Z" },
      storage
    );
    rememberJob(
      { jobId: "job-2", name: "Second", openedAt: "2026-07-23T01:00:00Z" },
      storage
    );
    expect(readRecentJobs(storage).map((item) => item.jobId)).toEqual(["job-2", "job-1"]);
    expect(storage.getItem("nanoloop:recent-jobs:v1")).not.toContain("runs");
  });
});
