import { describe, expect, it } from "vitest";

import {
  clearRecentJobs,
  parseJobReference,
  readRecentJobs,
  removeRecentJob,
  rememberJob
} from "@/lib/recent-jobs";

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

    expect(removeRecentJob("job-2", storage).map((item) => item.jobId)).toEqual(["job-1"]);
    expect(clearRecentJobs(storage)).toEqual([]);
    expect(readRecentJobs(storage)).toEqual([]);
  });

  it("accepts pasted task links or complete IDs without asking users to retype them", () => {
    const jobId = "job_96fcd32e0bcf4ebeb9864242dd0f6a73";
    expect(parseJobReference(jobId)).toBe(jobId);
    expect(parseJobReference(`http://127.0.0.1:3000/workspace/${jobId}?run=run-1`)).toBe(
      jobId
    );
    expect(parseJobReference(`/workspace/${jobId}`)).toBe(jobId);
    expect(parseJobReference("96fcd32e0bcf")).toBeNull();
  });
});
