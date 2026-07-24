import { describe, expect, it, vi } from "vitest";

import { fetchArtifact, toBffArtifactUrl } from "@/lib/api/client";

describe("toBffArtifactUrl", () => {
  it("maps an opaque signed file token to the same-origin BFF", () => {
    expect(toBffArtifactUrl("/api/v1/files/v2.kid.payload.signature")).toBe(
      "/api/nanoloop/files/v2.kid.payload.signature"
    );
  });

  it.each([
    "https://evil.test/api/v1/files/token",
    "/api/v1/health",
    "/api/v1/files/token/extra",
    "/arbitrary/file",
    ""
  ])("rejects unsafe artifact URL %s", (value) => {
    expect(toBffArtifactUrl(value)).toBeNull();
  });

  it("adds only the trusted preview flag to artifact fetches", async () => {
    const fetchMock = vi
      .spyOn(globalThis, "fetch")
      .mockResolvedValueOnce(new Response("image", { status: 200 }));

    await fetchArtifact("/api/v1/files/v2.kid.payload.signature", { preview: true });

    expect(fetchMock).toHaveBeenCalledWith(
      "/api/nanoloop/files/v2.kid.payload.signature?preview=1",
      { cache: "no-store" }
    );
    fetchMock.mockRestore();
  });
});
