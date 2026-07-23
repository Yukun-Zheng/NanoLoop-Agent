import { describe, expect, it } from "vitest";

import { toBffArtifactUrl } from "@/lib/api/client";

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
});
