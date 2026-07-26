import { describe, expect, it, vi } from "vitest";

import {
  artifactPreviewIdentity,
  fetchArtifact,
  toBffArtifactUrl
} from "@/lib/api/client";

function v2Token(
  claims: Record<string, unknown>,
  signature = "signature"
): string {
  const payload = Buffer.from(JSON.stringify(claims)).toString("base64url");
  return `v2.initial.${payload}.${signature}`;
}

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

  it("adds the trusted inline flag for browser PDF previews", () => {
    expect(
      toBffArtifactUrl("/api/v1/files/v2.kid.payload.signature", { inline: true })
    ).toBe("/api/nanoloop/files/v2.kid.payload.signature?inline=1");
  });

  it("keeps one preview identity when a v2 capability token is reissued", () => {
    const immutableClaims = {
      v: 2,
      tid: "tnt_00000000000000000000000000000000",
      sub: "prn_00000000000000000000000000000000",
      jid: "job_123",
      aid: "art_456",
      pur: "download.run_artifact",
      sha256: "a".repeat(64)
    };
    const first = `/api/v1/files/${v2Token({
      ...immutableClaims,
      iat: 100,
      exp: 1000,
      jti: "first"
    })}`;
    const reissued = `/api/v1/files/${v2Token(
      {
        ...immutableClaims,
        iat: 200,
        exp: 1100,
        jti: "second"
      },
      "another-signature"
    )}`;

    expect(artifactPreviewIdentity(first)).toBe(artifactPreviewIdentity(reissued));
  });

  it("does not merge two different immutable artifacts", () => {
    const baseClaims = {
      v: 2,
      tid: "tnt_00000000000000000000000000000000",
      sub: "prn_00000000000000000000000000000000",
      jid: "job_123",
      pur: "download.run_artifact",
      sha256: "a".repeat(64)
    };
    const first = `/api/v1/files/${v2Token({ ...baseClaims, aid: "art_first" })}`;
    const second = `/api/v1/files/${v2Token({ ...baseClaims, aid: "art_second" })}`;

    expect(artifactPreviewIdentity(first)).not.toBe(artifactPreviewIdentity(second));
  });
});
