import { describe, expect, it } from "vitest";

import { parseEnvelope } from "@/lib/api/envelope";
import { NanoLoopApiError } from "@/lib/api/errors";

describe("parseEnvelope", () => {
  it("accepts success and preserves request_id", () => {
    const result = parseEnvelope<{ value: number }>({
      request_id: "req-1",
      status: "success",
      data: { value: 7 },
      error: null
    });
    expect(result.request_id).toBe("req-1");
    expect(result.data.value).toBe(7);
  });

  it("accepts an accepted response with data", () => {
    expect(
      parseEnvelope({
        request_id: "req-2",
        status: "accepted",
        data: { run_ids: ["run-1"] },
        error: null
      }).status
    ).toBe("accepted");
  });

  it("raises a structured API error", () => {
    try {
      parseEnvelope(
        {
          request_id: "req-error",
          status: "error",
          data: null,
          error: {
            code: "BOX_REVISION_CONFLICT",
            message: "revision conflict",
            details: { expected: 2 },
            retryable: false
          }
        },
        409
      );
      throw new Error("expected parseEnvelope to fail");
    } catch (error) {
      expect(error).toBeInstanceOf(NanoLoopApiError);
      const apiError = error as NanoLoopApiError;
      expect(apiError.status).toBe(409);
      expect(apiError.code).toBe("BOX_REVISION_CONFLICT");
      expect(apiError.requestId).toBe("req-error");
    }
  });

  it.each([
    {},
    { request_id: "x", status: "mystery", data: {}, error: null },
    { request_id: "x", status: "success", data: null, error: null }
  ])("rejects malformed success shapes", (payload) => {
    expect(() => parseEnvelope(payload)).toThrow(NanoLoopApiError);
  });
});
