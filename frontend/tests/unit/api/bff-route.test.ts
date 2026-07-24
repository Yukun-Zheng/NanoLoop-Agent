import { afterEach, describe, expect, it, vi } from "vitest";

import { GET, POST } from "@/app/api/nanoloop/[...path]/route";

afterEach(() => {
  vi.unstubAllGlobals();
  delete process.env.NANOLOOP_API_KEY;
  delete process.env.NANOLOOP_API_INTERNAL_URL;
  delete process.env.NANOLOOP_FRONTEND_ALLOWED_ORIGINS;
});

describe("NanoLoop BFF route", () => {
  it("injects only the server API key and strips browser credentials", async () => {
    process.env.NANOLOOP_API_INTERNAL_URL = "http://backend:8000";
    process.env.NANOLOOP_API_KEY = "server-secret";
    const upstream = vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
      void input;
      void init;
      return new Response(
        JSON.stringify({
          request_id: "req-upstream",
          status: "success",
          data: { ok: true },
          error: null
        }),
        {
          headers: {
            "content-type": "application/json",
            "x-request-id": "req-upstream"
          }
        }
      );
    });
    vi.stubGlobal("fetch", upstream);

    const response = await GET(
      new Request("http://localhost:3000/api/nanoloop/health", {
        headers: {
          authorization: "Bearer browser-secret",
          cookie: "session=browser-secret",
          "x-api-key": "browser-secret",
          "x-request-id": "valid-request-id"
        }
      }),
      { params: Promise.resolve({ path: ["health"] }) }
    );

    expect(response.status).toBe(200);
    expect(response.headers.get("cache-control")).toBe("private, no-store");
    expect(response.headers.get("content-length")).toBeNull();
    expect(upstream).toHaveBeenCalledOnce();
    const [target, init] = upstream.mock.calls[0]!;
    expect(String(target)).toBe("http://backend:8000/api/v1/health");
    const headers = new Headers(init?.headers);
    expect(headers.get("x-api-key")).toBe("server-secret");
    expect(headers.get("x-request-id")).toBe("valid-request-id");
    expect(headers.has("authorization")).toBe(false);
    expect(headers.has("cookie")).toBe(false);
  });

  it("rejects unknown and malformed routes before contacting the backend", async () => {
    const upstream = vi.fn();
    vi.stubGlobal("fetch", upstream);

    const unknown = await GET(
      new Request("http://localhost:3000/api/nanoloop/openapi.json"),
      { params: Promise.resolve({ path: ["openapi.json"] }) }
    );
    const malformed = await GET(
      new Request("http://localhost:3000/api/nanoloop/%"),
      { params: Promise.resolve({ path: ["%"] }) }
    );

    expect(unknown.status).toBe(404);
    expect(malformed.status).toBe(404);
    expect(upstream).not.toHaveBeenCalled();
  });

  it("rejects cross-site writes before injecting the server credential", async () => {
    process.env.NANOLOOP_API_INTERNAL_URL = "http://backend:8000";
    process.env.NANOLOOP_API_KEY = "server-secret";
    const upstream = vi.fn();
    vi.stubGlobal("fetch", upstream);

    const response = await POST(
      new Request("http://localhost:3000/api/nanoloop/analyses", {
        method: "POST",
        headers: {
          origin: "https://attacker.example",
          "sec-fetch-site": "cross-site",
          "content-type": "multipart/form-data; boundary=test",
          "x-request-id": "csrf-attempt"
        },
        body: "--test--\r\n"
      }),
      { params: Promise.resolve({ path: ["analyses"] }) }
    );

    expect(response.status).toBe(403);
    expect(response.headers.get("x-request-id")).toBe("csrf-attempt");
    await expect(response.json()).resolves.toMatchObject({
      request_id: "csrf-attempt",
      status: "error",
      error: { code: "CROSS_SITE_MUTATION_FORBIDDEN", retryable: false }
    });
    expect(upstream).not.toHaveBeenCalled();
  });

  it("allows same-origin writes and keeps browser credentials stripped", async () => {
    process.env.NANOLOOP_API_INTERNAL_URL = "http://backend:8000";
    process.env.NANOLOOP_API_KEY = "server-secret";
    const upstream = vi.fn(
      async (input: RequestInfo | URL, init?: RequestInit) => {
        void input;
        void init;
        return (
        new Response(
        JSON.stringify({
          request_id: "req-upstream",
          status: "accepted",
          data: { job_id: "job-1" },
          error: null
        }),
          { status: 202, headers: { "content-type": "application/json" } }
        )
        );
      }
    );
    vi.stubGlobal("fetch", upstream);

    const response = await POST(
      new Request("http://localhost:3000/api/nanoloop/analyses", {
        method: "POST",
        headers: {
          origin: "http://localhost:3000",
          "sec-fetch-site": "same-origin",
          authorization: "Bearer browser-secret",
          cookie: "session=browser-secret",
          "x-api-key": "browser-secret",
          "content-type": "application/json"
        },
        body: "{}"
      }),
      { params: Promise.resolve({ path: ["analyses"] }) }
    );

    expect(response.status).toBe(202);
    expect(upstream).toHaveBeenCalledOnce();
    const [, init] = upstream.mock.calls[0]!;
    const headers = new Headers(init?.headers);
    expect(headers.get("x-api-key")).toBe("server-secret");
    expect(headers.has("authorization")).toBe(false);
    expect(headers.has("cookie")).toBe(false);
  });

  it("rejects requests for an untrusted Host before contacting the backend", async () => {
    const upstream = vi.fn();
    vi.stubGlobal("fetch", upstream);

    const response = await GET(
      new Request("http://localhost:3000/api/nanoloop/health", {
        headers: { host: "rebound.attacker" }
      }),
      { params: Promise.resolve({ path: ["health"] }) }
    );

    expect(response.status).toBe(403);
    expect(response.headers.get("cache-control")).toBe("private, no-store");
    await expect(response.json()).resolves.toMatchObject({
      status: "error",
      error: { code: "UNTRUSTED_FRONTEND_ORIGIN" }
    });
    expect(upstream).not.toHaveBeenCalled();
  });

  it("forwards artifact preview requests and preserves the backend PNG", async () => {
    process.env.NANOLOOP_API_INTERNAL_URL = "http://backend:8000";
    const png = new Uint8Array([137, 80, 78, 71]);
    const upstream = vi.fn(async (input: RequestInfo | URL) => {
      void input;
      return new Response(png, {
        headers: {
          "content-disposition": 'inline; filename="preview.png"',
          "content-type": "image/png",
          "x-request-id": "req-tiff-preview"
        }
      });
    });
    vi.stubGlobal("fetch", upstream);

    const response = await GET(
      new Request("http://localhost:3000/api/nanoloop/files/signed-token?preview=1"),
      { params: Promise.resolve({ path: ["files", "signed-token"] }) }
    );

    expect(response.status).toBe(200);
    expect(response.headers.get("content-type")).toBe("image/png");
    expect(response.headers.get("content-disposition")).toBe(
      'inline; filename="preview.png"'
    );
    expect(new Uint8Array(await response.arrayBuffer())).toEqual(png);
    expect(upstream).toHaveBeenCalledOnce();
    expect(String(upstream.mock.calls[0]![0])).toBe(
      "http://backend:8000/api/v1/files/signed-token?preview=1"
    );
  });

  it("leaves non-preview TIFF downloads byte-for-byte unchanged", async () => {
    process.env.NANOLOOP_API_INTERNAL_URL = "http://backend:8000";
    const bytes = new Uint8Array([73, 73, 42, 0]);
    vi.stubGlobal(
      "fetch",
      vi.fn(async () => {
        return new Response(bytes, {
          headers: { "content-type": "image/tiff" }
        });
      })
    );

    const response = await GET(
      new Request("http://localhost:3000/api/nanoloop/files/signed-token"),
      { params: Promise.resolve({ path: ["files", "signed-token"] }) }
    );

    expect(response.headers.get("content-type")).toBe("image/tiff");
    expect(new Uint8Array(await response.arrayBuffer())).toEqual(bytes);
  });
});
