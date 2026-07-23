import { NextResponse } from "next/server";

import { isAllowedProxyRequest, isKnownProxyPath } from "@/lib/api/route-policy";
import { getServerEnv } from "@/lib/env";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

type RouteContext = {
  params: Promise<{ path: string[] }>;
};

const forwardedRequestHeaders = ["accept", "content-type", "range"] as const;
const forwardedResponseHeaders = [
  "cache-control",
  "content-disposition",
  "content-range",
  "content-type",
  "retry-after",
  "x-request-id"
] as const;
const mutationMethods = new Set(["POST", "PUT", "PATCH"]);
const trustedFetchSites = new Set(["same-origin", "same-site", "none"]);

function requestHost(request: Request): string {
  return (request.headers.get("host") || new URL(request.url).host).toLowerCase();
}

function isAllowedFrontendHost(request: Request, allowedOrigins: string[]): boolean {
  const allowedHosts = new Set(
    allowedOrigins.map((origin) => new URL(origin).host.toLowerCase())
  );
  return allowedHosts.has(requestHost(request));
}

function isAllowedMutationSource(
  request: Request,
  allowedOrigins: string[]
): boolean {
  if (!mutationMethods.has(request.method.toUpperCase())) return true;

  const origin = request.headers.get("origin");
  if (origin) {
    try {
      const parsed = new URL(origin);
      if (!["http:", "https:"].includes(parsed.protocol) || parsed.origin !== origin) {
        return false;
      }
      if (
        !allowedOrigins.includes(parsed.origin) ||
        parsed.host.toLowerCase() !== requestHost(request)
      ) {
        return false;
      }
    } catch {
      return false;
    }
  }

  const fetchSite = request.headers.get("sec-fetch-site")?.toLowerCase();
  return !fetchSite || trustedFetchSites.has(fetchSite);
}

function bffError(
  requestId: string,
  status: number,
  code: string,
  message: string,
  retryable = false
) {
  return NextResponse.json(
    {
      request_id: requestId,
      status: "error",
      data: null,
      error: {
        code,
        message,
        details: {},
        retryable
      }
    },
    {
      status,
      headers: {
        "Cache-Control": "private, no-store",
        "X-Content-Type-Options": "nosniff",
        "X-Request-ID": requestId
      }
    }
  );
}

async function proxy(request: Request, context: RouteContext) {
  const incomingRequestId = request.headers.get("x-request-id");
  const requestId =
    incomingRequestId && /^[A-Za-z0-9][A-Za-z0-9._-]{0,99}$/.test(incomingRequestId)
      ? incomingRequestId
      : crypto.randomUUID();
  const env = getServerEnv();
  if (!isAllowedFrontendHost(request, env.NANOLOOP_FRONTEND_ALLOWED_ORIGINS)) {
    return bffError(
      requestId,
      403,
      "UNTRUSTED_FRONTEND_ORIGIN",
      "当前前端 Host/Origin 不在服务端允许列表中"
    );
  }

  const { path: segments } = await context.params;
  let path: string;
  try {
    path = segments.map((segment) => decodeURIComponent(segment)).join("/");
  } catch {
    return bffError(
      requestId,
      404,
      "BFF_ROUTE_NOT_ALLOWED",
      "该前端代理路径包含无效编码"
    );
  }
  if (!isAllowedProxyRequest(path, request.method)) {
    return bffError(
      requestId,
      isKnownProxyPath(path) ? 405 : 404,
      "BFF_ROUTE_NOT_ALLOWED",
      "该前端代理路径或方法不在允许范围内"
    );
  }
  if (!isAllowedMutationSource(request, env.NANOLOOP_FRONTEND_ALLOWED_ORIGINS)) {
    return bffError(
      requestId,
      403,
      "CROSS_SITE_MUTATION_FORBIDDEN",
      "浏览器跨站写请求已被拒绝"
    );
  }

  const base = new URL(env.NANOLOOP_API_INTERNAL_URL);
  const target = new URL(`/api/v1/${path}`, base);
  target.search = new URL(request.url).search;

  const headers = new Headers();
  for (const name of forwardedRequestHeaders) {
    const value = request.headers.get(name);
    if (value) headers.set(name, value);
  }
  headers.set("X-Request-ID", requestId);
  if (env.NANOLOOP_API_KEY) {
    headers.set("X-API-Key", env.NANOLOOP_API_KEY);
  }

  const hasBody = !["GET", "HEAD"].includes(request.method);
  const init: RequestInit & { duplex?: "half" } = {
    method: request.method,
    headers,
    body: hasBody ? request.body : undefined,
    cache: "no-store",
    redirect: "manual",
    signal: request.signal
  };
  if (hasBody) init.duplex = "half";

  let upstream: Response;
  try {
    upstream = await fetch(target, init);
  } catch {
    return bffError(
      requestId,
      503,
      "UPSTREAM_UNAVAILABLE",
      "NanoLoop API 当前不可达",
      true
    );
  }

  if (upstream.status >= 300 && upstream.status < 400) {
    await upstream.body?.cancel();
    return bffError(
      requestId,
      502,
      "UPSTREAM_REDIRECT_BLOCKED",
      "后端返回了不允许的重定向"
    );
  }

  const responseHeaders = new Headers();
  for (const name of forwardedResponseHeaders) {
    const value = upstream.headers.get(name);
    if (value) responseHeaders.set(name, value);
  }
  responseHeaders.set("Cache-Control", "private, no-store");
  responseHeaders.set("X-Request-ID", upstream.headers.get("x-request-id") || requestId);
  responseHeaders.set("X-Content-Type-Options", "nosniff");

  return new Response(upstream.body, {
    status: upstream.status,
    statusText: upstream.statusText,
    headers: responseHeaders
  });
}

export const GET = proxy;
export const POST = proxy;
export const PUT = proxy;
export const PATCH = proxy;
