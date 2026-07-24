import { parseEnvelope, type SuccessfulApiEnvelope } from "./envelope";
import { NanoLoopApiError } from "./errors";

const BFF_ROOT = "/api/nanoloop";

type JsonRequestOptions = Omit<RequestInit, "body"> & {
  body?: unknown;
};

export async function apiRequest<T>(
  path: string,
  options: JsonRequestOptions = {}
): Promise<SuccessfulApiEnvelope<T>> {
  const headers = new Headers(options.headers);
  let body: BodyInit | undefined;
  if (options.body instanceof FormData) {
    body = options.body;
  } else if (options.body !== undefined) {
    headers.set("Content-Type", "application/json");
    body = JSON.stringify(options.body);
  }
  headers.set("Accept", "application/json");

  const response = await fetch(`${BFF_ROOT}/${path.replace(/^\/+/, "")}`, {
    ...options,
    headers,
    body,
    cache: "no-store"
  });

  let payload: unknown;
  try {
    payload = await response.json();
  } catch {
    throw new NanoLoopApiError({
      status: response.status,
      code: "NON_JSON_RESPONSE",
      message: "后端没有返回 JSON 响应",
      requestId: response.headers.get("x-request-id")
    });
  }
  return parseEnvelope<T>(payload, response.status);
}

export function apiUpload<T>(
  path: string,
  body: FormData,
  onProgress: (percent: number) => void
): Promise<SuccessfulApiEnvelope<T>> {
  return new Promise((resolve, reject) => {
    const request = new XMLHttpRequest();
    request.open("POST", `${BFF_ROOT}/${path.replace(/^\/+/, "")}`);
    request.setRequestHeader("Accept", "application/json");
    request.upload.addEventListener("progress", (event) => {
      if (!event.lengthComputable || event.total <= 0) return;
      onProgress(Math.min(99, Math.round((event.loaded / event.total) * 100)));
    });
    request.addEventListener("load", () => {
      let payload: unknown;
      try {
        payload = JSON.parse(request.responseText);
      } catch {
        reject(
          new NanoLoopApiError({
            status: request.status,
            code: "NON_JSON_RESPONSE",
            message: "后端没有返回 JSON 响应",
            requestId: request.getResponseHeader("x-request-id")
          })
        );
        return;
      }
      try {
        const envelope = parseEnvelope<T>(payload, request.status);
        onProgress(100);
        resolve(envelope);
      } catch (error) {
        reject(error);
      }
    });
    request.addEventListener("error", () => {
      reject(
        new NanoLoopApiError({
          status: 0,
          code: "NETWORK_ERROR",
          message: "无法连接 NanoLoop 前端代理",
          retryable: true,
          requestId: request.getResponseHeader("x-request-id")
        })
      );
    });
    request.addEventListener("abort", () => {
      reject(
        new NanoLoopApiError({
          status: 0,
          code: "REQUEST_ABORTED",
          message: "上传已取消",
          retryable: true
        })
      );
    });
    request.send(body);
  });
}

export function toBffArtifactUrl(
  value: string | null | undefined,
  options: { preview?: boolean } = {}
): string | null {
  if (!value) return null;
  let pathname: string;
  try {
    const base =
      typeof window === "undefined" ? "http://nanoloop.invalid" : window.location.origin;
    const parsed = new URL(value, base);
    if (parsed.origin !== base && parsed.origin !== "http://nanoloop.invalid") return null;
    pathname = parsed.pathname;
  } catch {
    return null;
  }
  const prefix = "/api/v1/files/";
  if (!pathname.startsWith(prefix)) return null;
  const token = pathname.slice(prefix.length);
  if (!token || token.includes("/") || token.length > 4096) return null;
  return `${BFF_ROOT}/files/${token}${options.preview ? "?preview=1" : ""}`;
}

export async function fetchArtifact(
  value: string,
  options: { preview?: boolean } = {}
): Promise<Response> {
  const url = toBffArtifactUrl(value, options);
  if (!url) {
    throw new NanoLoopApiError({
      status: 400,
      code: "UNSAFE_ARTIFACT_URL",
      message: "后端返回的制品地址不在允许范围内"
    });
  }
  const response = await fetch(url, { cache: "no-store" });
  if (!response.ok) {
    throw new NanoLoopApiError({
      status: response.status,
      code: "ARTIFACT_DOWNLOAD_FAILED",
      message: `制品下载失败（HTTP ${response.status}）`,
      requestId: response.headers.get("x-request-id")
    });
  }
  return response;
}
