import { NanoLoopApiError } from "./errors";

export type ApiEnvelope<T> = {
  request_id: string;
  status: "success" | "accepted" | "error";
  data: T | null;
  error: {
    code: string;
    message: string;
    details: Record<string, unknown>;
    retryable: boolean;
  } | null;
};

export type SuccessfulApiEnvelope<T> = Omit<ApiEnvelope<T>, "data" | "error"> & {
  status: "success" | "accepted";
  data: T;
  error: null;
};

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

export function parseEnvelope<T>(
  value: unknown,
  httpStatus = 200
): SuccessfulApiEnvelope<T> {
  if (!isRecord(value) || typeof value.request_id !== "string") {
    throw new NanoLoopApiError({
      status: httpStatus,
      code: "INVALID_RESPONSE",
      message: "后端返回了无法识别的响应格式"
    });
  }

  const status = value.status;
  if (status !== "success" && status !== "accepted" && status !== "error") {
    throw new NanoLoopApiError({
      status: httpStatus,
      code: "INVALID_RESPONSE",
      message: "后端响应缺少有效状态",
      requestId: value.request_id
    });
  }

  const envelope = value as ApiEnvelope<T>;
  if (status === "error" || httpStatus >= 400 || envelope.error) {
    const payload: Record<string, unknown> = isRecord(envelope.error) ? envelope.error : {};
    throw new NanoLoopApiError({
      status: httpStatus,
      code: typeof payload["code"] === "string" ? payload["code"] : "HTTP_ERROR",
      message:
        typeof payload["message"] === "string"
          ? payload["message"]
          : `请求失败（HTTP ${httpStatus}）`,
      details: isRecord(payload["details"]) ? payload["details"] : {},
      retryable: payload["retryable"] === true,
      requestId: envelope.request_id
    });
  }

  if (envelope.data === null) {
    throw new NanoLoopApiError({
      status: httpStatus,
      code: "EMPTY_RESPONSE",
      message: "后端没有返回数据",
      requestId: envelope.request_id
    });
  }
  return envelope as SuccessfulApiEnvelope<T>;
}
