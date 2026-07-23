export class NanoLoopApiError extends Error {
  readonly status: number;
  readonly code: string;
  readonly details: Record<string, unknown>;
  readonly retryable: boolean;
  readonly requestId: string | null;

  constructor(input: {
    status: number;
    code: string;
    message: string;
    details?: Record<string, unknown>;
    retryable?: boolean;
    requestId?: string | null;
  }) {
    super(input.message);
    this.name = "NanoLoopApiError";
    this.status = input.status;
    this.code = input.code;
    this.details = input.details ?? {};
    this.retryable = input.retryable ?? false;
    this.requestId = input.requestId ?? null;
  }
}

export function errorMessage(error: unknown): string {
  if (error instanceof NanoLoopApiError) {
    return `${error.message}${error.requestId ? ` · 请求 ${error.requestId}` : ""}`;
  }
  if (error instanceof Error) {
    return error.message;
  }
  return "发生未知错误";
}
