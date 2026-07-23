import { AlertTriangle, Copy } from "lucide-react";

import { NanoLoopApiError, errorMessage } from "@/lib/api/errors";

import { Button } from "./button";

export function RequestError({ error }: { error: unknown }) {
  const apiError = error instanceof NanoLoopApiError ? error : null;
  const requestId = apiError?.requestId ?? null;
  const title = apiError ? errorTitle(apiError) : "请求没有完成";
  const details = apiError ? Object.entries(apiError.details) : [];
  const validationIssues = apiError ? getValidationIssues(apiError) : [];
  return (
    <div className="request-error" role="alert">
      <AlertTriangle size={18} aria-hidden="true" />
      <div>
        <strong>{title}</strong>
        <p>{errorMessage(error)}</p>
        {apiError?.retryable ? <p>后端标记该错误可重试；请稍后再次提交。</p> : null}
        {validationIssues.length ? (
          <ul>
            {validationIssues.map((issue, index) => (
              <li key={`${issue.location}-${index}`}>
                <strong>{issue.location}</strong>：{issue.message}
              </li>
            ))}
          </ul>
        ) : null}
        {details.length ? (
          <details>
            <summary>查看错误详情</summary>
            <pre>{JSON.stringify(apiError?.details, null, 2)}</pre>
          </details>
        ) : null}
        {requestId ? (
          <Button
            size="sm"
            tone="ghost"
            onClick={() => void navigator.clipboard.writeText(requestId)}
          >
            <Copy size={14} />
            复制 request_id
          </Button>
        ) : null}
      </div>
    </div>
  );
}

function getValidationIssues(
  error: NanoLoopApiError
): Array<{ location: string; message: string }> {
  if (error.status !== 422 || !Array.isArray(error.details.issues)) return [];
  return error.details.issues.flatMap((value) => {
    if (!value || typeof value !== "object") return [];
    const issue = value as Record<string, unknown>;
    const location = Array.isArray(issue.location)
      ? issue.location.filter((part) => typeof part === "string").join(" → ")
      : "";
    if (!location || typeof issue.message !== "string") return [];
    return [{ location, message: issue.message }];
  });
}

function errorTitle(error: NanoLoopApiError): string {
  if (error.status === 401) return "前端服务器认证配置错误";
  if (error.status === 403) return "当前身份权限不足";
  if (error.status === 404) return "资源不存在或当前不可见";
  if (error.status === 409) return "服务器状态已发生变化";
  if (error.status === 413) return "上传内容超过后端限制";
  if (error.status === 422) return "提交内容没有通过校验";
  if (error.status === 429) return "请求过于频繁";
  if (error.status >= 500) return "NanoLoop 服务暂时不可用";
  return "请求没有完成";
}
