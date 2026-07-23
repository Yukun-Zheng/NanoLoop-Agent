import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import { RequestError } from "@/components/ui/request-error";
import { NanoLoopApiError } from "@/lib/api/errors";

describe("RequestError", () => {
  it("maps backend validation issues to readable field locations", () => {
    render(
      <RequestError
        error={
          new NanoLoopApiError({
            status: 422,
            code: "VALIDATION_ERROR",
            message: "请求参数校验失败",
            requestId: "req-validation",
            details: {
              issues: [
                {
                  location: ["body", "images", "0", "sample_id"],
                  message: "Field required",
                  type: "missing"
                }
              ]
            }
          })
        }
      />
    );

    expect(screen.getByText("提交内容没有通过校验")).toBeVisible();
    expect(screen.getByText("body → images → 0 → sample_id")).toBeVisible();
    expect(screen.getAllByText(/Field required/)).not.toHaveLength(0);
    expect(screen.getByRole("button", { name: "复制 request_id" })).toBeVisible();
  });

  it("explains that browser users must not enter a shared key after a 401", () => {
    render(
      <RequestError
        error={
          new NanoLoopApiError({
            status: 401,
            code: "AUTHENTICATION_REQUIRED",
            message: "missing credential"
          })
        }
      />
    );
    expect(screen.getByText("前端服务器认证配置错误")).toBeVisible();
  });
});
