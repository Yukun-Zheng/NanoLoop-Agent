import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

import { CommandComposer } from "@/components/agent/command-composer";
import type { UnifiedQueryResponse } from "@/lib/api/types";

function renderComposer(clarification: UnifiedQueryResponse | null) {
  const queryClient = new QueryClient({
    defaultOptions: {
      mutations: { retry: false },
      queries: { retry: false }
    }
  });
  return render(
    <QueryClientProvider client={queryClient}>
      <CommandComposer
        clarification={clarification}
        image={null}
        jobId="job-1"
        onAnswer={vi.fn()}
        runIds={[]}
        writeBlocker={null}
      />
    </QueryClientProvider>
  );
}

function clarificationResponse(
  limitations: string[]
): UnifiedQueryResponse {
  return {
    query_type: "auto",
    answer: "请补充信息后重试。",
    confidence: "low",
    outcome_code: "INSUFFICIENT_EVIDENCE",
    data_evidence: [],
    citations: [],
    limitations,
    needs_clarification: true,
    tool_calls: []
  };
}

describe("CommandComposer clarification UI", () => {
  it("shows material fields only for a material-context limitation", () => {
    renderComposer(
      clarificationResponse(["缺少材料上下文，未执行知识检索"])
    );

    expect(screen.getByLabelText("补充材料上下文")).toBeVisible();
    expect(screen.getByLabelText("补充材料名称")).toBeVisible();
    expect(screen.getByLabelText("补充材料化学式")).toBeVisible();
    expect(screen.getByLabelText("补充材料别名")).toBeVisible();
  });

  it("uses generic guidance for non-material clarification", () => {
    renderComposer(
      clarificationResponse(["跨图像比较缺少统一物理尺度；请补充比例尺"])
    );

    expect(screen.queryByLabelText("补充材料上下文")).not.toBeInTheDocument();
    expect(screen.queryByLabelText("补充材料名称")).not.toBeInTheDocument();
    expect(
      screen.getByText(
        "回答需要进一步澄清。请根据上方回答中的限制，补充运行范围、指标或实验条件后重试。"
      )
    ).toBeVisible();
  });
});
