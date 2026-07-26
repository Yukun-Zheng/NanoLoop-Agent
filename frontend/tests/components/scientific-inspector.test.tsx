import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { ScientificInspector } from "@/components/shell/scientific-inspector";
import type { HealthData } from "@/lib/api/types";
import { useWorkspaceStore } from "@/lib/store/workspace";

const health = {
  service: { status: "healthy", detail: null },
  database: { status: "healthy", detail: null },
  model_registry: { status: "healthy", detail: null },
  rag_index: { status: "healthy", detail: null },
  llm_provider: {
    status: "degraded",
    detail: "extractive fallback active; generative provider not configured"
  },
  online_research: {
    status: "degraded",
    detail: "文献检索可用；配置 TAVILY_API_KEY 后可搜索通用网页"
  },
  version: "0.1.0"
} as HealthData;

function renderInspector() {
  const queryClient = new QueryClient({
    defaultOptions: {
      mutations: { retry: false },
      queries: { retry: false }
    }
  });
  render(
    <QueryClientProvider client={queryClient}>
      <ScientificInspector
        collapsible
        jobId="job_1"
        image={null}
        runIds={[]}
        writeBlocker={null}
        health={health}
        model={null}
        run={null}
        answer={null}
        onLatestAnswer={vi.fn()}
        onChildCreated={vi.fn()}
      />
    </QueryClientProvider>
  );
}

beforeEach(() => {
  useWorkspaceStore.setState({
    inspectorTab: "system",
    inspectorCollapsed: false
  });
});

describe("ScientificInspector", () => {
  it("explains technical status in plain language and can collapse", async () => {
    const user = userEvent.setup();
    renderInspector();

    expect(screen.getByText("联网与文献检索")).toBeVisible();
    expect(
      screen.getByText(/学术文献检索可用；配置网页搜索密钥后还能检索通用网页/)
    ).toBeVisible();
    expect(screen.queryByText("Backend 0.1.0")).not.toBeInTheDocument();

    await user.click(screen.getByRole("button", { name: "折叠科研助手" }));
    expect(screen.queryByText("科研助手与实验信息")).not.toBeInTheDocument();
    expect(screen.getByRole("button", { name: "展开科研助手" })).toBeVisible();

    await user.click(screen.getByRole("button", { name: "展开科研助手" }));
    expect(screen.getByText("科研助手与实验信息")).toBeVisible();
  });
});
