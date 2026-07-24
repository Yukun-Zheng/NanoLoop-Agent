import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { ConversationPanel } from "@/components/agent/conversation-panel";
import { apiRequest } from "@/lib/api/client";
import type { ConversationDetail, HealthData } from "@/lib/api/types";

vi.mock("@/lib/api/client", () => ({
  apiRequest: vi.fn(),
  toBffArtifactUrl: vi.fn(() => null)
}));

const mockedApi = vi.mocked(apiRequest);
const now = "2026-07-24T08:00:00Z";
const conversation: ConversationDetail = {
  conversation_id: "conv_1",
  job_id: "job_1",
  title: "任务概括",
  created_by: `prn_${"0".repeat(32)}`,
  created_at: now,
  updated_at: now,
  message_count: 2,
  messages: [
    {
      message_id: "msg_user",
      conversation_id: "conv_1",
      role: "user",
      content: "这张图有多少颗粒",
      query_type: "analysis_data",
      run_ids: [],
      created_at: now
    },
    {
      message_id: "msg_assistant",
      conversation_id: "conv_1",
      role: "assistant",
      content: "当前运行有 42 个颗粒 [D1]。",
      query_type: "analysis_data",
      run_ids: [],
      confidence: "high",
      outcome_code: "OK",
      created_at: now,
      evidence: {
        citations: [],
        data_evidence: [
          {
            tool_name: "get_run_summary",
            validated_arguments: { run_id: "run_1" },
            source_run_ids: ["run_1"],
            aggregates: { particle_count: 42 },
            rows: [],
            units: { particle_count: "count" },
            quality_warnings: []
          }
        ],
        tool_calls: [],
        limitations: ["生成式回答不可用，已使用可信降级结果"],
        llm_provider: "extractive",
        fallback_used: true,
        generation_time_ms: 2,
        prompt_template_id: "nanoloop-scientist-copilot-v1",
        prompt_template_sha256: "a".repeat(64)
      }
    }
  ]
};

const health: HealthData = {
  service: { status: "healthy" },
  database: { status: "healthy" },
  model_registry: { status: "healthy" },
  rag_index: { status: "degraded" },
  llm_provider: { status: "unavailable", detail: "model missing" },
  version: "0.1.0"
};

function renderPanel() {
  const queryClient = new QueryClient({
    defaultOptions: {
      mutations: { retry: false },
      queries: { retry: false }
    }
  });
  return render(
    <QueryClientProvider client={queryClient}>
      <ConversationPanel
        health={health}
        image={null}
        jobId="job_1"
        onLatestAnswer={vi.fn()}
        runIds={[]}
        writeBlocker={null}
      />
    </QueryClientProvider>
  );
}

beforeEach(() => {
  mockedApi.mockReset();
  mockedApi.mockImplementation((async (path, options) => {
    if (path.endsWith("/messages") && options?.method === "POST") {
      const updated: ConversationDetail = {
        ...conversation,
        message_count: 4,
        messages: [
          ...(conversation.messages ?? []),
          {
            ...conversation.messages![0]!,
            message_id: "msg_user_2",
            content: "这个系统能做什么"
          },
          {
            ...conversation.messages![1]!,
            message_id: "msg_assistant_2",
            content: "我可以调用数据工具和知识库。",
            query_type: "general_chat",
            evidence: {
              ...conversation.messages![1]!.evidence!,
              data_evidence: []
            }
          }
        ]
      };
      return {
        request_id: "req_send",
        status: "success",
        data: updated,
        error: null
      };
    }
    if (path.endsWith("/conversations") && options?.method === "POST") {
      return {
        request_id: "req_create",
        status: "success",
        data: { ...conversation, conversation_id: "conv_new", messages: [] },
        error: null
      };
    }
    if (path.endsWith("/conversations")) {
      return {
        request_id: "req_list",
        status: "success",
        data: {
          conversations: [
            {
              conversation_id: conversation.conversation_id,
              job_id: conversation.job_id,
              title: conversation.title,
              created_by: conversation.created_by,
              created_at: conversation.created_at,
              updated_at: conversation.updated_at,
              message_count: conversation.message_count
            }
          ]
        },
        error: null
      };
    }
    return {
      request_id: "req_get",
      status: "success",
      data: conversation,
      error: null
    };
  }) as typeof apiRequest);
});

describe("ConversationPanel", () => {
  it("reloads history and exposes fallback evidence and Ollama status", async () => {
    const user = userEvent.setup();
    renderPanel();

    expect(
      await screen.findByRole("button", { name: "展开并定位证据 D1" })
    ).toBeVisible();
    expect(screen.getByText("已安全降级")).toBeVisible();
    expect(screen.getByText("本地模型不可用，回答将安全降级")).toBeVisible();
    expect(screen.getByText(/查看证据与限制/)).toBeVisible();

    await user.click(screen.getByRole("button", { name: "展开并定位证据 D1" }));
    expect(await screen.findByText("实验数据证据")).toBeVisible();
    expect(screen.getByText(/get_run_summary/)).toBeVisible();
  });

  it("creates a new conversation", async () => {
    const user = userEvent.setup();
    renderPanel();

    await user.click(await screen.findByRole("button", { name: /新建/ }));
    await waitFor(() =>
      expect(mockedApi).toHaveBeenCalledWith(
        "analyses/job_1/conversations",
        expect.objectContaining({ method: "POST" })
      )
    );
  });

  it("offers Codex-style task starters and sends a selected prompt", async () => {
    const user = userEvent.setup();
    mockedApi.mockImplementation((async (path, options) => {
      if (path.endsWith("/messages") && options?.method === "POST") {
        return {
          request_id: "req_send_suggestion",
          status: "success",
          data: {
            ...conversation,
            conversation_id: "conv_new",
            title: "下一步",
            message_count: 2,
            messages: [
              {
                message_id: "msg_suggestion_user",
                conversation_id: "conv_new",
                role: "user",
                content: "帮我看看接下来该做什么",
                query_type: "general_chat",
                run_ids: [],
                created_at: now
              },
              {
                ...conversation.messages![1]!,
                message_id: "msg_suggestion_assistant",
                conversation_id: "conv_new",
                content: "可以先选择模型并创建一次分析运行。",
                query_type: "general_chat"
              }
            ]
          },
          error: null
        };
      }
      if (path.endsWith("/conversations") && options?.method === "POST") {
        return {
          request_id: "req_create_suggestion",
          status: "success",
          data: { ...conversation, conversation_id: "conv_new", messages: [] },
          error: null
        };
      }
      if (path.endsWith("/conversations")) {
        return {
          request_id: "req_empty_list",
          status: "success",
          data: { conversations: [] },
          error: null
        };
      }
      return {
        request_id: "req_empty_detail",
        status: "success",
        data: { ...conversation, conversation_id: "conv_new", messages: [] },
        error: null
      };
    }) as typeof apiRequest);
    renderPanel();

    expect(
      await screen.findByRole("heading", {
        name: "和 NanoLoop 一起分析这次实验"
      })
    ).toBeVisible();
    await user.click(
      screen.getByRole("button", { name: /帮我看看接下来该做什么/ })
    );

    await waitFor(() =>
      expect(mockedApi).toHaveBeenCalledWith(
        "analyses/job_1/conversations/conv_new/messages",
        expect.objectContaining({
          method: "POST",
          body: expect.objectContaining({
            content: "帮我看看接下来该做什么",
            query_type: "auto"
          })
        })
      )
    );
    expect(
      await screen.findByText("可以先选择模型并创建一次分析运行。")
    ).toBeVisible();
  });

  it("sends a general chat message through the conversation endpoint", async () => {
    const user = userEvent.setup();
    renderPanel();

    const input = await screen.findByLabelText("发送实验问题");
    await user.type(input, "这个系统能做什么");
    await user.click(screen.getByRole("button", { name: "发送消息" }));

    await waitFor(() =>
      expect(mockedApi).toHaveBeenCalledWith(
        "analyses/job_1/conversations/conv_1/messages",
        expect.objectContaining({
          method: "POST",
          body: expect.objectContaining({
            content: "这个系统能做什么",
            query_type: "auto"
          })
        })
      )
    );
    expect(await screen.findByText("我可以调用数据工具和知识库。")).toBeVisible();
  });

  it("sends optional material context for a follow-up without requiring a formula", async () => {
    const user = userEvent.setup();
    renderPanel();

    await user.click(await screen.findByText("高级选项"));
    await user.type(screen.getByLabelText("材料名称"), "NdNi");
    await user.type(screen.getByLabelText("材料别名"), "NdNi, 镍钕样品");
    await user.type(screen.getByLabelText("发送实验问题"), "那这个样品呢");
    await user.click(screen.getByRole("button", { name: "发送消息" }));

    await waitFor(() =>
      expect(mockedApi).toHaveBeenCalledWith(
        "analyses/job_1/conversations/conv_1/messages",
        expect.objectContaining({
          body: expect.objectContaining({
            content: "那这个样品呢",
            material_context: {
              aliases: ["NdNi", "镍钕样品"],
              formula: null,
              name: "NdNi",
              source: "user_confirmation"
            }
          })
        })
      )
    );
  });
});
