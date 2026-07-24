"use client";

import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import {
  ArrowUp,
  Bot,
  ChevronDown,
  MessageSquarePlus,
  SlidersHorizontal,
  Sparkles,
  UserRound
} from "lucide-react";
import { useEffect, useMemo, useRef, useState } from "react";

import { QueryAnswer } from "@/components/agent/query-answer";
import { Button } from "@/components/ui/button";
import { RequestError } from "@/components/ui/request-error";
import { StatusBadge } from "@/components/ui/status-badge";
import { apiRequest } from "@/lib/api/client";
import { queryKeys } from "@/lib/api/query-keys";
import type {
  ChatMessage,
  ConversationDetail,
  ConversationList,
  ConversationMessageRequest,
  HealthData,
  ImageAsset,
  UnifiedQueryResponse
} from "@/lib/api/types";
import { compactId } from "@/lib/format/value";
import type { QueryMode } from "@/lib/store/workspace";

const phases = [
  "正在理解问题",
  "正在查询实验数据",
  "正在检索知识库",
  "正在组织回答",
  "正在校验证据"
];

const advancedModes: Array<{ value: QueryMode; label: string }> = [
  { value: "auto", label: "自动判断（推荐）" },
  { value: "analysis_data", label: "只查实验数据" },
  { value: "material_knowledge", label: "只查知识库" },
  { value: "mixed", label: "数据与知识综合" }
];

export function ConversationPanel({
  jobId,
  image,
  runIds,
  health,
  writeBlocker,
  onLatestAnswer
}: {
  jobId: string;
  image: ImageAsset | null;
  runIds: string[];
  health: HealthData | null;
  writeBlocker: string | null;
  onLatestAnswer: (answer: UnifiedQueryResponse | null) => void;
}) {
  const queryClient = useQueryClient();
  const [activeId, setActiveId] = useState<string | null>(null);
  const [question, setQuestion] = useState("");
  const [mode, setMode] = useState<QueryMode>("auto");
  const [materialName, setMaterialName] = useState("");
  const [materialFormula, setMaterialFormula] = useState("");
  const [materialAliases, setMaterialAliases] = useState("");
  const [phaseIndex, setPhaseIndex] = useState(0);
  const [pendingQuestion, setPendingQuestion] = useState<string | null>(null);
  const scrollAnchor = useRef<HTMLDivElement>(null);

  const conversations = useQuery({
    queryKey: queryKeys.conversations(jobId),
    queryFn: () =>
      apiRequest<ConversationList>(
        `analyses/${encodeURIComponent(jobId)}/conversations`
      ).then((response) => response.data)
  });
  const resolvedActiveId =
    activeId ?? conversations.data?.conversations[0]?.conversation_id ?? null;

  const detail = useQuery({
    queryKey: queryKeys.conversation(jobId, resolvedActiveId || "none"),
    queryFn: () =>
      apiRequest<ConversationDetail>(
        `analyses/${encodeURIComponent(jobId)}/conversations/${encodeURIComponent(resolvedActiveId || "")}`
      ).then((response) => response.data),
    enabled: Boolean(resolvedActiveId),
    staleTime: 10_000
  });

  const createConversation = useMutation({
    mutationFn: () =>
      apiRequest<ConversationDetail>(
        `analyses/${encodeURIComponent(jobId)}/conversations`,
        { method: "POST", body: {} }
      ),
    async onSuccess(response) {
      setActiveId(response.data.conversation_id);
      queryClient.setQueryData(
        queryKeys.conversation(jobId, response.data.conversation_id),
        response.data
      );
      await queryClient.invalidateQueries({ queryKey: queryKeys.conversations(jobId) });
    }
  });

  const sendMessage = useMutation({
    mutationFn: ({
      conversationId,
      body
    }: {
      conversationId: string;
      body: ConversationMessageRequest;
    }) =>
      apiRequest<ConversationDetail>(
        `analyses/${encodeURIComponent(jobId)}/conversations/${encodeURIComponent(conversationId)}/messages`,
        { method: "POST", body }
      ),
    onMutate(variables) {
      setPendingQuestion(variables.body.content);
      setQuestion("");
    },
    async onSuccess(response) {
      await queryClient.cancelQueries({
        queryKey: queryKeys.conversation(jobId, response.data.conversation_id)
      });
      queryClient.setQueryData(
        queryKeys.conversation(jobId, response.data.conversation_id),
        response.data
      );
      setPendingQuestion(null);
      setPhaseIndex(phases.length);
      await queryClient.invalidateQueries({ queryKey: queryKeys.conversations(jobId) });
    },
    onError(_error, variables) {
      setPendingQuestion(null);
      setQuestion((current) => current || variables.body.content);
    }
  });

  useEffect(() => {
    if (!sendMessage.isPending) return;
    const timer = window.setInterval(
      () => setPhaseIndex((value) => Math.min(value + 1, phases.length - 1)),
      850
    );
    return () => window.clearInterval(timer);
  }, [sendMessage.isPending]);

  useEffect(() => {
    if (typeof scrollAnchor.current?.scrollIntoView === "function") {
      scrollAnchor.current.scrollIntoView({ behavior: "smooth", block: "end" });
    }
  }, [detail.data?.messages?.length, sendMessage.isPending]);

  const latestAnswer = useMemo(() => {
    const assistant = [...(detail.data?.messages ?? [])]
      .reverse()
      .find((message) => message.role === "assistant");
    return assistant ? messageAsQueryResponse(assistant) : null;
  }, [detail.data?.messages]);
  useEffect(() => onLatestAnswer(latestAnswer), [latestAnswer, onLatestAnswer]);

  const aliases = materialAliases
    .split(/[,，]/)
    .map((item) => item.trim())
    .filter(Boolean)
    .slice(0, 32);
  const hasManualMaterial = Boolean(
    materialName.trim() || materialFormula.trim() || aliases.length
  );

  async function submit(contentOverride?: string) {
    const content = (contentOverride ?? question).trim();
    if (!content || writeBlocker || sendMessage.isPending) return;
    let conversationId = resolvedActiveId;
    if (!conversationId) {
      const response = await createConversation.mutateAsync();
      conversationId = response.data.conversation_id;
      setActiveId(conversationId);
    }
    setPhaseIndex(0);
    sendMessage.mutate({
      conversationId,
      body: {
        content,
        query_type: mode,
        image_id: image?.image_id ?? null,
        run_ids: runIds.slice(0, 50),
        material_context: hasManualMaterial
          ? {
              name: materialName.trim() || null,
              formula: materialFormula.trim() || null,
              aliases,
              source: "user_confirmation"
            }
          : null
      }
    });
  }

  const messages = detail.data?.messages ?? [];
  const llmHealth = health?.llm_provider;
  const llmUnavailable = Boolean(llmHealth && llmHealth.status !== "healthy");
  const suggestions = runIds.length
    ? [
        "帮我概括当前任务和已有结果",
        "这次运行最值得先检查什么？",
        "当前结果有哪些质量限制？",
        "结合证据，建议我下一步怎么做？"
      ]
    : [
        "帮我看看接下来该做什么",
        "这张图现在可以做哪些分析？",
        "开始分析前需要准备什么？",
        "介绍一下你能调用的工具"
      ];

  return (
    <div className="conversation-shell">
      <aside className="conversation-list" aria-label="对话列表">
        <div className="conversation-list-heading">
          <div>
            <span>CONVERSATIONS</span>
            <strong>实验对话</strong>
          </div>
          <Button
            size="sm"
            tone="secondary"
            onClick={() => createConversation.mutate()}
            disabled={createConversation.isPending}
          >
            <MessageSquarePlus size={14} />新建
          </Button>
        </div>
        {(conversations.data?.conversations ?? []).map((conversation) => (
          <button
            className={
              conversation.conversation_id === resolvedActiveId ? "active" : undefined
            }
            key={conversation.conversation_id}
            onClick={() => setActiveId(conversation.conversation_id)}
          >
            <strong>{conversation.title}</strong>
            <span>{conversation.message_count} 条消息</span>
          </button>
        ))}
        {!conversations.isPending && !conversations.data?.conversations.length ? (
          <p>尚无对话。直接在右侧输入问题即可开始。</p>
        ) : null}
      </aside>

      <section className="conversation-main">
        <div className="conversation-context">
          <span>当前图像：{image?.filename || "未选择"}</span>
          <span>运行作用域：{runIds.length || "无"}</span>
          <span>
            材料：{materialName || image?.material_name || image?.material_formula || "未填写"}
          </span>
          {!llmHealth ? (
            <StatusBadge value="pending" label="正在检查本地 Qwen" />
          ) : llmUnavailable ? (
            <StatusBadge value="degraded" label="本地模型不可用，回答将安全降级" />
          ) : (
            <StatusBadge value="healthy" label="Qwen 已连接" />
          )}
        </div>

        <div className="conversation-messages" aria-live="polite">
          {conversations.isError ? <RequestError error={conversations.error} /> : null}
          {detail.isError ? <RequestError error={detail.error} /> : null}
          {!messages.length && (!resolvedActiveId || !detail.isPending) ? (
            <div className="conversation-welcome">
              <span className="conversation-welcome-icon"><Sparkles size={22} /></span>
              <h2>和 NanoLoop 一起分析这次实验</h2>
              <p>
                像使用 Codex 一样直接描述目标。Qwen 会自然对话，并在涉及实验事实时
                自动调用数据工具或知识库，把结论和证据放在同一条回答里。
              </p>
              <div className="conversation-suggestions">
                {suggestions.map((suggestion) => (
                  <button
                    type="button"
                    key={suggestion}
                    disabled={Boolean(writeBlocker) || sendMessage.isPending}
                    onClick={() => void submit(suggestion)}
                  >
                    <span>{suggestion}</span>
                    <ArrowUp size={14} />
                  </button>
                ))}
              </div>
            </div>
          ) : null}
          {messages.map((message) => (
            <MessageBubble message={message} key={message.message_id} />
          ))}
          {pendingQuestion ? <PendingUserMessage content={pendingQuestion} /> : null}
          {sendMessage.isPending ? (
            <div className="message-row assistant">
              <span className="message-avatar"><Bot size={15} /></span>
              <div className="message-bubble progress">
                <span className="status-spinner" />
                <strong>{phases[phaseIndex]}</strong>
                <small>只展示处理阶段，不展示模型思维过程</small>
              </div>
            </div>
          ) : null}
          <div ref={scrollAnchor} />
        </div>

        <div className="conversation-composer">
          {sendMessage.isError ? <RequestError error={sendMessage.error} /> : null}
          {writeBlocker ? <p className="form-warning">{writeBlocker}</p> : null}
          <details className="conversation-advanced">
            <summary><SlidersHorizontal size={14} />高级选项<ChevronDown size={14} /></summary>
            <div>
              <label>
                <span>查询模式</span>
                <select
                  className="select"
                  value={mode}
                  onChange={(event) => setMode(event.target.value as QueryMode)}
                >
                  {advancedModes.map((item) => (
                    <option key={item.value} value={item.value}>{item.label}</option>
                  ))}
                </select>
              </label>
              <label>
                <span>材料名称</span>
                <input
                  className="input"
                  value={materialName}
                  onChange={(event) => setMaterialName(event.target.value)}
                  placeholder="可选；仅在需要纠正图像元数据时填写"
                />
              </label>
              <label>
                <span>化学式</span>
                <input
                  className="input"
                  value={materialFormula}
                  onChange={(event) => setMaterialFormula(event.target.value)}
                  placeholder="可选；不会根据简称自动猜测"
                />
              </label>
              <label>
                <span>材料别名</span>
                <input
                  className="input"
                  value={materialAliases}
                  onChange={(event) => setMaterialAliases(event.target.value)}
                  placeholder="可选，多个别名用逗号分隔"
                />
              </label>
            </div>
          </details>
          <div className="conversation-input">
            <textarea
              aria-label="发送实验问题"
              value={question}
              maxLength={4000}
              onChange={(event) => setQuestion(event.target.value)}
              onKeyDown={(event) => {
                if (event.key === "Enter" && !event.shiftKey) {
                  event.preventDefault();
                  void submit();
                }
              }}
              placeholder="向 NanoLoop 提问或安排下一步…"
            />
            <button
              type="button"
              aria-label="发送消息"
              onClick={() => void submit()}
              disabled={!question.trim() || Boolean(writeBlocker) || sendMessage.isPending}
            >
              <ArrowUp size={18} />
            </button>
          </div>
          <small>
            Enter 发送 · Shift+Enter 换行 · {image?.filename || "未选图像"} ·{" "}
            {runIds.map((id) => compactId(id)).join("、") || "无运行"}
          </small>
        </div>
      </section>
    </div>
  );
}

function PendingUserMessage({ content }: { content: string }) {
  return (
    <div className="message-row user pending-message" aria-label="正在发送的消息">
      <span className="message-avatar"><UserRound size={15} /></span>
      <article className="message-bubble">
        <div className="message-meta"><strong>你</strong><span>正在发送</span></div>
        <p>{content}</p>
      </article>
    </div>
  );
}

function MessageBubble({ message }: { message: ChatMessage }) {
  const [evidenceOpen, setEvidenceOpen] = useState(false);
  const response =
    message.role === "assistant" ? messageAsQueryResponse(message) : null;
  const evidenceIdPrefix = `message-${safeDomToken(message.message_id)}`;

  function revealEvidence(targetId: string) {
    setEvidenceOpen(true);
    window.requestAnimationFrame(() => {
      const target = document.getElementById(targetId);
      if (typeof target?.scrollIntoView === "function") {
        target.scrollIntoView({
          behavior: "smooth",
          block: "nearest"
        });
      }
    });
  }

  return (
    <div className={`message-row ${message.role}`}>
      <span className="message-avatar">
        {message.role === "assistant" ? <Bot size={15} /> : <UserRound size={15} />}
      </span>
      <article className="message-bubble">
        <div className="message-meta">
          <strong>{message.role === "assistant" ? "NanoLoop" : "你"}</strong>
          {message.role === "assistant" ? (
            <>
              {message.evidence?.llm_model ? (
                <span className="message-model">{message.evidence.llm_model}</span>
              ) : null}
              <StatusBadge value={message.outcome_code || "unknown"} />
              <span>{message.query_type}</span>
              {message.evidence?.fallback_used ? (
                <StatusBadge value="degraded" label="已安全降级" />
              ) : null}
            </>
          ) : null}
        </div>
        <p>
          {response ? (
            <MessageAnswer
              content={message.content}
              evidenceIdPrefix={evidenceIdPrefix}
              response={response}
              revealEvidence={revealEvidence}
            />
          ) : (
            message.content
          )}
        </p>
        {response && message.evidence ? (
          <details
            className="message-evidence"
            open={evidenceOpen}
            onToggle={(event) => setEvidenceOpen(event.currentTarget.open)}
          >
            <summary>
              查看证据与限制（数据 {message.evidence.data_evidence?.length || 0} ·
              引用 {message.evidence.citations?.length || 0}）
            </summary>
            <QueryAnswer
              response={response}
              evidenceOnly
              idPrefix={evidenceIdPrefix}
            />
          </details>
        ) : null}
      </article>
    </div>
  );
}

function MessageAnswer({
  content,
  response,
  evidenceIdPrefix,
  revealEvidence
}: {
  content: string;
  response: UnifiedQueryResponse;
  evidenceIdPrefix: string;
  revealEvidence: (targetId: string) => void;
}) {
  return content.split(/(\[(?:C|D)\d+\])/g).map((part, index) => {
    const evidenceId = /^\[((?:C|D)\d+)\]$/.exec(part)?.[1];
    const targetId = evidenceId
      ? messageEvidenceTarget(response, evidenceIdPrefix, evidenceId)
      : null;
    return targetId ? (
      <button
        aria-label={`展开并定位证据 ${evidenceId}`}
        className="citation-reference message-reference"
        key={`${part}-${index}`}
        onClick={() => revealEvidence(targetId)}
        type="button"
      >
        {part}
      </button>
    ) : (
      part
    );
  });
}

function messageEvidenceTarget(
  response: UnifiedQueryResponse,
  prefix: string,
  evidenceId: string
): string | null {
  if (evidenceId.startsWith("D")) {
    const index = Number(evidenceId.slice(1));
    return index > 0 && index <= (response.data_evidence?.length || 0)
      ? `${prefix}-data-evidence-${index}`
      : null;
  }
  const index = (response.citations ?? []).findIndex(
    (citation) => citation.citation_id === evidenceId
  );
  return index >= 0
    ? `${prefix}-citation-${safeDomToken(evidenceId)}-${index + 1}`
    : null;
}

function safeDomToken(value: string): string {
  return value.replace(/[^A-Za-z0-9_-]/g, "-") || "unknown";
}

function messageAsQueryResponse(message: ChatMessage): UnifiedQueryResponse {
  return {
    query_type: message.query_type,
    answer: message.content,
    data_evidence: message.evidence?.data_evidence ?? [],
    citations: message.evidence?.citations ?? [],
    tool_calls: message.evidence?.tool_calls ?? [],
    material_context: message.material_context ?? null,
    confidence: message.confidence ?? "low",
    limitations: message.evidence?.limitations ?? [],
    needs_clarification: false,
    outcome_code: message.outcome_code ?? "INSUFFICIENT_EVIDENCE"
  };
}
