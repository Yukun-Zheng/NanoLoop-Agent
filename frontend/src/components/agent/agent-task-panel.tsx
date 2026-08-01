"use client";

import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import {
  Bot,
  Check,
  CircleStop,
  Clock3,
  FileDown,
  Play,
  RotateCcw,
  ShieldCheck,
  Sparkles,
  X
} from "lucide-react";
import { useMemo, useState } from "react";

import { Button } from "@/components/ui/button";
import { RequestError } from "@/components/ui/request-error";
import { StatusBadge } from "@/components/ui/status-badge";
import { apiRequest, toBffArtifactUrl } from "@/lib/api/client";
import { queryKeys } from "@/lib/api/query-keys";
import type {
  AgentTask,
  AgentTaskList,
  CreateAgentTaskRequest,
  HealthData,
  ImageAsset,
  ResolveAgentApprovalRequest,
  SubmitAgentInputRequest
} from "@/lib/api/types";
import { compactId, formatDate } from "@/lib/format/value";

const goalSuggestions = [
  "检查当前任务；若还没有运行，推荐合适模型并在我批准后开始分析。",
  "检查现有运行的质量；如果存在需要复核的结果，提出下一步动作并等待我确认。",
  "比较当前选中的运行，识别质量差异与异常视野，并给出有证据的结论。"
];

const statusLabels: Record<AgentTask["status"], string> = {
  created: "待规划",
  running: "规划执行中",
  waiting_for_approval: "等待批准",
  waiting_for_input: "等待输入",
  waiting_for_external: "等待分析完成",
  completed: "已完成",
  failed: "已停止",
  cancelled: "已取消"
};

type AgentDownload = {
  filename: string;
  href: string;
  sha256?: string;
};

export function AgentTaskPanel({
  jobId,
  image,
  runIds,
  health,
  writeBlocker
}: {
  jobId: string;
  image: ImageAsset | null;
  runIds: string[];
  health: HealthData | null;
  writeBlocker: string | null;
}) {
  const queryClient = useQueryClient();
  const [goal, setGoal] = useState(goalSuggestions[0] ?? "");
  const [activeId, setActiveId] = useState<string | null>(null);
  const [input, setInput] = useState("");

  const tasks = useQuery({
    queryKey: queryKeys.agentTasks(jobId),
    queryFn: () =>
      apiRequest<AgentTaskList>(
        `analyses/${encodeURIComponent(jobId)}/agent-tasks`
      ).then((response) => response.data),
    refetchInterval(query) {
      return (query.state.data?.tasks ?? []).some(
        (task) => task.status === "waiting_for_external"
      )
        ? 2_000
        : false;
    }
  });
  const activeTask = useMemo(
    () =>
      (tasks.data?.tasks ?? []).find((task) => task.task_id === activeId) ??
      tasks.data?.tasks?.[0] ??
      null,
    [activeId, tasks.data]
  );
  const downloads = useMemo(
    () => collectAgentDownloads(activeTask),
    [activeTask]
  );

  function rememberTask(task: AgentTask) {
    setActiveId(task.task_id);
    queryClient.setQueryData<AgentTaskList>(
      queryKeys.agentTasks(jobId),
      (current) => ({
        tasks: [
          task,
          ...(current?.tasks ?? []).filter(
            (candidate) => candidate.task_id !== task.task_id
          )
        ]
      })
    );
  }

  async function refreshWorkspace(task: AgentTask) {
    rememberTask(task);
    await queryClient.invalidateQueries({ queryKey: queryKeys.analysis(jobId) });
  }

  const createTask = useMutation({
    mutationFn: () => {
      const body: CreateAgentTaskRequest = {
        goal: goal.trim(),
        auto_start: true,
        context: {
          selected_image_id: image?.image_id ?? null,
          selected_run_ids: runIds.slice(0, 20)
        }
      };
      return apiRequest<AgentTask>(
        `analyses/${encodeURIComponent(jobId)}/agent-tasks`,
        { method: "POST", body }
      );
    },
    onSuccess: (response) => refreshWorkspace(response.data)
  });

  const runTask = useMutation({
    mutationFn: (taskId: string) =>
      apiRequest<AgentTask>(
        `agent-tasks/${encodeURIComponent(taskId)}/run`,
        { method: "POST", body: { max_steps: 4 } }
      ),
    onSuccess: (response) => refreshWorkspace(response.data)
  });

  const resolveApproval = useMutation({
    mutationFn: ({
      taskId,
      approvalId,
      body
    }: {
      taskId: string;
      approvalId: string;
      body: ResolveAgentApprovalRequest;
    }) =>
      apiRequest<AgentTask>(
        `agent-tasks/${encodeURIComponent(taskId)}/approvals/${encodeURIComponent(approvalId)}`,
        { method: "POST", body }
      ),
    onSuccess: (response) => refreshWorkspace(response.data)
  });

  const submitInput = useMutation({
    mutationFn: ({
      taskId,
      body
    }: {
      taskId: string;
      body: SubmitAgentInputRequest;
    }) =>
      apiRequest<AgentTask>(
        `agent-tasks/${encodeURIComponent(taskId)}/input`,
        { method: "POST", body }
      ),
    async onSuccess(response) {
      setInput("");
      await refreshWorkspace(response.data);
      runTask.mutate(response.data.task_id);
    }
  });

  const cancelTask = useMutation({
    mutationFn: (taskId: string) =>
      apiRequest<AgentTask>(
        `agent-tasks/${encodeURIComponent(taskId)}/cancel`,
        { method: "POST", body: {} }
      ),
    onSuccess: (response) => rememberTask(response.data)
  });

  const agentHealth = health?.agent_runtime;
  const unavailable = agentHealth?.status === "unavailable";
  const pendingApproval = activeTask?.approvals
    ?.slice()
    .reverse()
    .find((approval) => approval.status === "pending");
  const error =
    tasks.error ||
    createTask.error ||
    runTask.error ||
    resolveApproval.error ||
    submitInput.error ||
    cancelTask.error;
  const busy =
    createTask.isPending ||
    runTask.isPending ||
    resolveApproval.isPending ||
    submitInput.isPending ||
    cancelTask.isPending;

  return (
    <div className="agent-task-shell">
      <section className="agent-task-create">
        <div className="agent-task-kicker">
          <span><Sparkles size={13} /> LOCAL SCIENTIFIC AGENT</span>
          <StatusBadge value={agentHealth?.status ?? "unavailable"} />
        </div>
        <h3>把目标交给 Agent</h3>
        <p>
          本地模型负责规划和选择下一动作；数字、图像分析与写操作仍由确定性工具完成。
        </p>
        <textarea
          value={goal}
          onChange={(event) => setGoal(event.target.value)}
          maxLength={4000}
          rows={5}
          aria-label="科研 Agent 目标"
        />
        <div className="agent-goal-suggestions">
          {goalSuggestions.map((suggestion) => (
            <button type="button" key={suggestion} onClick={() => setGoal(suggestion)}>
              {suggestion}
            </button>
          ))}
        </div>
        <Button
          tone="primary"
          onClick={() => createTask.mutate()}
          disabled={busy || unavailable || Boolean(writeBlocker) || goal.trim().length < 3}
        >
          {createTask.isPending ? <Clock3 size={15} /> : <Play size={15} />}
          创建并开始规划
        </Button>
        {unavailable ? (
          <small>{agentHealth?.detail || "本地决策模型当前不可用。"}</small>
        ) : writeBlocker ? (
          <small>{writeBlocker}</small>
        ) : null}
      </section>

      <section className="agent-task-list" aria-label="Agent 任务列表">
        <header>
          <strong>任务</strong>
          <span>{tasks.data?.tasks?.length ?? 0}</span>
        </header>
        {(tasks.data?.tasks ?? []).map((task) => (
          <button
            type="button"
            className={task.task_id === activeTask?.task_id ? "active" : undefined}
            key={task.task_id}
            onClick={() => setActiveId(task.task_id)}
          >
            <span>{statusLabels[task.status]}</span>
            <strong>{task.goal}</strong>
            <small>{compactId(task.task_id, 14)} · {formatDate(task.updated_at)}</small>
          </button>
        ))}
      </section>

      <section className="agent-task-detail">
        {activeTask ? (
          <>
            <header className="agent-task-status">
              <div>
                <span>ACTIVE OBJECTIVE</span>
                <h3>{activeTask.goal}</h3>
              </div>
              <StatusBadge value={activeTask.status} />
            </header>

            <div className="agent-task-meta">
              <span><Bot size={13} />{activeTask.model?.model || "未配置模型"}</span>
              <span>步骤 {activeTask.step_count}/{activeTask.budget.max_steps}</span>
              <span>失败 {activeTask.failure_count}/{activeTask.budget.max_failures}</span>
            </div>

            {(activeTask.plan ?? []).length ? (
              <ol className="agent-plan">
                {(activeTask.plan ?? []).map((step, index) => (
                  <li
                    key={`${index}-${step}`}
                    className={step === activeTask.current_step ? "active" : undefined}
                  >
                    <b>{index + 1}</b>
                    <span>{step}</span>
                  </li>
                ))}
              </ol>
            ) : (
              <p className="agent-empty-plan">模型尚未生成执行计划。</p>
            )}

            {activeTask.status === "waiting_for_external" ? (
              <div className="agent-boundary waiting">
                <span className="status-spinner" />
                <div>
                  <strong>确定性分析正在运行</strong>
                  <p>服务端会在后台自动续查；关闭浏览器也不会中断。</p>
                </div>
              </div>
            ) : null}

            {pendingApproval ? (
              <div className="agent-approval">
                <div>
                  <ShieldCheck size={18} />
                  <div>
                    <strong>需要人工批准：{pendingApproval.tool_name}</strong>
                    <p>{pendingApproval.reason}</p>
                  </div>
                </div>
                <pre>{JSON.stringify(pendingApproval.tool_arguments ?? {}, null, 2)}</pre>
                <div>
                  <Button
                    tone="primary"
                    size="sm"
                    disabled={busy}
                    onClick={() =>
                      resolveApproval.mutate({
                        taskId: activeTask.task_id,
                        approvalId: pendingApproval.approval_id,
                        body: { decision: "approve", comment: "用户在工作台确认" }
                      })
                    }
                  >
                    <Check size={14} />批准
                  </Button>
                  <Button
                    tone="secondary"
                    size="sm"
                    disabled={busy}
                    onClick={() =>
                      resolveApproval.mutate({
                        taskId: activeTask.task_id,
                        approvalId: pendingApproval.approval_id,
                        body: { decision: "reject", comment: "用户在工作台拒绝" }
                      })
                    }
                  >
                    <X size={14} />拒绝
                  </Button>
                </div>
              </div>
            ) : null}

            {activeTask.status === "waiting_for_input" ? (
              <div className="agent-input-request">
                <strong>{activeTask.waiting_question}</strong>
                <textarea
                  value={input}
                  onChange={(event) => setInput(event.target.value)}
                  maxLength={4000}
                  rows={3}
                  placeholder="补充 Agent 继续执行所需的信息"
                />
                <Button
                  size="sm"
                  tone="primary"
                  disabled={busy || !input.trim()}
                  onClick={() =>
                    submitInput.mutate({
                      taskId: activeTask.task_id,
                      body: { content: input.trim() }
                    })
                  }
                >
                  提交并继续
                </Button>
              </div>
            ) : null}

            {activeTask.final_answer ? (
              <div className="agent-final">
                <Check size={17} />
                <div>
                  <strong>任务结果</strong>
                  <p>{activeTask.final_answer}</p>
                  {(activeTask.final_evidence_refs ?? []).length ? (
                    <small>
                      证据：{(activeTask.final_evidence_refs ?? []).join(" · ")}
                    </small>
                  ) : null}
                </div>
              </div>
            ) : null}
            {downloads.length ? (
              <div className="agent-downloads">
                <strong>生成的制品</strong>
                {downloads.map((artifact) => (
                  <a
                    href={artifact.href}
                    key={`${artifact.filename}-${artifact.sha256 ?? artifact.href}`}
                    download
                  >
                    <FileDown size={14} />
                    <span>{artifact.filename}</span>
                  </a>
                ))}
                <small>下载令牌有时效；失效后可让 Agent 重新生成制品。</small>
              </div>
            ) : null}
            {activeTask.error ? (
              <div className="agent-boundary error">
                <CircleStop size={17} />
                <div>
                  <strong>本轮安全停止</strong>
                  <p>{activeTask.error}</p>
                </div>
              </div>
            ) : null}

            <div className="agent-task-actions">
              {["created", "running"].includes(activeTask.status) ? (
                <Button
                  tone="primary"
                  size="sm"
                  disabled={busy}
                  onClick={() => runTask.mutate(activeTask.task_id)}
                >
                  <RotateCcw size={14} />继续执行
                </Button>
              ) : null}
              {!["completed", "failed", "cancelled"].includes(activeTask.status) ? (
                <Button
                  tone="ghost"
                  size="sm"
                  disabled={busy}
                  onClick={() => cancelTask.mutate(activeTask.task_id)}
                >
                  <CircleStop size={14} />取消任务
                </Button>
              ) : null}
            </div>

            <div className="agent-events">
              <header>
                <strong>公开执行轨迹</strong>
                <span>不保存隐藏思维链</span>
              </header>
              <ol>
                {(activeTask.events ?? []).map((event) => (
                  <li key={event.event_id}>
                    <b>{event.sequence}</b>
                    <div>
                      <span>{event.event_type}</span>
                      <p>{event.summary}</p>
                      <small>{formatDate(event.created_at)}</small>
                    </div>
                  </li>
                ))}
              </ol>
            </div>
          </>
        ) : (
          <div className="agent-empty">
            <Bot size={28} />
            <strong>还没有 Agent 任务</strong>
            <p>输入一个可验证的科研目标，模型会从只读检查开始规划。</p>
          </div>
        )}
      </section>

      {error ? <RequestError error={error} /> : null}
    </div>
  );
}

function collectAgentDownloads(task: AgentTask | null): AgentDownload[] {
  const downloads: AgentDownload[] = [];
  for (const observation of task?.latest_observations ?? []) {
    const data = asRecord(observation.data);
    for (const candidate of [data, asRecord(data.docx), asRecord(data.pdf)]) {
      const filename =
        typeof candidate.filename === "string" ? candidate.filename : null;
      const rawUrl =
        typeof candidate.download_url === "string"
          ? candidate.download_url
          : null;
      const href = toBffArtifactUrl(rawUrl);
      if (!filename || !href) continue;
      downloads.push({
        filename,
        href,
        sha256:
          typeof candidate.sha256 === "string" ? candidate.sha256 : undefined
      });
    }
  }
  return Array.from(
    new Map(
      downloads.map((artifact) => [
        artifact.sha256 ?? `${artifact.filename}:${artifact.href}`,
        artifact
      ])
    ).values()
  );
}

function asRecord(value: unknown): Record<string, unknown> {
  return value !== null && typeof value === "object" && !Array.isArray(value)
    ? (value as Record<string, unknown>)
    : {};
}
