"use client";

import * as Dialog from "@radix-ui/react-dialog";
import { useQuery } from "@tanstack/react-query";
import {
  Activity,
  ArrowLeft,
  Bot,
  BoxSelect,
  CheckCircle2,
  CircleDot,
  FileImage,
  FolderKanban,
  Layers3,
  Library,
  Microscope,
  PanelLeftClose,
  PanelLeftOpen,
  PanelRight,
  X,
} from "lucide-react";
import dynamic from "next/dynamic";
import Link from "next/link";
import { useEffect, useMemo, useRef, useState } from "react";

import { CommandComposer } from "@/components/agent/command-composer";
import { AgentWelcome } from "@/components/agent/agent-welcome";
import { QueryHistory } from "@/components/agent/query-history";
import { QueryAnswer } from "@/components/agent/query-answer";
import { ModelSelector } from "@/components/models/model-selector";
import { ProjectOverview } from "@/components/project/project-overview";
import { ResultView } from "@/components/results/result-view";
import { RunTimeline } from "@/components/runs/run-timeline";
import { Brand } from "@/components/shell/brand";
import { HealthIndicator } from "@/components/shell/health-indicator";
import { ScientificInspector } from "@/components/shell/scientific-inspector";
import { Button } from "@/components/ui/button";
import { EmptyState } from "@/components/ui/empty-state";
import { RequestError } from "@/components/ui/request-error";
import { StatusBadge } from "@/components/ui/status-badge";
import { apiRequest } from "@/lib/api/client";
import { NanoLoopApiError } from "@/lib/api/errors";
import { getHealth } from "@/lib/api/openapi-client";
import { queryKeys } from "@/lib/api/query-keys";
import type {
  BoxSet,
  JobDetail,
  ModelList,
  QueryHistoryData,
  Run,
  UnifiedQueryResponse
} from "@/lib/api/types";
import { compactId } from "@/lib/format/value";
import { coreMutationBlocker } from "@/lib/health";
import { rememberJob } from "@/lib/recent-jobs";
import { buildQueryScopeKey, selectRunForImage } from "@/lib/runs/selection";
import { TERMINAL_RUN_STATUSES } from "@/lib/runs/timeline";
import {
  type WorkspaceStage,
  useWorkspaceStore
} from "@/lib/store/workspace";

const RoiEditor = dynamic(
  () => import("@/components/roi/roi-editor").then((module) => module.RoiEditor),
  {
    ssr: false,
    loading: () => (
      <div className="centered-stage">
        <span className="status-spinner" />
        <p>正在准备 ROI 画布…</p>
      </div>
    )
  }
);

const stages: Array<{ value: WorkspaceStage; label: string; icon: typeof FolderKanban }> = [
  { value: "project", label: "任务概览", icon: FolderKanban },
  { value: "roi", label: "局部区域（可跳过）", icon: BoxSelect },
  { value: "models", label: "开始分析", icon: Microscope },
  { value: "runs", label: "运行进度", icon: Activity },
  { value: "results", label: "查看结果", icon: Layers3 },
  { value: "agent", label: "证据问答", icon: Bot }
];

const COMPARABLE_RUN_STATUSES = new Set(["COMPLETED", "COMPLETED_WITH_WARNINGS"]);

export function WorkspaceCommandCenter({
  jobId,
  initialRunId = null,
  autoRun = false,
  launchWarning = null
}: {
  jobId: string;
  initialRunId?: string | null;
  autoRun?: boolean;
  launchWarning?: string | null;
}) {
  const stage = useWorkspaceStore((state) => state.stage);
  const setStage = useWorkspaceStore((state) => state.setStage);
  const activeImageId = useWorkspaceStore((state) => state.activeImageId);
  const setActiveImage = useWorkspaceStore((state) => state.setActiveImage);
  const activeRunId = useWorkspaceStore((state) => state.activeRunId);
  const setActiveRun = useWorkspaceStore((state) => state.setActiveRun);
  const selectedRunIds = useWorkspaceStore((state) => state.selectedRunIds);
  const setSelectedRuns = useWorkspaceStore((state) => state.setSelectedRuns);
  const setInspectorTab = useWorkspaceStore((state) => state.setInspectorTab);
  const railCollapsed = useWorkspaceStore((state) => state.railCollapsed);
  const toggleRail = useWorkspaceStore((state) => state.toggleRail);
  const [answerState, setAnswerState] = useState<{
    scope: string;
    value: UnifiedQueryResponse;
  } | null>(null);
  const [inspectorOpen, setInspectorOpen] = useState(false);
  const [autoRevealRunIds, setAutoRevealRunIds] = useState<string[]>(
    autoRun && initialRunId ? [initialRunId] : []
  );
  const activeAnswerScope = useRef<string | null>(null);
  const initializedAutoRevealKey = useRef("");

  const analysis = useQuery({
    queryKey: queryKeys.analysis(jobId),
    queryFn: () =>
      apiRequest<JobDetail>(`analyses/${encodeURIComponent(jobId)}`).then(
        (response) => response.data
      ),
    refetchInterval(query) {
      if (query.state.error) return false;
      const data = query.state.data;
      const hasActive = (data?.runs ?? []).some(
        (run) => !TERMINAL_RUN_STATUSES.has(run.status)
      );
      if (!hasActive) return false;
      return typeof document !== "undefined" && document.hidden ? 5_000 : 1_500;
    }
  });

  const health = useQuery({
    queryKey: queryKeys.health,
    queryFn: () => getHealth().then((response) => response.data),
    refetchInterval: 15_000
  });

  const models = useQuery({
    queryKey: queryKeys.models,
    queryFn: () => apiRequest<ModelList>("models").then((response) => response.data)
  });

  const activeImage = useMemo(
    () =>
      (analysis.data?.images ?? []).find((image) => image.image_id === activeImageId) ||
      (analysis.data?.images ?? [])[0] ||
      null,
    [activeImageId, analysis.data]
  );
  const activeRun = useMemo(
    () =>
      selectRunForImage(
        analysis.data?.runs ?? [],
        activeRunId,
        activeImage?.image_id ?? null
      ),
    [activeImage, activeRunId, analysis.data]
  );
  const composerRunIds = useMemo(() => {
    const runs = analysis.data?.runs ?? [];
    const selectedForImage = selectedRunIds.filter((runId) =>
      runs.some(
        (run) =>
          run.run_id === runId &&
          (!activeImage || run.image_id === activeImage.image_id)
      )
    );
    if (selectedForImage.length) return selectedForImage;
    return activeRun && (!activeImage || activeRun.image_id === activeImage.image_id)
      ? [activeRun.run_id]
      : [];
  }, [activeImage, activeRun, analysis.data, selectedRunIds]);
  const answerScope = buildQueryScopeKey(
    jobId,
    activeImage?.image_id ?? null,
    composerRunIds
  );
  useEffect(() => {
    activeAnswerScope.current = answerScope;
  }, [answerScope]);
  const answer = answerState?.scope === answerScope ? answerState.value : null;
  const queryHistory = useQuery({
    queryKey: queryKeys.queryHistory(jobId),
    queryFn: () =>
      apiRequest<QueryHistoryData>(
        `analyses/${encodeURIComponent(jobId)}/queries?limit=50`
      ).then((response) => response.data),
    enabled: stage === "agent"
  });
  const scopedQueryHistory = useMemo(
    () =>
      (queryHistory.data?.items ?? []).filter(
        (item) =>
          buildQueryScopeKey(
            item.job_id,
            item.request.image_id ?? null,
            item.request.run_ids ?? []
          ) === answerScope
      ),
    [answerScope, queryHistory.data]
  );

  const boxes = useQuery({
    queryKey: queryKeys.boxes(jobId, activeImage?.image_id || "none"),
    queryFn: () =>
      apiRequest<BoxSet>(
        `analyses/${encodeURIComponent(jobId)}/images/${encodeURIComponent(activeImage?.image_id || "")}/boxes`
      ).then((response) => response.data),
    enabled: Boolean(activeImage)
  });

  useEffect(() => {
    if (!analysis.data) return;
    const nextImages = analysis.data.images ?? [];
    const nextRuns = analysis.data.runs ?? [];
    const normalizedImageId =
      activeImageId && nextImages.some((item) => item.image_id === activeImageId)
        ? activeImageId
        : nextImages[0]?.image_id ?? null;
    if (normalizedImageId !== activeImageId) {
      setActiveImage(normalizedImageId);
    }
    const activeRunCandidate = nextRuns.find((item) => item.run_id === activeRunId);
    if (
      !activeRunCandidate ||
      (normalizedImageId && activeRunCandidate.image_id !== normalizedImageId)
    ) {
      setActiveRun(
        nextRuns.find((item) => item.image_id === normalizedImageId)?.run_id ?? null
      );
    }
    const validSelectedRunIds = selectedRunIds.filter((runId) =>
      nextRuns.some((item) => item.run_id === runId)
    );
    if (
      validSelectedRunIds.length !== selectedRunIds.length ||
      validSelectedRunIds.some((runId, index) => runId !== selectedRunIds[index])
    ) {
      setSelectedRuns(validSelectedRunIds);
    }
    rememberJob({ jobId, name: analysis.data.job.name });
  }, [
    activeImageId,
    activeRunId,
    analysis.data,
    jobId,
    selectedRunIds,
    setActiveImage,
    setActiveRun,
    setSelectedRuns
  ]);

  useEffect(() => {
    if (!analysis.data || !autoRevealRunIds.length) return;
    const trackedRuns = autoRevealRunIds
      .map((runId) => (analysis.data?.runs ?? []).find((run) => run.run_id === runId))
      .filter((run): run is Run => Boolean(run));
    if (trackedRuns.length !== autoRevealRunIds.length) return;

    const first = trackedRuns[0];
    const autoRevealKey = autoRevealRunIds.join("|");
    if (first && initializedAutoRevealKey.current !== autoRevealKey) {
      initializedAutoRevealKey.current = autoRevealKey;
      setActiveImage(first.image_id);
      setActiveRun(first.run_id);
      setStage(
        COMPARABLE_RUN_STATUSES.has(first.status) ? "results" : "runs"
      );
    }
    if (!trackedRuns.every((run) => TERMINAL_RUN_STATUSES.has(run.status))) return;

    const completed =
      trackedRuns.find((run) => COMPARABLE_RUN_STATUSES.has(run.status)) || first;
    if (completed) {
      setActiveImage(completed.image_id);
      setActiveRun(completed.run_id);
      setSelectedRuns(
        trackedRuns
          .filter(
            (run) =>
              run.image_id === completed.image_id &&
              COMPARABLE_RUN_STATUSES.has(run.status)
          )
          .map((run) => run.run_id)
          .slice(0, 3)
      );
      setStage(COMPARABLE_RUN_STATUSES.has(completed.status) ? "results" : "runs");
    }
    queueMicrotask(() => setAutoRevealRunIds([]));
  }, [
    analysis.data,
    autoRevealRunIds,
    setActiveImage,
    setActiveRun,
    setSelectedRuns,
    setStage
  ]);

  if (analysis.isPending) {
    return (
      <main className="centered-state">
        <span className="status-spinner" />
        <p>正在载入科研任务…</p>
      </main>
    );
  }

  if (!analysis.data) {
    return (
      <main className="centered-state">
        <RequestError error={analysis.error} />
        <Button asChild tone="primary">
          <Link href="/"><ArrowLeft size={16} />返回任务首页</Link>
        </Button>
      </main>
    );
  }

  const detail = analysis.data;
  const images = detail.images ?? [];
  const runs = detail.runs ?? [];
  const activeModel =
    (models.data?.models ?? []).find((model) => model.model_id === activeRun?.model_id) ||
    null;
  const availableRuns = runs.filter((run) => run.image_id === activeImage?.image_id);
  const comparisonRuns = runs.filter(
    (run) =>
      selectedRunIds.includes(run.run_id) &&
      run.image_id === activeImage?.image_id &&
      COMPARABLE_RUN_STATUSES.has(run.status)
  );
  const writeBlocker = coreMutationBlocker(health.data, {
    failed: health.isError,
    pending: health.isPending
  });

  function handleRunsCreated(runIds: string[]) {
    const first = runIds[0];
    if (first) setActiveRun(first);
    setSelectedRuns(runIds.slice(0, 3));
    setAutoRevealRunIds(runIds);
    setStage("runs");
  }

  function toggleComparison(run: Run) {
    if (!COMPARABLE_RUN_STATUSES.has(run.status)) return;
    if (selectedRunIds.includes(run.run_id)) {
      setSelectedRuns(selectedRunIds.filter((id) => id !== run.run_id));
      return;
    }
    const sameImage = runs.filter(
      (candidate) =>
        selectedRunIds.includes(candidate.run_id) && candidate.image_id === run.image_id
    );
    setSelectedRuns([...sameImage.map((candidate) => candidate.run_id), run.run_id].slice(-3));
  }

  return (
    <main className="workspace-page">
      <header className="workspace-topbar">
        <div className="workspace-brand">
          <Button
            tone="ghost"
            size="sm"
            onClick={toggleRail}
            aria-label={railCollapsed ? "展开项目栏" : "折叠项目栏"}
          >
            {railCollapsed ? <PanelLeftOpen size={17} /> : <PanelLeftClose size={17} />}
          </Button>
          <Brand />
          <span className="topbar-divider" />
          <div className="workspace-title">
            <strong>{detail.job.name}</strong>
            <code>{compactId(detail.job.job_id, 16)}</code>
          </div>
        </div>
        <div className="topbar-actions">
          <StatusBadge value={detail.job.status} />
          <HealthIndicator />
          <Button asChild tone="ghost" size="sm">
            <Link className="workspace-knowledge-link" href="/knowledge">
              <Library size={15} />
              <span className="workspace-knowledge-label">知识库</span>
            </Link>
          </Button>
        </div>
      </header>

      <div className={`workspace-grid${railCollapsed ? " rail-collapsed" : ""}`}>
        <aside className="project-rail" aria-label="项目与任务">
          <div className="rail-heading">
            <span>PROJECT</span>
            <strong>{detail.job.name}</strong>
          </div>
          <nav className="stage-navigation" aria-label="工作区阶段">
            {stages.map((item) => {
              const Icon = item.icon;
              return (
                <button
                  className={stage === item.value ? "active" : undefined}
                  key={item.value}
                  onClick={() => setStage(item.value)}
                  title={railCollapsed ? item.label : undefined}
                  aria-label={item.label}
                  aria-current={stage === item.value ? "page" : undefined}
                >
                  <Icon size={16} />
                  <span>{item.label}</span>
                </button>
              );
            })}
          </nav>

          <div className="rail-section">
            <div className="rail-section-title">
              <span>图像</span>
              <small>{images.length}</small>
            </div>
            <div className="rail-list">
              {images.map((image) => (
                <button
                  className={activeImage?.image_id === image.image_id ? "active" : undefined}
                  key={image.image_id}
                  onClick={() => {
                    setActiveImage(image.image_id);
                    setActiveRun(
                      runs.find((run) => run.image_id === image.image_id)?.run_id ?? null
                    );
                    setSelectedRuns([]);
                  }}
                  title={image.filename}
                  aria-label={`${image.filename}${image.sample_id ? ` · ${image.sample_id}` : ""}`}
                >
                  <FileImage size={14} />
                  <span>
                    <strong>{image.filename}</strong>
                    <small>{image.sample_id}</small>
                  </span>
                </button>
              ))}
            </div>
          </div>

          <div className="rail-section rail-runs">
            <div className="rail-section-title">
              <span>运行</span>
              <small>{runs.length}</small>
            </div>
            <div className="rail-list">
              {runs.map((run) => (
                <div
                  className={`rail-run${activeRun?.run_id === run.run_id ? " active" : ""}`}
                  key={run.run_id}
                >
                  <button
                    onClick={() => {
                      setActiveRun(run.run_id);
                      setActiveImage(run.image_id);
                      setSelectedRuns(
                        selectedRunIds.filter((runId) =>
                          runs.some(
                            (candidate) =>
                              candidate.run_id === runId &&
                              candidate.image_id === run.image_id
                          )
                        )
                      );
                      setStage(TERMINAL_RUN_STATUSES.has(run.status) ? "results" : "runs");
                    }}
                    title={`${run.model_id} · ${run.status}`}
                    aria-label={`${run.model_id} · ${run.status}`}
                  >
                    <CircleDot size={14} />
                    <span>
                      <strong>{run.model_id}</strong>
                      <small>{run.status}</small>
                    </span>
                  </button>
                  <input
                    type="checkbox"
                    aria-label={`将 ${run.model_id} 加入比较`}
                    checked={selectedRunIds.includes(run.run_id)}
                    disabled={!COMPARABLE_RUN_STATUSES.has(run.status)}
                    onChange={() => toggleComparison(run)}
                  />
                </div>
              ))}
              {!runs.length ? <p>尚无运行</p> : null}
            </div>
          </div>
        </aside>

        <section className="active-work-canvas">
          <div className="stage-header">
            <div>
              <span>{stage.toUpperCase()}</span>
              <h1>{stages.find((item) => item.value === stage)?.label}</h1>
              <p>{stageDescription(stage, activeImage?.filename, activeRun)}</p>
            </div>
            <div className="stage-header-actions">
              {activeImage ? <code>{activeImage.filename}</code> : null}
              {activeRun ? <StatusBadge value={activeRun.status} /> : null}
            </div>
          </div>

          <div className="stage-content">
            {analysis.isRefetchError ? <RequestError error={analysis.error} /> : null}
            {launchWarning ? (
              <section className="next-action-banner launch-warning" role="alert">
                <div>
                  <CircleDot size={18} />
                  <span>
                    <strong>项目已保存，但自动分割没有启动</strong>
                    <p>{launchWarning}。可以检查模型后重试，上传的图像不会丢失。</p>
                  </span>
                </div>
                <Button tone="primary" onClick={() => setStage("models")}>
                  检查模型并重试
                </Button>
              </section>
            ) : null}
            {stage === "project" ? (
              <>
                <ProjectOverview detail={detail} />
                {!runs.length ? (
                  <>
                    <section className="first-run-guide" aria-labelledby="first-run-title">
                      <div className="first-run-heading">
                        <span>第一次使用，只需走这三步</span>
                        <h2 id="first-run-title">接下来系统会做什么</h2>
                      </div>
                      <ol>
                        <li className="complete">
                          <span><CheckCircle2 size={17} /></span>
                          <div>
                            <strong>图像已上传</strong>
                            <p>{images.length} 张图像已经校验，可以直接分析。</p>
                          </div>
                        </li>
                        <li className="current">
                          <span>2</span>
                          <div>
                            <strong>开始全图分割</strong>
                            <p>系统已准备好模型和默认参数；ROI 与调参都可以跳过。</p>
                          </div>
                        </li>
                        <li>
                          <span>3</span>
                          <div>
                            <strong>自动打开结果</strong>
                            <p>运行完成后会直接展示颗粒叠加图、统计和质量状态。</p>
                          </div>
                        </li>
                      </ol>
                    </section>
                    <section className="next-action-banner">
                      <div>
                        <Microscope size={20} />
                        <div>
                          <strong>现在只需点击一次</strong>
                          <p>
                            下一页会先显示本次分析范围、模型和参数摘要，再由你确认开始。
                          </p>
                        </div>
                      </div>
                      <Button tone="primary" onClick={() => setStage("models")}>
                        下一步：确认并开始
                      </Button>
                    </section>
                  </>
                ) : null}
              </>
            ) : null}

            {stage === "roi" ? (
              !activeImage ? (
                <EmptyState icon={FileImage} title="没有图像" detail="该任务没有可编辑的图像。" />
              ) : boxes.data ? (
                <>
                  {boxes.isRefetchError ? <RequestError error={boxes.error} /> : null}
                  <RoiEditor
                    jobId={jobId}
                    image={activeImage}
                    serverBoxes={boxes.data}
                    writeBlocker={writeBlocker}
                  />
                </>
              ) : boxes.isError ? (
                <RequestError error={boxes.error} />
              ) : (
                <div className="centered-stage"><span className="status-spinner" /></div>
              )
            ) : null}

            {stage === "models" ? (
              models.isError ? (
                <RequestError error={models.error} />
              ) : models.data ? (
                <ModelSelector
                  key={`${jobId}:${activeImage?.image_id ?? "no-image"}:roi-${boxes.data?.revision ?? "none"}`}
                  jobId={jobId}
                  image={activeImage}
                  boxSet={boxes.data || null}
                  catalog={models.data}
                  writeBlocker={writeBlocker}
                  onRunsCreated={handleRunsCreated}
                />
              ) : (
                <div className="centered-stage"><span className="status-spinner" /></div>
              )
            ) : null}

            {stage === "runs" ? (
              activeRun ? (
                <div className="run-stage">
                  <section className="run-identity panel">
                    <div>
                      <span>ACTIVE RUN</span>
                      <h2>{activeRun.model_id}</h2>
                      <code>{activeRun.run_id}</code>
                    </div>
                    <StatusBadge value={activeRun.status} />
                  </section>
                  <RunTimeline run={activeRun} />
                </div>
              ) : (
                <EmptyState
                  icon={Activity}
                  title="尚未创建运行"
                  detail="选择一个真实 ready 模型并确认参数后，执行状态会出现在这里。"
                  action={
                    <Button onClick={() => setStage("models")}>前往模型与运行</Button>
                  }
                />
              )
            ) : null}

            {stage === "results" ? (
              <ResultView
                key={activeRun?.run_id ?? "no-run"}
                jobId={jobId}
                image={activeImage}
                run={activeRun}
                availableRuns={availableRuns}
                comparisonRuns={comparisonRuns}
                writeBlocker={writeBlocker}
                onSelectRun={(runId) => setActiveRun(runId)}
                onChildCreated={(runId) => {
                  setActiveRun(runId);
                  setAutoRevealRunIds([runId]);
                  setStage("runs");
                }}
              />
            ) : null}

            {stage === "agent" ? (
              <div className="agent-stage">
                {answer ? (
                  <QueryAnswer response={answer} />
                ) : (
                  <AgentWelcome
                    imageName={activeImage?.filename ?? null}
                    runCount={composerRunIds.length}
                  />
                )}
                {queryHistory.isError &&
                !(
                  queryHistory.error instanceof NanoLoopApiError &&
                  queryHistory.error.status === 404
                ) ? (
                  <RequestError error={queryHistory.error} />
                ) : !queryHistory.isError ? (
                  <QueryHistory
                    items={scopedQueryHistory}
                    onSelect={(item) =>
                      setAnswerState({ scope: answerScope, value: item.response })
                    }
                  />
                ) : null}
              </div>
            ) : null}
          </div>
        </section>

        <ScientificInspector
          health={health.data || null}
          model={activeModel}
          run={activeRun}
          answer={answer}
        />

        <Dialog.Root open={inspectorOpen} onOpenChange={setInspectorOpen}>
          <Dialog.Trigger asChild>
            <button
              className="mobile-inspector-trigger"
              aria-label="打开科学审查器"
            >
              <PanelRight size={18} />
            </button>
          </Dialog.Trigger>
          <Dialog.Portal>
            <Dialog.Overlay className="mobile-inspector-backdrop" />
            <Dialog.Content
              className="mobile-inspector-drawer"
              aria-describedby={undefined}
            >
              <Dialog.Title className="sr-only">科学证据审查器</Dialog.Title>
              <Dialog.Close asChild>
              <button
                className="mobile-inspector-close"
                aria-label="关闭科学审查器"
              >
                <X size={18} />
              </button>
              </Dialog.Close>
              <ScientificInspector
                health={health.data || null}
                model={activeModel}
                run={activeRun}
                answer={answer}
              />
            </Dialog.Content>
          </Dialog.Portal>
        </Dialog.Root>

        {stage === "agent" ? (
          <CommandComposer
            key={answerScope}
            jobId={jobId}
            image={activeImage}
            runIds={composerRunIds}
            writeBlocker={writeBlocker}
            clarification={answer}
            onAnswer={(value, scope) => {
              if (scope !== activeAnswerScope.current) return;
              setAnswerState({ scope, value });
              setStage("agent");
              setInspectorTab("evidence");
            }}
          />
        ) : null}
      </div>
    </main>
  );
}

function stageDescription(
  stage: WorkspaceStage,
  image: string | undefined,
  run: Run | null | undefined
) {
  const model = run?.model_id || "当前模型";
  const runDescription = !run
    ? "尚未创建运行；先到“模型与运行”选择默认流程。"
    : run.status === "COMPLETED"
      ? `${model} 已完成；可以打开结果并继续形成实验结论。`
      : run.status === "COMPLETED_WITH_WARNINGS"
        ? `${model} 已完成但有警告；请先审查质量与溯源，再使用结果。`
        : run.status === "FAILED"
          ? `${model} 运行失败；请查看时间线中的错误与审计记录。`
          : `正在执行 ${model}；完成后会自动打开结果。`;

  const descriptions: Record<WorkspaceStage, string> = {
    project: "先确认图像，然后按页面中的主按钮继续；系统会告诉你下一页发生什么。",
    roi: `只有想分析局部区域时，才需要为 ${image || "当前图像"} 画框；普通全图分析请直接跳过。`,
    models: "先查看本次分析摘要，再点“开始分割”；模型、阈值和设备都已有默认值。",
    runs: runDescription,
    results: "默认显示识别叠加图，并明确区分原图、掩码和模型结果。",
    agent: "这里不是通用聊天；每个问题都限定在当前图像、所选运行和材料证据内。"
  };
  return descriptions[stage];
}
