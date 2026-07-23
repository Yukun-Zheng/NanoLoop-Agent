"use client";

import { useMutation, useQuery } from "@tanstack/react-query";
import {
  ArrowRight,
  BookOpen,
  Boxes,
  FileImage,
  FolderClock,
  ImagePlus,
  Layers3,
  Search,
  Sparkles,
  X
} from "lucide-react";
import Link from "next/link";
import { useRouter } from "next/navigation";
import { useEffect, useMemo, useRef, useState } from "react";

import { Brand } from "@/components/shell/brand";
import { HealthIndicator } from "@/components/shell/health-indicator";
import { Button } from "@/components/ui/button";
import { EmptyState } from "@/components/ui/empty-state";
import { RequestError } from "@/components/ui/request-error";
import { apiUpload } from "@/lib/api/client";
import { getHealth } from "@/lib/api/openapi-client";
import { queryKeys } from "@/lib/api/query-keys";
import type { JobDetail } from "@/lib/api/types";
import { analysisMetadataSchema } from "@/lib/contracts/metadata";
import { formatDate } from "@/lib/format/value";
import { coreMutationBlocker } from "@/lib/health";
import {
  readRecentJobs,
  rememberJob,
  type RecentJob
} from "@/lib/recent-jobs";

type ScaleMode = "pixel_only" | "nm_per_pixel";

type ImageDraft = {
  file: File;
  sampleId: string;
  materialName: string;
  materialFormula: string;
  scaleMode: ScaleMode;
  scaleValue: string;
  conditionKey: string;
  conditionValue: string;
};

function makeDraft(file: File): ImageDraft {
  return {
    file,
    sampleId: file.name.replace(/\.[^.]+$/, "").slice(0, 120),
    materialName: "",
    materialFormula: "",
    scaleMode: "pixel_only",
    scaleValue: "",
    conditionKey: "",
    conditionValue: ""
  };
}

export function Launchpad() {
  const router = useRouter();
  const fileInput = useRef<HTMLInputElement>(null);
  const [task, setTask] = useState("");
  const [openId, setOpenId] = useState("");
  const [drafts, setDrafts] = useState<ImageDraft[]>([]);
  const [recent, setRecent] = useState<RecentJob[]>([]);
  const [bulkMaterial, setBulkMaterial] = useState("");
  const [uploadProgress, setUploadProgress] = useState<number | null>(null);
  const health = useQuery({
    queryKey: queryKeys.health,
    queryFn: () => getHealth().then((response) => response.data),
    refetchInterval: 15_000
  });
  const writeBlocker = coreMutationBlocker(health.data, {
    failed: health.isError,
    pending: health.isPending
  });

  useEffect(() => {
    const frame = requestAnimationFrame(() => setRecent(readRecentJobs()));
    return () => cancelAnimationFrame(frame);
  }, []);

  const validation = useMemo(() => {
    if (!task.trim()) return "请填写本次分析任务名称";
    if (!drafts.length) return "请添加至少一张显微图像";
    if (new Set(drafts.map((draft) => draft.file.name)).size !== drafts.length) {
      return "同一任务中的文件名不能重复";
    }
    for (const draft of drafts) {
      if (!draft.sampleId.trim()) return `${draft.file.name} 缺少 sample_id`;
      if (draft.scaleMode === "nm_per_pixel" && Number(draft.scaleValue) <= 0) {
        return `${draft.file.name} 需要填写有效的 nm/px`;
      }
    }
    return null;
  }, [drafts, task]);

  const createAnalysis = useMutation({
    mutationFn: async () => {
      const body = new FormData();
      for (const draft of drafts) body.append("files", draft.file, draft.file.name);
      const metadata = analysisMetadataSchema.parse({
          job_name: task.trim(),
          images: drafts.map((draft) => ({
            filename: draft.file.name,
            sample_id: draft.sampleId.trim(),
            material_name: draft.materialName.trim() || null,
            material_formula: draft.materialFormula.trim() || null,
            experiment_conditions:
              draft.conditionKey.trim() && draft.conditionValue.trim()
                ? { [draft.conditionKey.trim()]: draft.conditionValue.trim() }
                : {},
            scale:
              draft.scaleMode === "nm_per_pixel"
                ? { mode: "nm_per_pixel", value: Number(draft.scaleValue) }
                : { mode: "pixel_only" }
          }))
        });
      body.append("metadata_json", JSON.stringify(metadata));
      setUploadProgress(0);
      return apiUpload<JobDetail>("analyses", body, setUploadProgress);
    },
    onSuccess(response) {
      const job = response.data.job;
      setRecent(rememberJob({ jobId: job.job_id, name: job.name }));
      router.push(`/workspace/${encodeURIComponent(job.job_id)}`);
    },
    onError() {
      setUploadProgress(null);
    }
  });

  function chooseFiles(files: FileList | null) {
    if (!files) return;
    const allowed = /\.(tif|tiff|png|jpe?g)$/i;
    const selected = [...files].filter((file) => allowed.test(file.name)).slice(0, 20);
    setDrafts(selected.map(makeDraft));
  }

  function updateDraft(index: number, patch: Partial<ImageDraft>) {
    setDrafts((current) =>
      current.map((draft, draftIndex) => (draftIndex === index ? { ...draft, ...patch } : draft))
    );
  }

  function openExisting() {
    const jobId = openId.trim();
    if (!jobId) return;
    router.push(`/workspace/${encodeURIComponent(jobId)}`);
  }

  return (
    <main className="launchpad-page">
      <header className="app-topbar">
        <Brand />
        <div className="topbar-actions">
          <HealthIndicator />
          <Button asChild tone="ghost">
            <Link href="/knowledge">
              <BookOpen size={16} />
              知识库
            </Link>
          </Button>
        </div>
      </header>

      <section className="hero">
        <div className="eyebrow">
          <Sparkles size={14} />
          材料实验智能体指挥中心
        </div>
        <h1>从显微图像到实验洞察</h1>
        <p>
          上传 SEM 图像，组织人工 ROI 与真实模型运行，并在同一个可追溯工作区中审查质量、证据和结果。
        </p>

        <section className="command-card" aria-label="创建分析任务">
          <label className="sr-only" htmlFor="task-name">
            分析任务名称
          </label>
          <textarea
            id="task-name"
            value={task}
            onChange={(event) => setTask(event.target.value)}
            placeholder="描述本次分析任务……&#10;例如：分析这组 LaNiO₃ 图像中的析出颗粒"
            maxLength={255}
          />

          {drafts.length ? (
            <div className="upload-summary">
              <div>
                <FileImage size={16} />
                <strong>{drafts.length} 张图像</strong>
                <span>
                  {new Intl.NumberFormat("zh-CN", {
                    style: "unit",
                    unit: "megabyte",
                    maximumFractionDigits: 1
                  }).format(drafts.reduce((sum, draft) => sum + draft.file.size, 0) / 1_000_000)}
                </span>
              </div>
              <button
                type="button"
                aria-label="清空已选图像"
                onClick={() => setDrafts([])}
              >
                <X size={16} />
              </button>
            </div>
          ) : null}

          {createAnalysis.isPending && uploadProgress !== null ? (
            <div
              className="upload-progress"
              role="progressbar"
              aria-label="图像上传进度"
              aria-valuemin={0}
              aria-valuemax={100}
              aria-valuenow={uploadProgress}
            >
              <span style={{ width: `${uploadProgress}%` }} />
            </div>
          ) : null}

          <div className="command-actions">
            <input
              ref={fileInput}
              className="sr-only"
              type="file"
              accept=".tif,.tiff,.png,.jpg,.jpeg,image/tiff,image/png,image/jpeg"
              multiple
              onChange={(event) => chooseFiles(event.target.files)}
            />
            <Button tone="ghost" onClick={() => fileInput.current?.click()}>
              <ImagePlus size={17} />
              添加图像
            </Button>
            <span className="command-hint">1–20 张 · TIF / PNG / JPG</span>
            <Button
              tone="primary"
              disabled={Boolean(validation || writeBlocker) || createAnalysis.isPending}
              title={validation || writeBlocker || undefined}
              onClick={() => createAnalysis.mutate()}
            >
              {createAnalysis.isPending
                ? `正在上传 ${uploadProgress ?? 0}%`
                : "创建任务"}
              <ArrowRight size={17} />
            </Button>
          </div>
        </section>

        {writeBlocker ? (
          <p className="form-warning" role="status">
            {writeBlocker}
          </p>
        ) : null}
        {createAnalysis.isError ? <RequestError error={createAnalysis.error} /> : null}

        {drafts.length ? (
          <section className="metadata-panel panel">
            <div className="metadata-toolbar">
              <div>
                <strong>逐图元数据</strong>
                <p>sample_id 必填；物理尺度缺失时只输出像素单位。</p>
              </div>
              <div className="bulk-field">
                <input
                  className="input"
                  value={bulkMaterial}
                  onChange={(event) => setBulkMaterial(event.target.value)}
                  placeholder="批量填写材料名称"
                />
                <Button
                  size="sm"
                  onClick={() =>
                    setDrafts((current) =>
                      current.map((draft) => ({ ...draft, materialName: bulkMaterial.trim() }))
                    )
                  }
                  disabled={!bulkMaterial.trim()}
                >
                  应用
                </Button>
              </div>
            </div>

            <div className="metadata-list">
              {drafts.map((draft, index) => (
                <article className="metadata-row" key={`${draft.file.name}-${draft.file.size}`}>
                  <div className="file-identity">
                    <FileImage size={17} />
                    <div>
                      <strong>{draft.file.name}</strong>
                      <span>{Math.ceil(draft.file.size / 1024)} KB</span>
                    </div>
                  </div>
                  <label className="field">
                    <span>sample_id *</span>
                    <input
                      className="input"
                      value={draft.sampleId}
                      maxLength={120}
                      onChange={(event) => updateDraft(index, { sampleId: event.target.value })}
                    />
                  </label>
                  <label className="field">
                    <span>材料名称</span>
                    <input
                      className="input"
                      value={draft.materialName}
                      onChange={(event) => updateDraft(index, { materialName: event.target.value })}
                    />
                  </label>
                  <label className="field">
                    <span>化学式</span>
                    <input
                      className="input"
                      value={draft.materialFormula}
                      onChange={(event) =>
                        updateDraft(index, { materialFormula: event.target.value })
                      }
                    />
                  </label>
                  <label className="field">
                    <span>尺度</span>
                    <select
                      className="select"
                      value={draft.scaleMode}
                      onChange={(event) =>
                        updateDraft(index, { scaleMode: event.target.value as ScaleMode })
                      }
                    >
                      <option value="pixel_only">仅像素</option>
                      <option value="nm_per_pixel">nm / px</option>
                    </select>
                  </label>
                  {draft.scaleMode === "nm_per_pixel" ? (
                    <label className="field">
                      <span>nm / px *</span>
                      <input
                        className="input"
                        value={draft.scaleValue}
                        type="number"
                        min="0"
                        step="any"
                        onChange={(event) => updateDraft(index, { scaleValue: event.target.value })}
                      />
                    </label>
                  ) : null}
                  <label className="field">
                    <span>实验条件键</span>
                    <input
                      className="input"
                      value={draft.conditionKey}
                      placeholder="temperature"
                      onChange={(event) =>
                        updateDraft(index, { conditionKey: event.target.value })
                      }
                    />
                  </label>
                  <label className="field">
                    <span>实验条件值</span>
                    <input
                      className="input"
                      value={draft.conditionValue}
                      placeholder="800 °C"
                      onChange={(event) =>
                        updateDraft(index, { conditionValue: event.target.value })
                      }
                    />
                  </label>
                </article>
              ))}
            </div>
            {validation ? <p className="form-warning">{validation}</p> : null}
          </section>
        ) : null}

        <div className="quick-intents" aria-label="常用任务">
          <button type="button" onClick={() => setTask("批量分析这组显微图像中的颗粒形貌")}>
            <Layers3 size={16} />
            批量颗粒分析
          </button>
          <button type="button" onClick={() => setTask("比较不同样品或模型的颗粒分析结果")}>
            <Boxes size={16} />
            比较不同样品
          </button>
          <Link href="/knowledge">
            <Search size={16} />
            查询材料知识
          </Link>
        </div>
      </section>

      <section className="recent-section">
        <div className="section-heading">
          <div>
            <span>LOCAL HISTORY</span>
            <h2>本机最近打开</h2>
          </div>
          <div className="open-job">
            <input
              className="input"
              value={openId}
              onChange={(event) => setOpenId(event.target.value)}
              onKeyDown={(event) => event.key === "Enter" && openExisting()}
              placeholder="输入 job_id"
              aria-label="输入 job_id 打开项目"
            />
            <Button onClick={openExisting} disabled={!openId.trim()}>
              打开
            </Button>
          </div>
        </div>

        {recent.length ? (
          <div className="recent-grid">
            {recent.map((job) => (
              <Link href={`/workspace/${encodeURIComponent(job.jobId)}`} key={job.jobId}>
                <span className="recent-icon">
                  <FolderClock size={18} />
                </span>
                <div>
                  <strong>{job.name}</strong>
                  <code>{job.jobId}</code>
                  <small>最近打开 {formatDate(job.openedAt)}</small>
                </div>
                <ArrowRight size={17} />
              </Link>
            ))}
          </div>
        ) : (
          <EmptyState
            icon={FolderClock}
            title="这台设备上还没有最近任务"
            detail="这里仅保存本机打开过的 job_id，不会伪装成服务器端项目列表。"
          />
        )}
      </section>
    </main>
  );
}
