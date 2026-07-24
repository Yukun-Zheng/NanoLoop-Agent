"use client";

import { useMutation, useQuery } from "@tanstack/react-query";
import {
  ArrowRight,
  BookOpen,
  Boxes,
  Check,
  Copy,
  FileImage,
  FolderClock,
  ImagePlus,
  Layers3,
  Play,
  Search,
  SlidersHorizontal,
  Sparkles,
  Trash2,
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
import { apiRequest, apiUpload } from "@/lib/api/client";
import { errorMessage } from "@/lib/api/errors";
import { getHealth } from "@/lib/api/openapi-client";
import { queryKeys } from "@/lib/api/query-keys";
import type {
  CreateRunsData,
  JobDetail,
  ModelList,
  ModelRecommendation
} from "@/lib/api/types";
import { analysisMetadataSchema } from "@/lib/contracts/metadata";
import { formatDate } from "@/lib/format/value";
import { coreMutationBlocker } from "@/lib/health";
import {
  clearRecentJobs,
  parseJobReference,
  readRecentJobs,
  removeRecentJob,
  rememberJob,
  type RecentJob
} from "@/lib/recent-jobs";
import { defaultAnalysisName } from "@/lib/runs/configuration";

type ScaleMode = "pixel_only" | "nm_per_pixel";
type ConditionType = "" | "temperature" | "duration" | "pressure" | "atmosphere";

type MaterialPreset = {
  name: string;
  formula: string;
};

const CUSTOM_MATERIAL = "__custom__";
const materialPresets: MaterialPreset[] = [
  { name: "镍", formula: "Ni" },
  { name: "二氧化硅", formula: "SiO2" },
  { name: "二氧化钛", formula: "TiO2" },
  { name: "氧化铝", formula: "Al2O3" },
  { name: "钛酸钡", formula: "BaTiO3" },
  { name: "镍酸镧", formula: "LaNiO3" }
];

const conditionOptions: Array<{
  value: ConditionType;
  label: string;
  placeholder: string;
}> = [
  { value: "", label: "暂不记录", placeholder: "" },
  { value: "temperature", label: "温度", placeholder: "例如 800 °C" },
  { value: "duration", label: "时间", placeholder: "例如 2 h" },
  { value: "pressure", label: "压力", placeholder: "例如 1 atm" },
  { value: "atmosphere", label: "气氛", placeholder: "例如 Ar / H₂" }
];

type ImageDraft = {
  file: File;
  sampleId: string;
  materialSelection: string;
  materialName: string;
  materialFormula: string;
  scaleMode: ScaleMode;
  scaleValue: string;
  conditionType: ConditionType;
  conditionValue: string;
};

type LaunchMode = "auto" | "project";

function makeDraft(file: File): ImageDraft {
  return {
    file,
    sampleId: file.name.replace(/\.[^.]+$/, "").slice(0, 120),
    materialSelection: "",
    materialName: "",
    materialFormula: "",
    scaleMode: "pixel_only",
    scaleValue: "",
    conditionType: "",
    conditionValue: ""
  };
}

function materialFromFormula(formula: string) {
  return materialPresets.find((item) => item.formula === formula) ?? null;
}

function Formula({ value }: { value: string }) {
  const parts = value.split(/(\d+)/).filter(Boolean);
  return (
    <span className="formula" aria-label={value}>
      {parts.map((part, index) =>
        /^\d+$/.test(part) ? <sub key={`${part}-${index}`}>{part}</sub> : part
      )}
    </span>
  );
}

export function Launchpad() {
  const router = useRouter();
  const fileInput = useRef<HTMLInputElement>(null);
  const copyResetTimer = useRef<number | null>(null);
  const [task, setTask] = useState("");
  const [openId, setOpenId] = useState("");
  const [openError, setOpenError] = useState<string | null>(null);
  const [copiedJobId, setCopiedJobId] = useState<string | null>(null);
  const [drafts, setDrafts] = useState<ImageDraft[]>([]);
  const [recent, setRecent] = useState<RecentJob[]>([]);
  const [bulkMaterialFormula, setBulkMaterialFormula] = useState("");
  const [uploadProgress, setUploadProgress] = useState<number | null>(null);
  const [launchStep, setLaunchStep] = useState("");
  const [fileNotice, setFileNotice] = useState<string | null>(null);
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
    return () => {
      cancelAnimationFrame(frame);
      if (copyResetTimer.current !== null) {
        window.clearTimeout(copyResetTimer.current);
      }
    };
  }, []);

  const validation = useMemo(() => {
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
  }, [drafts]);

  const createAnalysis = useMutation({
    mutationFn: async (mode: LaunchMode) => {
      let readyModels: ModelList["models"] = [];
      if (mode === "auto") {
        setLaunchStep("正在检查可用模型…");
        const catalog = await apiRequest<ModelList>("models");
        readyModels = (catalog.data.models ?? []).filter(
          (model) => model.status === "ready" && !model.health_error
        );
        if (!readyModels.length) {
          throw new Error("当前没有可运行模型。你仍可以选择“只创建项目”，稍后再配置模型。");
        }
      }

      const body = new FormData();
      for (const draft of drafts) body.append("files", draft.file, draft.file.name);
      const metadata = analysisMetadataSchema.parse({
        job_name:
          task.trim() || defaultAnalysisName(drafts.map((draft) => draft.file.name)),
        images: drafts.map((draft) => ({
          filename: draft.file.name,
          sample_id: draft.sampleId.trim(),
          material_name: draft.materialName.trim() || null,
          material_formula: draft.materialFormula.trim() || null,
          experiment_conditions:
            draft.conditionType && draft.conditionValue.trim()
              ? { [draft.conditionType]: draft.conditionValue.trim() }
              : {},
          scale:
            draft.scaleMode === "nm_per_pixel"
              ? { mode: "nm_per_pixel", value: Number(draft.scaleValue) }
              : { mode: "pixel_only" }
        }))
      });
      body.append("metadata_json", JSON.stringify(metadata));
      setLaunchStep("正在上传并校验图像…");
      setUploadProgress(0);
      const analysis = await apiUpload<JobDetail>("analyses", body, setUploadProgress);

      if (mode === "project") {
        return {
          analysis: analysis.data,
          runIds: [] as string[],
          mode,
          launchWarning: null
        };
      }

      try {
        const images = analysis.data.images ?? [];
        const firstImage = images[0];
        if (!firstImage) {
          throw new Error("后端没有返回可运行图像");
        }

        setLaunchStep("正在为图像选择合适模型…");
        let modelId = readyModels[0]?.model_id;
        try {
          const recommendation = await apiRequest<ModelRecommendation>("models/recommend", {
            method: "POST",
            body: {
              image_id: firstImage.image_id,
              roi_mode: "full_image",
              target_profile: "general",
              prefer: "accuracy",
              device: "auto"
            }
          });
          const readyIds = new Set(readyModels.map((model) => model.model_id));
          modelId =
            (recommendation.data.candidates ?? []).find((candidate) =>
              readyIds.has(candidate.model_id)
            )?.model_id || modelId;
        } catch {
          // Recommendation is an optimization. A verified ready model is a safe fallback.
        }
        if (!modelId) throw new Error("没有可用于自动分割的模型");

        setLaunchStep(`正在创建 ${images.length} 个分割运行…`);
        const runs = await apiRequest<CreateRunsData>(
          `analyses/${encodeURIComponent(analysis.data.job.job_id)}/runs`,
          {
            method: "POST",
            body: {
              image_ids: images.map((image) => image.image_id),
              model_ids: [modelId],
              roi_mode: "full_image",
              inference: {
                watershed_enabled: false,
                exclude_border: true,
                device: "auto",
                seed: 42
              }
            }
          }
        );
        return {
          analysis: analysis.data,
          runIds: runs.data.run_ids,
          mode,
          launchWarning: null
        };
      } catch (error) {
        return {
          analysis: analysis.data,
          runIds: [] as string[],
          mode,
          launchWarning: errorMessage(error)
        };
      }
    },
    onSuccess(result) {
      const job = result.analysis.job;
      setRecent(rememberJob({ jobId: job.job_id, name: job.name }));
      const firstRun = result.runIds[0];
      const query =
        result.mode === "auto" && firstRun
          ? `?autorun=1&run=${encodeURIComponent(firstRun)}`
          : result.launchWarning
            ? `?autostart_failed=${encodeURIComponent(result.launchWarning)}`
          : "";
      router.push(`/workspace/${encodeURIComponent(job.job_id)}${query}`);
    },
    onError() {
      setUploadProgress(null);
      setLaunchStep("");
    }
  });

  function chooseFiles(files: FileList | null) {
    if (!files) return;
    const allowed = /\.(tif|tiff|png|jpe?g)$/i;
    const allFiles = [...files];
    const supported = allFiles.filter((file) => allowed.test(file.name));
    const selected = supported.slice(0, 20);
    const notices: string[] = [];
    if (supported.length !== allFiles.length) {
      notices.push(`已忽略 ${allFiles.length - supported.length} 个不支持的文件`);
    }
    if (supported.length > 20) notices.push("一次最多处理 20 张图像");
    setFileNotice(
      selected.length
        ? notices.join("；") || null
        : "没有找到可用图像，请选择 TIF、PNG 或 JPG 文件"
    );
    setDrafts(selected.map(makeDraft));
    setUploadProgress(null);
    setLaunchStep("");
  }

  function updateDraft(index: number, patch: Partial<ImageDraft>) {
    setDrafts((current) =>
      current.map((draft, draftIndex) => (draftIndex === index ? { ...draft, ...patch } : draft))
    );
  }

  function openExisting() {
    const jobId = parseJobReference(openId);
    if (!jobId) {
      setOpenError("请粘贴完整任务链接或有效的 job_id");
      return;
    }
    setOpenError(null);
    router.push(`/workspace/${encodeURIComponent(jobId)}`);
  }

  async function copyJobId(jobId: string) {
    await navigator.clipboard.writeText(jobId);
    setCopiedJobId(jobId);
    if (copyResetTimer.current !== null) {
      window.clearTimeout(copyResetTimer.current);
    }
    copyResetTimer.current = window.setTimeout(() => {
      setCopiedJobId((current) => (current === jobId ? null : current));
      copyResetTimer.current = null;
    }, 1600);
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
          <div className="quick-start-heading">
            <div>
              <Play size={17} />
              <strong>新建颗粒分析</strong>
            </div>
            <span>只需要选择图像，其余都可以稍后补充</span>
          </div>
          <label className="task-name-field" htmlFor="task-name">
            <span>
              <strong>任务名称</strong>
              <small>选填，仅用于以后找到这次分析</small>
            </span>
            <input
              id="task-name"
              className="input"
              value={task}
              onChange={(event) => setTask(event.target.value)}
              placeholder="例如：BaNi-3 颗粒分割"
              maxLength={255}
              aria-label="任务名称"
            />
          </label>

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
                onClick={() => {
                  setDrafts([]);
                  setFileNotice(null);
                  setUploadProgress(null);
                  setLaunchStep("");
                }}
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
          {createAnalysis.isPending && launchStep ? (
            <p className="launch-step" role="status">
              {launchStep}
            </p>
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
              {drafts.length ? "重新选择" : "添加图像"}
            </Button>
            <span className="command-hint">1–20 张 · TIF / PNG / JPG</span>
            {drafts.length ? (
              <Button
                tone="ghost"
                disabled={Boolean(validation || writeBlocker) || createAnalysis.isPending}
                title={validation || writeBlocker || undefined}
                onClick={() => createAnalysis.mutate("project")}
              >
                只创建项目
              </Button>
            ) : null}
            <Button
              tone="primary"
              disabled={
                Boolean((drafts.length ? validation : null) || writeBlocker) ||
                createAnalysis.isPending
              }
              title={(drafts.length ? validation : null) || writeBlocker || undefined}
              onClick={() =>
                drafts.length
                  ? createAnalysis.mutate("auto")
                  : fileInput.current?.click()
              }
            >
              {createAnalysis.isPending
                ? "正在准备…"
                : drafts.length
                  ? `自动分割 ${drafts.length} 张图像`
                  : "选择图像开始"}
              <ArrowRight size={17} />
            </Button>
          </div>
        </section>

        {fileNotice ? (
          <p className="form-warning" role="status">
            {fileNotice}
          </p>
        ) : null}
        {writeBlocker ? (
          <p className="form-warning" role="status">
            {writeBlocker}
          </p>
        ) : null}
        {createAnalysis.isError ? <RequestError error={createAnalysis.error} /> : null}

        {drafts.length ? (
          <details className="metadata-panel panel">
            <summary>
              <SlidersHorizontal size={17} />
              <div>
                <strong>补充样品信息（全部选填）</strong>
                <p>不展开也能直接分割；材料、实验条件和物理尺度只用于结果解释。</p>
              </div>
              <span>展开填写</span>
            </summary>
            {drafts.length > 1 ? (
              <div className="metadata-toolbar">
                <div>
                  <strong>这些图像来自同一种材料？</strong>
                  <p>选择一次即可应用到全部 {drafts.length} 张图；不同材料请在下方逐张选择。</p>
                </div>
                <div className="bulk-field">
                  <select
                    className="select"
                    value={bulkMaterialFormula}
                    onChange={(event) => setBulkMaterialFormula(event.target.value)}
                    aria-label="选择全部图像的材料"
                  >
                    <option value="">不批量设置</option>
                    {materialPresets.map((material) => (
                      <option value={material.formula} key={material.formula}>
                        {material.name}（{material.formula}）
                      </option>
                    ))}
                  </select>
                  <Button
                    size="sm"
                    onClick={() => {
                      const material = materialFromFormula(bulkMaterialFormula);
                      if (!material) return;
                      setDrafts((current) =>
                        current.map((draft) => ({
                          ...draft,
                          materialSelection: material.formula,
                          materialName: material.name,
                          materialFormula: material.formula
                        }))
                      );
                    }}
                    disabled={!bulkMaterialFormula}
                  >
                    应用到全部
                  </Button>
                </div>
              </div>
            ) : null}

            <div className="metadata-list">
              {drafts.map((draft, index) => (
                <article className="metadata-row" key={`${draft.file.name}-${draft.file.size}`}>
                  <header className="metadata-row-heading">
                    <div className="file-identity">
                      <FileImage size={17} />
                      <div>
                        <strong>{draft.file.name}</strong>
                        <span>{Math.ceil(draft.file.size / 1024)} KB</span>
                      </div>
                    </div>
                    <span>已满足运行条件</span>
                  </header>
                  <label className="field">
                    <span>
                      <strong>样品编号</strong>
                      <small>已从文件名自动生成，只用于区分图像</small>
                    </span>
                    <input
                      className="input"
                      value={draft.sampleId}
                      maxLength={120}
                      onChange={(event) => updateDraft(index, { sampleId: event.target.value })}
                    />
                  </label>
                  <label className="field">
                    <span>
                      <strong>材料</strong>
                      <small>选填；常用材料会自动带出化学式</small>
                    </span>
                    <select
                      className="select"
                      value={draft.materialSelection}
                      aria-label={`选择 ${draft.file.name} 的材料`}
                      onChange={(event) => {
                        const selection = event.target.value;
                        const material = materialFromFormula(selection);
                        updateDraft(index, {
                          materialSelection: selection,
                          materialName: material?.name ?? "",
                          materialFormula: material?.formula ?? ""
                        });
                      }}
                    >
                      <option value="">暂不填写</option>
                      {materialPresets.map((material) => (
                        <option value={material.formula} key={material.formula}>
                          {material.name}（{material.formula}）
                        </option>
                      ))}
                      <option value={CUSTOM_MATERIAL}>其他材料…</option>
                    </select>
                  </label>
                  {draft.materialSelection &&
                  draft.materialSelection !== CUSTOM_MATERIAL ? (
                    <div className="selected-material" aria-label="已选择材料">
                      <span>已自动填写</span>
                      <strong>{draft.materialName}</strong>
                      <Formula value={draft.materialFormula} />
                    </div>
                  ) : null}
                  {draft.materialSelection === CUSTOM_MATERIAL ? (
                    <div className="custom-material-fields">
                      <label className="field">
                        <span>
                          <strong>材料名称</strong>
                          <small>只填名称也可以</small>
                        </span>
                        <input
                          className="input"
                          value={draft.materialName}
                          onChange={(event) =>
                            updateDraft(index, { materialName: event.target.value })
                          }
                          placeholder="例如：自定义复合材料"
                        />
                      </label>
                      <label className="field">
                        <span>
                          <strong>化学式</strong>
                          <small>可粘贴，留空不影响分割</small>
                        </span>
                        <input
                          className="input"
                          value={draft.materialFormula}
                          onChange={(event) =>
                            updateDraft(index, { materialFormula: event.target.value })
                          }
                          placeholder="例如：BaTiO3"
                        />
                      </label>
                    </div>
                  ) : null}
                  <label className="field">
                    <span>
                      <strong>像素对应实际长度</strong>
                      <small>不知道时保持默认，结果仍会给出像素统计</small>
                    </span>
                    <select
                      className="select"
                      value={draft.scaleMode}
                      onChange={(event) =>
                        updateDraft(index, { scaleMode: event.target.value as ScaleMode })
                      }
                    >
                      <option value="pixel_only">不知道 / 仅使用像素</option>
                      <option value="nm_per_pixel">我知道 nm / px</option>
                    </select>
                  </label>
                  {draft.scaleMode === "nm_per_pixel" ? (
                    <label className="field">
                      <span>
                        <strong>每像素纳米数</strong>
                        <small>输入显微镜标定值</small>
                      </span>
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
                    <span>
                      <strong>实验条件</strong>
                      <small>选填；本页最多记录一个主要条件</small>
                    </span>
                    <select
                      className="select"
                      value={draft.conditionType}
                      onChange={(event) =>
                        updateDraft(index, {
                          conditionType: event.target.value as ConditionType,
                          conditionValue: ""
                        })
                      }
                    >
                      {conditionOptions.map((option) => (
                        <option value={option.value} key={option.value || "none"}>
                          {option.label}
                        </option>
                      ))}
                    </select>
                  </label>
                  {draft.conditionType ? (
                    <label className="field">
                      <span>
                        <strong>
                          {
                            conditionOptions.find(
                              (option) => option.value === draft.conditionType
                            )?.label
                          }
                        </strong>
                        <small>数值和单位写在一起</small>
                      </span>
                      <input
                        className="input"
                        value={draft.conditionValue}
                        placeholder={
                          conditionOptions.find(
                            (option) => option.value === draft.conditionType
                          )?.placeholder
                        }
                        onChange={(event) =>
                          updateDraft(index, { conditionValue: event.target.value })
                        }
                      />
                    </label>
                  ) : null}
                </article>
              ))}
            </div>
            {validation ? <p className="form-warning">{validation}</p> : null}
          </details>
        ) : null}

        <section className="workflow-strip" aria-label="分析工作流">
          <article>
            <span>01</span>
            <Layers3 size={18} />
            <div>
              <strong>组织实验输入</strong>
              <p>集中录入图像、样品身份、材料信息与物理尺度。</p>
            </div>
          </article>
          <article>
            <span>02</span>
            <Boxes size={18} />
            <div>
              <strong>运行与并列比较</strong>
              <p>在同一图像和 ROI 范围内创建可追溯模型运行。</p>
            </div>
          </article>
          <article>
            <span>03</span>
            <Search size={18} />
            <div>
              <strong>审查证据边界</strong>
              <p>先看质量与溯源，再形成带引用和限制的结论。</p>
            </div>
          </article>
        </section>
      </section>

      <section className="recent-section">
        <div className="section-heading">
          <div>
            <span>LOCAL HISTORY</span>
            <h2>本机最近打开</h2>
            <p>直接打开最近任务；完整编号可一键复制，不需要手动录入。</p>
          </div>
          {recent.length ? (
            <Button
              size="sm"
              tone="ghost"
              onClick={() => setRecent(clearRecentJobs())}
              title="只清空这台设备上的打开记录，不删除服务器任务"
            >
              <Trash2 size={14} />
              清空记录
            </Button>
          ) : null}
        </div>

        <div className="recent-layout">
          <div className="recent-list-area">
            {recent.length ? (
              <div className="recent-grid">
                {recent.map((job) => (
                  <article className="recent-card" key={job.jobId}>
                    <Link
                      className="recent-card-main"
                      href={`/workspace/${encodeURIComponent(job.jobId)}`}
                    >
                      <span className="recent-icon">
                        <FolderClock size={18} />
                      </span>
                      <div>
                        <strong>{job.name}</strong>
                        <code title={job.jobId}>{compactJobId(job.jobId)}</code>
                        <small>最近打开 {formatDate(job.openedAt)}</small>
                      </div>
                      <ArrowRight size={17} />
                    </Link>
                    <div className="recent-card-actions">
                      <button
                        type="button"
                        onClick={() => void copyJobId(job.jobId)}
                        aria-label={`复制 ${job.name} 的任务 ID`}
                        title="复制完整任务 ID"
                      >
                        {copiedJobId === job.jobId ? (
                          <Check size={14} />
                        ) : (
                          <Copy size={14} />
                        )}
                        {copiedJobId === job.jobId ? "已复制" : "复制 ID"}
                      </button>
                      <button
                        type="button"
                        className="recent-remove"
                        onClick={() => setRecent(removeRecentJob(job.jobId))}
                        aria-label={`从最近记录移除 ${job.name}`}
                        title="仅移除本机记录，不删除服务器任务"
                      >
                        <Trash2 size={14} />
                        移除
                      </button>
                    </div>
                  </article>
                ))}
              </div>
            ) : (
              <EmptyState
                icon={FolderClock}
                title="这台设备上还没有最近任务"
                detail="这里仅保存本机打开过的 job_id，不会伪装成服务器端项目列表。"
              />
            )}
          </div>
          <aside className="direct-open-card">
            <span className="recent-icon"><FolderClock size={18} /></span>
            <div>
              <strong>从分享内容打开</strong>
              <p>粘贴完整任务链接或 job_id；无需逐字输入长编号。</p>
            </div>
            <div className="open-job">
              <input
                className="input"
                value={openId}
                onChange={(event) => {
                  setOpenId(event.target.value);
                  setOpenError(null);
                }}
                onKeyDown={(event) => event.key === "Enter" && openExisting()}
                placeholder="粘贴任务链接或完整 ID"
                aria-label="粘贴任务链接或完整 ID"
              />
              <Button onClick={openExisting} disabled={!openId.trim()}>
                打开
              </Button>
            </div>
            {openError ? <p className="recent-open-error" role="alert">{openError}</p> : null}
          </aside>
        </div>
      </section>
    </main>
  );
}

function compactJobId(jobId: string) {
  if (jobId.length <= 22) return jobId;
  return `${jobId.slice(0, 12)}…${jobId.slice(-6)}`;
}
