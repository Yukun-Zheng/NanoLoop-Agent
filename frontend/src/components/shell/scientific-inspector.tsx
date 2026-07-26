"use client";

import * as Tabs from "@radix-ui/react-tabs";
import { useMutation, useQueryClient } from "@tanstack/react-query";
import {
  Activity,
  BookOpenCheck,
  Bot,
  Calculator,
  CheckCircle2,
  FlaskConical,
  Microscope,
  PanelRightClose,
  PanelRightOpen,
  Ruler,
  ShieldCheck
} from "lucide-react";
import { useMemo, useState } from "react";

import { ConversationPanel } from "@/components/agent/conversation-panel";
import { Button } from "@/components/ui/button";
import { RequestError } from "@/components/ui/request-error";
import { StatusBadge } from "@/components/ui/status-badge";
import { apiRequest } from "@/lib/api/client";
import { queryKeys } from "@/lib/api/query-keys";
import type {
  HealthData,
  ImageAsset,
  ModelMetadata,
  ReviewRunData,
  Run,
  UnifiedQueryResponse
} from "@/lib/api/types";
import { compactId, formatDate, formatNumber } from "@/lib/format/value";
import {
  type InspectorTab,
  useWorkspaceStore
} from "@/lib/store/workspace";

const tabItems: Array<{ value: InspectorTab; label: string; icon: typeof Activity }> = [
  { value: "assistant", label: "问答", icon: Bot },
  { value: "scale", label: "尺度", icon: Ruler },
  { value: "system", label: "系统", icon: Activity },
  { value: "model", label: "模型", icon: Microscope },
  { value: "quality", label: "质量", icon: ShieldCheck },
  { value: "provenance", label: "溯源", icon: FlaskConical },
  { value: "evidence", label: "证据", icon: BookOpenCheck }
];

export function ScientificInspector({
  collapsible = false,
  jobId,
  image,
  runIds,
  writeBlocker,
  health,
  model,
  run,
  answer,
  onLatestAnswer,
  onChildCreated
}: {
  collapsible?: boolean;
  jobId: string;
  image: ImageAsset | null;
  runIds: string[];
  writeBlocker: string | null;
  health: HealthData | null;
  model: ModelMetadata | null;
  run: Run | null;
  answer: UnifiedQueryResponse | null;
  onLatestAnswer: (answer: UnifiedQueryResponse | null) => void;
  onChildCreated: (runId: string) => void;
}) {
  const tab = useWorkspaceStore((state) => state.inspectorTab);
  const setTab = useWorkspaceStore((state) => state.setInspectorTab);
  const inspectorCollapsed = useWorkspaceStore((state) => state.inspectorCollapsed);
  const toggleInspector = useWorkspaceStore((state) => state.toggleInspector);
  const collapsed = collapsible && inspectorCollapsed;

  return (
    <aside
      className={`scientific-inspector${collapsed ? " collapsed" : ""}`}
      aria-label="科研助手与实验信息"
    >
      <div className="inspector-heading">
        {!collapsed ? (
          <div>
            <span>RESEARCH COPILOT</span>
            <h2>科研助手与实验信息</h2>
          </div>
        ) : null}
        {collapsible ? (
          <button
            className="inspector-collapse-button"
            type="button"
            onClick={toggleInspector}
            aria-label={collapsed ? "展开科研助手" : "折叠科研助手"}
            title={collapsed ? "展开科研助手" : "折叠科研助手"}
          >
            {collapsed ? <PanelRightOpen size={17} /> : <PanelRightClose size={17} />}
          </button>
        ) : null}
      </div>
      {!collapsed ? (
        <Tabs.Root
          className="inspector-root"
          value={tab}
          onValueChange={(value) => setTab(value as InspectorTab)}
        >
          <Tabs.List className="inspector-tabs" aria-label="科研助手分类">
            {tabItems.map((item) => {
              const Icon = item.icon;
              return (
                <Tabs.Trigger value={item.value} key={item.value} title={item.label}>
                  <Icon size={15} />
                  <span>{item.label}</span>
                </Tabs.Trigger>
              );
            })}
          </Tabs.List>
          <Tabs.Content value="assistant" className="inspector-assistant-content">
            <ConversationPanel
              jobId={jobId}
              image={image}
              runIds={runIds}
              health={health}
              writeBlocker={writeBlocker}
              onLatestAnswer={onLatestAnswer}
              variant="inspector"
            />
          </Tabs.Content>
          <Tabs.Content value="scale">
            <ScaleCalibrationInspector
              key={run?.run_id || "no-run"}
              jobId={jobId}
              image={image}
              run={run}
              writeBlocker={writeBlocker}
              onChildCreated={onChildCreated}
            />
          </Tabs.Content>
          <Tabs.Content value="system">
            <SystemInspector health={health} />
          </Tabs.Content>
          <Tabs.Content value="model">
            <ModelInspector model={model} />
          </Tabs.Content>
          <Tabs.Content value="quality">
            <QualityInspector run={run} />
          </Tabs.Content>
          <Tabs.Content value="provenance">
            <ProvenanceInspector run={run} />
          </Tabs.Content>
          <Tabs.Content value="evidence">
            <EvidenceInspector answer={answer} />
          </Tabs.Content>
        </Tabs.Root>
      ) : null}
    </aside>
  );
}

function ScaleCalibrationInspector({
  jobId,
  image,
  run,
  writeBlocker,
  onChildCreated
}: {
  jobId: string;
  image: ImageAsset | null;
  run: Run | null;
  writeBlocker: string | null;
  onChildCreated: (runId: string) => void;
}) {
  const queryClient = useQueryClient();
  const frozen = run?.configuration.scale_calibration;
  const semMetadata = run?.configuration.sem_metadata || image?.sem_metadata;
  const scaleSource = run?.configuration.scale_source || image?.scale_source || "none";
  const [physicalLength, setPhysicalLength] = useState(
    frozen ? String(frozen.physical_length_nm) : ""
  );
  const [pixelLength, setPixelLength] = useState(
    frozen ? String(frozen.pixel_length_px) : ""
  );
  const [labelText, setLabelText] = useState(frozen?.label_text || "");
  const [created, setCreated] = useState<string | null>(null);

  const validation = useMemo(() => {
    if (!physicalLength || !pixelLength) return "请输入标尺物理长度和像素长度。";
    const physical = Number(physicalLength);
    const pixels = Number(pixelLength);
    if (!Number.isFinite(physical) || physical <= 0) return "物理长度必须大于 0。";
    if (!Number.isFinite(pixels) || pixels <= 0) return "像素长度必须大于 0。";
    return null;
  }, [physicalLength, pixelLength]);
  const computed =
    validation === null ? Number(physicalLength) / Number(pixelLength) : null;

  const calibrate = useMutation({
    mutationFn: async () => {
      if (!run) throw new Error("请先选择一个已完成运行");
      if (validation) throw new Error(validation);
      return apiRequest<ReviewRunData>(`runs/${encodeURIComponent(run.run_id)}/review`, {
        method: "POST",
        body: {
          scale_calibration: {
            physical_length_nm: Number(physicalLength),
            pixel_length_px: Number(pixelLength),
            label_text: labelText.trim() || null,
            method: "manual_scale_bar"
          }
        }
      });
    },
    async onSuccess(response) {
      await queryClient.invalidateQueries({ queryKey: queryKeys.analysis(jobId) });
      setCreated(response.data.run_id);
      onChildCreated(response.data.run_id);
    }
  });

  if (!run || !image) {
    return <InspectorEmpty text="选择图像和运行后进行尺度校准。" />;
  }

  return (
    <div className="inspector-content scale-calibration">
      <div className={`inspector-callout${run.configuration.scale_nm_per_pixel ? "" : " warning"}`}>
        <StatusBadge
          value={run.configuration.scale_nm_per_pixel ? "pass" : "review_required"}
        />
        <strong>
          {run.configuration.scale_nm_per_pixel
            ? `当前运行：${formatNumber(run.configuration.scale_nm_per_pixel, 6)} nm/px${
                scaleSource === "sem_metadata" ? "（仪器自动读取）" : ""
              }`
            : "当前运行只有像素尺度"}
        </strong>
        <p>
          {scaleSource === "sem_metadata"
            ? "已直接使用原始 SEM 文件中的可信尺度；只有发现仪器记录有误时才需要人工覆盖。"
            : "标尺换算会生成新的不可变复核子运行；历史运行保持原样，报告与问答使用新运行的物理单位。"}
        </p>
      </div>

      {semMetadata ? (
        <InspectorRows
          rows={[
            ["识别来源", semMetadata.vendor ? `${semMetadata.vendor} 原始文件元数据` : "图像仪器栏"],
            ["仪器", [semMetadata.instrument_model, semMetadata.instrument_serial].filter(Boolean).join(" · ") || "未解析"],
            ["探测器", semMetadata.detector || "未解析"],
            ["加速电压", semMetadata.accelerating_voltage_kv ? `${formatNumber(semMetadata.accelerating_voltage_kv)} kV` : "未解析"],
            ["工作距离", semMetadata.working_distance_mm ? `${formatNumber(semMetadata.working_distance_mm)} mm` : "未解析"],
            ["放大倍数", semMetadata.magnification_x ? `${formatNumber(semMetadata.magnification_x)}×` : "未解析"],
            ["光阑", semMetadata.aperture_size_um ? `${formatNumber(semMetadata.aperture_size_um)} µm` : "未解析"],
            ["采集时间", semMetadata.acquired_at ? formatDate(semMetadata.acquired_at) : "未解析"],
            [
              "分割范围",
              semMetadata.footer_rect
                ? `只分析上部 0–${semMetadata.footer_rect.y1} px；底部仪器栏已排除`
                : "未检测到独立仪器栏，分析完整图像"
            ]
          ]}
        />
      ) : null}

      {frozen ? (
        <InspectorRows
          rows={[
            ["校准方法", "可见标尺人工复核"],
            ["标尺标签", frozen.label_text || "—"],
            ["物理长度", `${formatNumber(frozen.physical_length_nm)} nm`],
            ["像素长度", `${formatNumber(frozen.pixel_length_px)} px`],
            ["冻结尺度", `${formatNumber(frozen.scale_nm_per_pixel, 6)} nm/px`]
          ]}
        />
      ) : null}

      <section className="scale-calibration-form" aria-labelledby="scale-calibration-title">
        <div>
          <span>PHYSICAL CALIBRATION</span>
          <h3 id="scale-calibration-title">
            {scaleSource === "sem_metadata" ? "人工覆盖尺度（可选）" : "按原图标尺校准"}
          </h3>
          <p>
            {image.filename} · {image.width}×{image.height} px
            {scaleSource === "sem_metadata" ? " · 当前无需手工量标尺" : ""}
          </p>
        </div>
        <label className="field">
          <span>标尺物理长度（nm）</span>
          <input
            className="input"
            type="number"
            min="0"
            step="any"
            value={physicalLength}
            onChange={(event) => setPhysicalLength(event.target.value)}
            placeholder="例如 100"
          />
        </label>
        <label className="field">
          <span>标尺像素长度（px）</span>
          <input
            className="input"
            type="number"
            min="0"
            step="any"
            value={pixelLength}
            onChange={(event) => setPixelLength(event.target.value)}
            placeholder="例如 184"
          />
        </label>
        <label className="field">
          <span>原图标签（用于审计）</span>
          <input
            className="input"
            value={labelText}
            maxLength={120}
            onChange={(event) => setLabelText(event.target.value)}
            placeholder="例如 100 nm"
          />
        </label>
        <div className="scale-equation" aria-live="polite">
          <Calculator size={16} />
          <span>
            {computed === null
              ? "尺度 = 物理长度 ÷ 像素长度"
              : `尺度 = ${formatNumber(computed, 6)} nm/px`}
          </span>
        </div>
        {writeBlocker || validation ? (
          <p className="form-warning" role="status">{writeBlocker || validation}</p>
        ) : null}
        <Button
          tone="primary"
          onClick={() => calibrate.mutate()}
          disabled={Boolean(writeBlocker || validation) || calibrate.isPending}
          title={writeBlocker || validation || undefined}
        >
          <Ruler size={15} />
          {calibrate.isPending ? "正在创建…" : "应用尺度并创建复核运行"}
        </Button>
        {created ? (
          <p className="verified-message">
            <CheckCircle2 size={15} />
            已创建物理尺度运行 {compactId(created)}
          </p>
        ) : null}
        {calibrate.isError ? <RequestError error={calibrate.error} /> : null}
      </section>
    </div>
  );
}

function SystemInspector({ health }: { health: HealthData | null }) {
  if (!health) return <InspectorEmpty text="尚未取得系统健康信息。" />;
  const unavailable = {
    status: "unavailable" as const,
    detail: "该能力尚未报告状态"
  };
  const items = [
    ["任务服务", "负责载入任务、图像和分析结果", health.service],
    ["数据存储", "负责保存任务、运行记录和审计信息", health.database],
    ["分割模型", "负责模型发现、加载和推理", health.model_registry],
    ["本地知识库", "检索团队导入的文档和资料", health.rag_index],
    ["回答引擎", "把实验数据与检索证据组织成回答", health.llm_provider ?? unavailable],
    [
      "联网与文献检索",
      "查找学术题录，并按配置补充网页结果",
      health.online_research ?? unavailable
    ]
  ] as const;
  const readyCount = items.filter(([, , item]) => item.status === "healthy").length;
  return (
    <div className="inspector-content">
      <div className="inspector-callout">
        <strong>{readyCount}/{items.length} 项能力状态正常</strong>
        <p>状态有提醒时仍可展开查看具体影响；界面会说明它对当前操作意味着什么。</p>
      </div>
      {items.map(([label, purpose, item]) => (
        <article className="health-row" key={label}>
          <div>
            <strong>{label}</strong>
            <p>{purpose}。{humanHealthDetail(label, item.status, item.detail)}</p>
          </div>
          <StatusBadge value={item.status} />
        </article>
      ))}
      <details className="audit-details">
        <summary>技术诊断信息</summary>
        <div className="audit-detail-body">
          <p>后端版本：{health.version}</p>
          {items.map(([label, , item]) => (
            <p key={label}>
              <strong>{label}：</strong>
              {item.detail || "没有额外诊断信息"}
            </p>
          ))}
        </div>
      </details>
    </div>
  );
}

function ModelInspector({ model }: { model: ModelMetadata | null }) {
  if (!model) return <InspectorEmpty text="选择运行或模型后查看身份与健康信息。" />;
  return (
    <div className="inspector-content">
      <div className="inspector-callout">
        <strong>{humanModelName(model)}</strong>
        <StatusBadge value={model.status} />
        <p>{humanModelStatus(model)}</p>
      </div>
      <InspectorRows
        rows={[
          ["模型类型", humanModelFamily(model.family)],
          ["适用方向", humanModelVariant(model.variant)],
          ["推理取向", humanQualityTier(model.quality_tier)],
          ["版本", model.version],
          ["默认阈值", formatNumber(model.default_threshold)],
          ["忽略小于", `${formatNumber(model.default_min_area_px, 0)} px 的区域`],
          ["图像准备", humanProfile(model.preprocess_profile)],
          ["结果整理", humanProfile(model.postprocess_profile)]
        ]}
      />
      <details className="audit-details">
        <summary>模型编号与审计标识</summary>
        <InspectorRows
          rows={[
            ["模型编号", model.model_id],
            ["权重指纹", model.weight_sha256 || "未提供"],
            ["配置指纹", model.config_sha256 || "未提供"],
            ["适配器指纹", model.adapter_sha256 || "未提供"]
          ]}
        />
      </details>
    </div>
  );
}

function QualityInspector({ run }: { run: Run | null }) {
  if (!run) return <InspectorEmpty text="选择运行后查看质量门控。" />;
  if (!run.quality) return <InspectorEmpty text="该运行尚无质量报告。" />;
  return (
    <div className="inspector-content">
      <div className="inspector-callout">
        <StatusBadge value={run.quality.status} />
        <strong>{humanQualityStatus(run.quality.status)}</strong>
        <p>这是自动检查提示，用来帮助复核分割结果，不替代人工判断。</p>
      </div>
      <InspectorList
        title="为什么这样判断"
        items={(run.quality.reasons ?? []).map(humanDiagnostic)}
      />
      <InspectorList
        title="建议怎么处理"
        items={(run.quality.recommendations ?? []).map(humanDiagnostic)}
      />
      <InspectorRows rows={humanQualityMetrics(run.quality.metrics ?? {})} />
      <details className="audit-details">
        <summary>完整质量指标</summary>
        <pre>{JSON.stringify(run.quality.metrics, null, 2)}</pre>
      </details>
    </div>
  );
}

function ProvenanceInspector({ run }: { run: Run | null }) {
  if (!run) return <InspectorEmpty text="选择运行后查看不可变配置和执行身份。" />;
  const configuration = run.configuration;
  return (
    <div className="inspector-content">
      <InspectorRows
        rows={[
          ["运行来源", run.parent_run_id ? "基于已有结果创建的复核运行" : "直接从原图创建"],
          ["使用模型", `${configuration.model_id} · ${configuration.model_version}`],
          [
            "尺寸单位",
            configuration.scale_nm_per_pixel
              ? `${formatNumber(configuration.scale_nm_per_pixel, 6)} nm/px`
              : "只有像素尺度，未校准物理尺寸"
          ],
          [
            "标尺依据",
            configuration.scale_calibration
              ? `${formatNumber(configuration.scale_calibration.physical_length_nm)} nm 对应 ${formatNumber(configuration.scale_calibration.pixel_length_px)} px`
              : "没有人工标尺记录"
          ],
          ["分析范围", humanRoiMode(configuration.roi_mode)],
          ["分割阈值", String(run.inference.threshold ?? "使用模型默认值")],
          ["最小保留面积", `${run.inference.min_area_px} px`],
          ["实际计算设备", humanDevice(run.execution?.actual_device || run.inference.device)],
          ["运行耗时", `${formatNumber(run.runtime_ms, 0)} ms`],
          ["创建时间", formatDate(run.created_at)]
        ]}
      />
      <details className="audit-details">
        <summary>运行编号与不可变配置</summary>
        <div className="audit-detail-body">
          <p><strong>运行编号：</strong>{run.run_id}</p>
          <p><strong>父运行：</strong>{run.parent_run_id || "无"}</p>
          <p><strong>图像指纹：</strong>{configuration.image_sha256 || "未提供"}</p>
          <p><strong>ROI 修订：</strong>{String(configuration.box_revision ?? "无")}</p>
          <p><strong>随机种子：</strong>{String(run.inference.seed)}</p>
        </div>
        <pre>{JSON.stringify(configuration, null, 2)}</pre>
      </details>
      {run.execution ? (
        <details className="audit-details">
          <summary>执行运行时</summary>
          <pre>{JSON.stringify(run.execution, null, 2)}</pre>
        </details>
      ) : null}
    </div>
  );
}

function EvidenceInspector({ answer }: { answer: UnifiedQueryResponse | null }) {
  if (!answer) return <InspectorEmpty text="提出实验问题后在此审查数据证据、引用和限制。" />;
  const dataEvidence = answer.data_evidence ?? [];
  const citations = answer.citations ?? [];
  const limitations = answer.limitations ?? [];
  return (
    <div className="inspector-content">
      <div className="inspector-callout">
        <StatusBadge
          value={
            answer.outcome_code === "INSUFFICIENT_EVIDENCE"
              ? "insufficient_evidence"
              : "pass"
          }
        />
        <p>{dataEvidence.length} 组数据证据 · {citations.length} 条引用</p>
      </div>
      <InspectorRows
        rows={[
          ["回答依据", humanQueryType(answer.query_type)],
          ["证据把握", humanConfidence(answer.confidence)],
          ["需要澄清", answer.needs_clarification ? "是" : "否"]
        ]}
      />
      <InspectorList
        title="回答引用了这些来源"
        items={citations.map(
          (item) =>
            `${humanSourceType(item.source_type)}：${item.title}${item.page ? ` · 第 ${item.page} 页` : ""}`
        )}
      />
      <InspectorList
        title="回答的适用范围与限制"
        items={limitations.map(humanDiagnostic)}
      />
    </div>
  );
}

function humanHealthDetail(
  label: string,
  status: "healthy" | "degraded" | "unavailable",
  detail?: string | null
) {
  if (label === "联网与文献检索" && detail?.includes("TAVILY_API_KEY")) {
    return "学术文献检索可用；配置网页搜索密钥后还能检索通用网页。";
  }
  if (label === "回答引擎" && detail?.includes("extractive fallback")) {
    return "当前使用可审计的证据摘录模式；仍会保留引用，但回答组织能力较弱。";
  }
  if (status === "healthy") return "当前可正常使用。";
  if (status === "degraded") return "部分能力受限，但不会阻断其他正常功能。";
  return "当前不可用，相关操作可能无法完成。";
}

function humanModelName(model: ModelMetadata) {
  return `${humanModelFamily(model.family)} · ${humanModelVariant(model.variant)}`;
}

function humanModelStatus(model: ModelMetadata) {
  if (model.status === "ready") {
    return model.notes || "模型文件已校验，可以用于当前分析。";
  }
  if (model.status === "disabled") return "该模型已停用，不能创建新的分析运行。";
  if (model.status === "loading") return "模型正在载入，请稍后再试。";
  return model.health_error
    ? `模型暂不可用；具体原因已收进下方技术信息。`
    : "模型当前不可用于推理。";
}

function humanModelFamily(value: string) {
  const labels: Record<string, string> = {
    unet: "U-Net 图像分割",
    sam2: "SAM2 通用分割",
    yolo_seg: "YOLO 实例分割",
    fixture: "演示模型"
  };
  return labels[value.toLowerCase()] || value;
}

function humanModelVariant(value: string) {
  const normalized = value.toLowerCase();
  if (normalized.includes("agglomerated")) return "团聚颗粒专用";
  if (normalized.includes("large")) return "大颗粒优化";
  if (normalized.includes("small")) return "小颗粒优化";
  if (normalized.includes("general")) return "通用场景";
  if (normalized.includes("balanced")) return "均衡场景";
  return value.replaceAll("-", " ");
}

function humanQualityTier(value: string) {
  return {
    fast: "速度优先",
    balanced: "速度与精度均衡",
    accurate: "精度优先"
  }[value] || value;
}

function humanProfile(value: string) {
  const normalized = value.toLowerCase();
  if (normalized.includes("grayscale")) return "灰度归一化";
  if (normalized.includes("imagenet")) return "ImageNet 标准归一化";
  if (normalized.includes("instance")) return "实例掩码清理与编号";
  if (normalized.includes("watershed")) return "分水岭分离粘连区域";
  if (normalized.includes("binary")) return "二值掩码清理";
  return value.replaceAll("_", " ").replaceAll("-", " ");
}

function humanQualityStatus(value: string) {
  if (value === "PASS") return "自动检查未发现明显风险";
  if (value === "WARN") return "结果可用，但建议复核提示项";
  return "需要人工复核后再采用结果";
}

function humanDiagnostic(value: string) {
  const labels: Record<string, string> = {
    foreground_ratio_too_low: "识别区域占比过低，可能漏检或阈值过高",
    foreground_ratio_too_high: "识别区域占比过高，可能把背景误认为目标",
    foreground_ratio_high: "识别区域占比较高，建议检查叠加图",
    model_confidence_low: "模型对当前结果的平均把握偏低",
    small_fragment_ratio_high: "结果中小碎片比例偏高",
    roi_edge_truncation: "较多目标被 ROI 边界截断",
    physical_scale_missing_pixel_metrics_only: "缺少物理标尺，目前只能报告像素单位",
    legacy_physical_scale_not_frozen_pixel_metrics_only:
      "旧运行没有冻结物理标尺，目前只能报告像素单位",
    empty_mask: "没有识别到目标区域",
    low_confidence: "模型置信度偏低"
  };
  if (labels[value]) return labels[value];
  if (/[\u3400-\u9fff]/.test(value)) return value;
  return value.replaceAll("_", " ").replaceAll("-", " ");
}

function humanQualityMetrics(
  metrics: Record<string, number | string | null>
): Array<[string, string]> {
  const rows: Array<[string, string]> = [];
  const ratioRows: Array<[string, string]> = [
    ["foreground_ratio", "识别面积占比"],
    ["mean_confidence", "平均模型置信度"],
    ["small_fragment_ratio", "小碎片占比"],
    ["edge_touch_ratio", "ROI 边界截断占比"]
  ];
  ratioRows.forEach(([key, label]) => {
    const value = metrics[key];
    if (typeof value === "number") rows.push([label, `${formatNumber(value * 100, 1)}%`]);
  });
  const countRows: Array<[string, string]> = [
    ["candidate_instance_count", "候选目标数"],
    ["boundary_instance_count", "接触边界的目标"],
    ["excluded_border_instance_count", "排除的边界目标"]
  ];
  countRows.forEach(([key, label]) => {
    const value = metrics[key];
    if (typeof value === "number") rows.push([label, formatNumber(value, 0)]);
  });
  return rows;
}

function humanRoiMode(value: string) {
  if (value === "full_image") return "整张图像";
  if (value === "boxes") return "已保存的 ROI 选区";
  return value.replaceAll("_", " ");
}

function humanDevice(value: string) {
  const normalized = value.toLowerCase();
  if (normalized.includes("cuda")) return `NVIDIA GPU（${value}）`;
  if (normalized.includes("mps")) return "Apple GPU";
  if (normalized.includes("cpu")) return "CPU";
  if (normalized === "auto") return "自动选择最佳设备";
  return value;
}

function humanQueryType(value: string) {
  const labels: Record<string, string> = {
    auto: "系统自动判断",
    general_chat: "任务说明与操作帮助",
    analysis_data: "当前实验数据",
    material_knowledge: "本地知识库、在线文献与网页",
    mixed: "实验数据与文献综合"
  };
  return labels[value] || value;
}

function humanConfidence(value: string) {
  return {
    low: "证据有限，需谨慎采用",
    medium: "有可核验依据，仍建议复核",
    high: "证据较充分"
  }[value] || value;
}

function humanSourceType(value?: string | null) {
  const labels: Record<string, string> = {
    external_literature: "在线文献",
    external_web: "网页资料",
    pdf: "本地 PDF",
    markdown: "本地文档",
    text: "本地资料"
  };
  return value ? labels[value] || "本地知识库" : "本地知识库";
}

function InspectorRows({ rows }: { rows: Array<[string, string]> }) {
  return (
    <dl className="inspector-rows">
      {rows.map(([label, value]) => (
        <div key={label}>
          <dt>{label}</dt>
          <dd title={value}>{value}</dd>
        </div>
      ))}
    </dl>
  );
}

function InspectorList({ title, items }: { title: string; items: string[] }) {
  return (
    <section className="inspector-list">
      <h3>{title}</h3>
      {items.length ? <ul>{items.map((item) => <li key={item}>{item}</li>)}</ul> : <p>无</p>}
    </section>
  );
}

function InspectorEmpty({ text }: { text: string }) {
  return <p className="inspector-empty">{text}</p>;
}
