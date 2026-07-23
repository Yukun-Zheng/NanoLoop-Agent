"use client";

import * as Tabs from "@radix-ui/react-tabs";
import { Activity, BookOpenCheck, FlaskConical, Microscope, ShieldCheck } from "lucide-react";

import { StatusBadge } from "@/components/ui/status-badge";
import type {
  HealthData,
  ModelMetadata,
  Run,
  UnifiedQueryResponse
} from "@/lib/api/types";
import { compactId, formatDate, formatNumber } from "@/lib/format/value";
import {
  type InspectorTab,
  useWorkspaceStore
} from "@/lib/store/workspace";

const tabItems: Array<{ value: InspectorTab; label: string; icon: typeof Activity }> = [
  { value: "system", label: "系统", icon: Activity },
  { value: "model", label: "模型", icon: Microscope },
  { value: "quality", label: "质量", icon: ShieldCheck },
  { value: "provenance", label: "溯源", icon: FlaskConical },
  { value: "evidence", label: "证据", icon: BookOpenCheck }
];

export function ScientificInspector({
  health,
  model,
  run,
  answer
}: {
  health: HealthData | null;
  model: ModelMetadata | null;
  run: Run | null;
  answer: UnifiedQueryResponse | null;
}) {
  const tab = useWorkspaceStore((state) => state.inspectorTab);
  const setTab = useWorkspaceStore((state) => state.setInspectorTab);

  return (
    <aside className="scientific-inspector" aria-label="科学证据审查器">
      <div className="inspector-heading">
        <span>SCIENTIFIC INSPECTOR</span>
        <h2>科学证据审查器</h2>
      </div>
      <Tabs.Root value={tab} onValueChange={(value) => setTab(value as InspectorTab)}>
        <Tabs.List className="inspector-tabs" aria-label="审查器分类">
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
    </aside>
  );
}

function SystemInspector({ health }: { health: HealthData | null }) {
  if (!health) return <InspectorEmpty text="尚未取得系统健康信息。" />;
  const items = [
    ["服务", health.service],
    ["数据库", health.database],
    ["模型注册表", health.model_registry],
    ["RAG 索引", health.rag_index]
  ] as const;
  return (
    <div className="inspector-content">
      <div className="inspector-version">Backend {health.version}</div>
      {items.map(([label, item]) => (
        <article className="health-row" key={label}>
          <div>
            <strong>{label}</strong>
            <p>{item.detail || "未报告额外说明"}</p>
          </div>
          <StatusBadge value={item.status} />
        </article>
      ))}
    </div>
  );
}

function ModelInspector({ model }: { model: ModelMetadata | null }) {
  if (!model) return <InspectorEmpty text="选择运行或模型后查看身份与健康信息。" />;
  return (
    <div className="inspector-content">
      <div className="inspector-callout">
        <strong>{model.model_id}</strong>
        <StatusBadge value={model.status} />
        <p>{model.health_error || model.notes || "没有额外模型说明。"}</p>
      </div>
      <InspectorRows
        rows={[
          ["家族 / 变体", `${model.family} / ${model.variant}`],
          ["质量层级", model.quality_tier],
          ["版本", model.version],
          ["默认阈值", formatNumber(model.default_threshold)],
          ["默认最小面积", `${formatNumber(model.default_min_area_px, 0)} px`],
          ["预处理", model.preprocess_profile],
          ["后处理", model.postprocess_profile],
          ["权重 SHA", model.weight_sha256 || "—"],
          ["配置 SHA", model.config_sha256 || "—"],
          ["Adapter SHA", model.adapter_sha256 || "—"]
        ]}
      />
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
        <p>质量状态来自后端 canonical gate。</p>
      </div>
      <InspectorList title="判断依据" items={run.quality.reasons ?? []} />
      <InspectorList title="建议" items={run.quality.recommendations ?? []} />
      <details className="audit-details">
        <summary>质量指标详情</summary>
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
          ["run_id", run.run_id],
          ["父运行", run.parent_run_id || "—"],
          ["模型", `${configuration.model_id} · ${configuration.model_version}`],
          ["图像 SHA", configuration.image_sha256 || "—"],
          ["ROI 模式", configuration.roi_mode],
          ["ROI revision", String(configuration.box_revision ?? "—")],
          ["threshold", String(run.inference.threshold ?? "模型默认")],
          ["min area", `${run.inference.min_area_px} px`],
          ["device", run.execution?.actual_device || run.inference.device],
          ["seed", String(run.inference.seed)],
          ["runtime", `${formatNumber(run.runtime_ms, 0)} ms`],
          ["创建时间", formatDate(run.created_at)]
        ]}
      />
      <details className="audit-details">
        <summary>完整运行配置</summary>
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
          ["查询类型", answer.query_type],
          ["置信度", answer.confidence],
          ["需要澄清", answer.needs_clarification ? "是" : "否"]
        ]}
      />
      <InspectorList
        title="引用定位"
        items={citations.map(
          (item) =>
            `${item.title} · p.${item.page || "—"} · ${compactId(item.chunk_id)}`
        )}
      />
      <InspectorList title="限制" items={limitations} />
    </div>
  );
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
