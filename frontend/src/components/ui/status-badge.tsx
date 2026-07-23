import { clsx } from "clsx";

const positive = new Set(["ready", "pass", "completed", "healthy", "enabled", "indexed"]);
const warning = new Set([
  "warn",
  "degraded",
  "completed_with_warnings",
  "processing",
  "queued",
  "preprocessing",
  "segmenting",
  "postprocessing",
  "quality_checking",
  "analyzing",
  "aggregating"
]);
const critical = new Set(["review_required", "failed", "unhealthy", "error"]);

const labels: Record<string, string> = {
  ready: "就绪",
  pass: "通过",
  completed: "已完成",
  healthy: "健康",
  enabled: "已启用",
  indexed: "已索引",
  warn: "有警告",
  degraded: "能力受限",
  completed_with_warnings: "完成但有警告",
  review_required: "需要人工复核",
  unavailable: "不可用",
  disabled: "已停用",
  failed: "失败",
  queued: "排队中",
  preprocessing: "预处理中",
  segmenting: "分割中",
  postprocessing: "后处理中",
  quality_checking: "质量检查",
  analyzing: "统计中",
  aggregating: "汇总中",
  insufficient_evidence: "证据不足"
};

export function StatusBadge({ value, label }: { value: string; label?: string }) {
  const normalized = value.toLowerCase();
  const tone = positive.has(normalized)
    ? "positive"
    : warning.has(normalized)
      ? "warning"
      : critical.has(normalized)
        ? "critical"
        : "neutral";
  return (
    <span className={clsx("status-badge", `status-${tone}`)}>
      <span className="status-dot" aria-hidden="true" />
      {label || labels[normalized] || value}
    </span>
  );
}
