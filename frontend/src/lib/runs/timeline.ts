export const TERMINAL_RUN_STATUSES = new Set([
  "COMPLETED",
  "COMPLETED_WITH_WARNINGS",
  "FAILED"
]);

export type TimelineStep = {
  status: string;
  label: string;
};

const steps: TimelineStep[] = [
  { status: "PREPROCESSING", label: "图像校验与 ROI 应用" },
  { status: "SEGMENTING", label: "模型分割" },
  { status: "POSTPROCESSING", label: "后处理与实例生成" },
  { status: "QUALITY_CHECKING", label: "质量门控" },
  { status: "ANALYZING", label: "形貌统计" },
  { status: "AGGREGATING", label: "汇总与报告制品" }
];

export function timelineFor(status: string) {
  const normalized = status.toUpperCase();
  const current = steps.findIndex((step) => step.status === normalized);
  const complete = normalized === "COMPLETED" || normalized === "COMPLETED_WITH_WARNINGS";
  return steps.map((step, index) => ({
    ...step,
    state:
      complete || (current >= 0 && index < current)
        ? ("complete" as const)
        : index === current
          ? ("active" as const)
          : ("pending" as const)
  }));
}
