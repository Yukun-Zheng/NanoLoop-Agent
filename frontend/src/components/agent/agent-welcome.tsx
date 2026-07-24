"use client";

import {
  BarChart3,
  BookOpenCheck,
  GitCompareArrows,
  Sparkles
} from "lucide-react";

import {
  type QueryMode,
  useWorkspaceStore
} from "@/lib/store/workspace";

const suggestions: Array<{
  icon: typeof Sparkles;
  mode: QueryMode;
  title: string;
  detail: string;
  question: string;
}> = [
  {
    icon: BarChart3,
    mode: "analysis_data",
    title: "解读当前结果",
    detail: "基于所选运行说明颗粒数量、覆盖率与质量门控。",
    question: "请解读当前运行的关键统计、质量状态和需要注意的限制。"
  },
  {
    icon: GitCompareArrows,
    mode: "analysis_data",
    title: "比较所选运行",
    detail: "并列比较最多三个同图运行，不自行判断最佳模型。",
    question: "请比较所选运行的颗粒数量、覆盖率、质量状态与运行耗时。"
  },
  {
    icon: BookOpenCheck,
    mode: "mixed",
    title: "结合材料知识",
    detail: "区分实验数据证据、知识引用和仍无法确认的部分。",
    question: "结合当前实验结果和材料知识，给出保守解释，并明确引用与限制。"
  }
];

export function AgentWelcome({
  imageName,
  runCount
}: {
  imageName: string | null;
  runCount: number;
}) {
  const setDraft = useWorkspaceStore((state) => state.setQueryDraft);
  const setMode = useWorkspaceStore((state) => state.setQueryMode);

  return (
    <section className="agent-welcome">
      <div className="agent-welcome-heading">
        <span className="empty-icon"><Sparkles size={20} /></span>
        <div>
          <span>BOUNDED EVIDENCE QUERY</span>
          <h2>这里不是通用聊天</h2>
          <p>
            每个回答都限定在 {imageName || "当前任务"}、{runCount} 个已选运行和可引用的材料知识内。
            选择一个明确问题，系统会区分实验数据、外部知识和无法确认的部分。
          </p>
        </div>
      </div>
      <div className="agent-suggestions">
        {suggestions.map((suggestion) => {
          const Icon = suggestion.icon;
          return (
            <button
              type="button"
              key={suggestion.title}
              onClick={() => {
                setMode(suggestion.mode);
                setDraft(suggestion.question);
              }}
            >
              <span><Icon size={17} /></span>
              <strong>{suggestion.title}</strong>
              <p>{suggestion.detail}</p>
            </button>
          );
        })}
      </div>
    </section>
  );
}
