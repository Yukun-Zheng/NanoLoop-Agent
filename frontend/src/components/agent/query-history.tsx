"use client";

import { History } from "lucide-react";

import type { QueryHistoryData } from "@/lib/api/types";

type QueryRecord = NonNullable<QueryHistoryData["items"]>[number];

const modeLabels: Record<string, string> = {
  auto: "自动",
  analysis_data: "数据",
  material_knowledge: "知识",
  mixed: "混合"
};

export function QueryHistory({
  items,
  onSelect
}: {
  items: QueryRecord[];
  onSelect: (item: QueryRecord) => void;
}) {
  return (
    <section className="agent-history" aria-label="当前作用域问答历史">
      <div className="agent-history-heading">
        <div>
          <History size={16} />
          <div>
            <span>SCOPED MEMORY</span>
            <h3>当前作用域问答历史</h3>
          </div>
        </div>
        <small>{items.length} 条审计记录</small>
      </div>

      {items.length ? (
        <div className="agent-history-list">
          {[...items].reverse().map((item) => (
            <button
              type="button"
              key={item.query_id}
              onClick={() => onSelect(item)}
              aria-label={`查看历史问答：${item.request.question}`}
            >
              <span className="agent-history-meta">
                <b>{modeLabels[item.request.query_type] || item.request.query_type}</b>
                <time dateTime={item.created_at}>{formatTimestamp(item.created_at)}</time>
              </span>
              <strong>{item.request.question}</strong>
              <small>
                {item.response.outcome_code === "INSUFFICIENT_EVIDENCE"
                  ? "证据不足"
                  : `置信度 ${item.response.confidence}`}
              </small>
            </button>
          ))}
        </div>
      ) : (
        <p className="muted-copy">
          当前图像与运行作用域还没有已提交问答；新回答会写入数据库审计并显示在这里。
        </p>
      )}
    </section>
  );
}

function formatTimestamp(value: string): string {
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return value;
  return new Intl.DateTimeFormat("zh-CN", {
    month: "2-digit",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit"
  }).format(date);
}
