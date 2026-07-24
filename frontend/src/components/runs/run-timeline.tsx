import {
  AlertCircle,
  CalendarDays,
  Check,
  Circle,
  Clock3,
  LoaderCircle
} from "lucide-react";

import type { Run } from "@/lib/api/types";
import { formatDate, formatNumber } from "@/lib/format/value";
import { timelineFor } from "@/lib/runs/timeline";

export function RunTimeline({ run }: { run: Run }) {
  const failed = run.status === "FAILED";
  const history = run.status_history ?? [];
  return (
    <div className="run-timeline">
      <div className="timeline-intro">
        <span>AGENT EXECUTION</span>
        <h3>智能体执行时间线</h3>
        <p>进度完全来自后端状态，不使用前端计时器补造百分比。</p>
        <div className="timeline-meta">
          <span><Clock3 size={13} />{formatNumber(run.runtime_ms, 0)} ms</span>
          <span><CalendarDays size={13} />更新于 {formatDate(run.updated_at)}</span>
        </div>
      </div>
      <ol>
        {timelineFor(run.status).map((step) => (
          <li className={`timeline-${step.state}`} key={step.status}>
            <span className="timeline-marker" aria-hidden="true">
              {step.state === "complete" ? (
                <Check size={14} />
              ) : step.state === "active" ? (
                <LoaderCircle size={14} />
              ) : (
                <Circle size={10} />
              )}
            </span>
            <div>
              <strong>{step.label}</strong>
              <small>{step.status}</small>
            </div>
          </li>
        ))}
        {failed ? (
          <li className="timeline-failed">
            <span className="timeline-marker">
              <AlertCircle size={14} />
            </span>
            <div>
              <strong>{run.error_code || "运行失败"}</strong>
              <small>{run.error_message || "后端没有提供更多错误信息"}</small>
            </div>
          </li>
        ) : null}
      </ol>
      {history.length ? (
        <details className="audit-details">
          <summary>查看 {history.length} 条状态审计记录</summary>
          <div className="audit-list">
            {history.map((event) => (
              <div key={event.event_id}>
                <code>{event.from_status || "NEW"} → {event.to_status}</code>
                <span>{formatDate(event.created_at)}</span>
              </div>
            ))}
          </div>
        </details>
      ) : null}
    </div>
  );
}
