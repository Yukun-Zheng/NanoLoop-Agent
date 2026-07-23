import { BookMarked, Braces, Database, Info, Quote } from "lucide-react";

import { StatusBadge } from "@/components/ui/status-badge";
import type { UnifiedQueryResponse } from "@/lib/api/types";
import { compactId, formatNumber } from "@/lib/format/value";

export function QueryAnswer({ response }: { response: UnifiedQueryResponse }) {
  const dataEvidence = response.data_evidence ?? [];
  const citations = response.citations ?? [];
  const limitations = response.limitations ?? [];
  const toolCalls = response.tool_calls ?? [];
  return (
    <div className="query-answer">
      <div className="answer-summary">
        <div>
          <span>AGENT ANSWER</span>
          <h3>NanoLoop 回答</h3>
        </div>
        <StatusBadge
          value={
            response.outcome_code === "INSUFFICIENT_EVIDENCE"
              ? "insufficient_evidence"
              : "pass"
          }
          label={
            response.outcome_code === "INSUFFICIENT_EVIDENCE"
              ? "证据不足"
              : `置信度 ${response.confidence}`
          }
        />
      </div>
      <p className="answer-copy">{response.answer || "后端没有返回回答正文。"}</p>

      <section className="evidence-section">
        <div className="evidence-title">
          <Database size={17} />
          <h4>实验数据证据</h4>
        </div>
        {dataEvidence.length ? (
          <div className="evidence-grid">
            {dataEvidence.map((item, index) => (
              <article key={`${item.tool_name}-${index}`}>
                <strong>{item.tool_name}</strong>
                <span>
                  来源 {(item.source_run_ids ?? []).map((id) => compactId(id)).join("、") || "—"}
                </span>
                <pre>{JSON.stringify(item.aggregates, null, 2)}</pre>
                {(item.quality_warnings ?? []).map((warning) => (
                  <p className="warning-copy" key={warning}>{warning}</p>
                ))}
              </article>
            ))}
          </div>
        ) : (
          <p className="muted-copy">没有返回确定性数据工具证据。</p>
        )}
      </section>

      <section className="evidence-section">
        <div className="evidence-title">
          <BookMarked size={17} />
          <h4>材料知识证据</h4>
        </div>
        {citations.length ? (
          <div className="citation-list">
            {citations.map((citation) => (
              <article key={citation.citation_id}>
                <Quote size={15} />
                <div>
                  <strong>{citation.title}</strong>
                  <p>{citation.excerpt}</p>
                  <span>
                    文档 {compactId(citation.doc_id)} · 页 {citation.page || "—"} · chunk{" "}
                    {compactId(citation.chunk_id)} · 相关度{" "}
                    {formatNumber(citation.retrieval_score, 3)}
                  </span>
                </div>
              </article>
            ))}
          </div>
        ) : (
          <p className="muted-copy">没有返回可定位的材料知识引用。</p>
        )}
      </section>

      <section className="limitations">
        <div className="evidence-title">
          <Info size={17} />
          <h4>限制</h4>
        </div>
        {limitations.length ? (
          <ul>{limitations.map((item) => <li key={item}>{item}</li>)}</ul>
        ) : (
          <p>后端未声明额外限制。</p>
        )}
      </section>

      {toolCalls.length ? (
        <details className="audit-details">
          <summary><Braces size={14} />查看工具调用审计日志</summary>
          <pre>{JSON.stringify(toolCalls, null, 2)}</pre>
        </details>
      ) : null}
    </div>
  );
}
