import {
  BookMarked,
  Braces,
  Database,
  ExternalLink,
  Info,
  Quote
} from "lucide-react";

import { StatusBadge } from "@/components/ui/status-badge";
import { toBffArtifactUrl } from "@/lib/api/client";
import type { UnifiedQueryResponse } from "@/lib/api/types";
import { compactId, formatNumber } from "@/lib/format/value";

export function QueryAnswer({ response }: { response: UnifiedQueryResponse }) {
  const dataEvidence = response.data_evidence ?? [];
  const citations = response.citations ?? [];
  const limitations = response.limitations ?? [];
  const toolCalls = response.tool_calls ?? [];
  const citationTargets = new Map(
    citations.map((citation, index) => [
      citation.citation_id,
      `citation-${safeDomToken(citation.citation_id)}-${index + 1}`
    ])
  );
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
      <div className="answer-copy" data-testid="answer-copy">
        <AnswerBody
          answer={response.answer || "后端没有返回回答正文。"}
          citationTargets={citationTargets}
        />
      </div>

      <section className="evidence-section">
        <div className="evidence-title">
          <Database size={17} />
          <h4>实验数据证据</h4>
        </div>
        {dataEvidence.length ? (
          <div className="evidence-grid">
            {dataEvidence.map((item, index) => (
              <ToolEvidenceCard
                item={item}
                key={`${item.tool_name}-${(item.source_run_ids ?? []).join("-")}-${index}`}
              />
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
            {citations.map((citation, index) => (
              <article
                id={citationTargets.get(citation.citation_id)}
                key={`${citation.citation_id}-${index}`}
              >
                <Quote size={15} />
                <div>
                  <span className="citation-marker">
                    {citationMarker(citation.citation_id)}
                  </span>
                  <strong>{citation.title}</strong>
                  <p className="citation-excerpt">{citation.excerpt}</p>
                  {citation.citation_text ? (
                    <p className="citation-text">
                      <span>规范引用</span>
                      {citation.citation_text}
                    </p>
                  ) : null}
                  <div className="citation-meta">
                    <span>来源类型 {citation.source_type || "未声明"}</span>
                    <span>
                    文档 {compactId(citation.doc_id)} · 页 {citation.page || "—"} · chunk{" "}
                    {compactId(citation.chunk_id)} · 相关度{" "}
                    {formatNumber(citation.retrieval_score, 3)}
                    </span>
                  </div>
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

type ToolEvidence = NonNullable<UnifiedQueryResponse["data_evidence"]>[number];

function AnswerBody({
  answer,
  citationTargets
}: {
  answer: string;
  citationTargets: ReadonlyMap<string, string>;
}) {
  return answer.split(/(\[C\d+\])/g).map((part, index) => {
    const citationId = /^\[(C\d+)\]$/.exec(part)?.[1];
    const target = citationId ? citationTargets.get(citationId) : undefined;
    return target ? (
      <a
        aria-label={`跳转到引用 ${citationId}`}
        className="citation-reference"
        href={`#${target}`}
        key={`${part}-${index}`}
      >
        {part}
      </a>
    ) : (
      part
    );
  });
}

function ToolEvidenceCard({ item }: { item: ToolEvidence }) {
  const rows = item.rows ?? [];
  const units = item.units ?? {};
  const columns = Array.from(new Set(rows.flatMap((row) => Object.keys(row))));
  const safeChartUrl = toBffArtifactUrl(item.chart_url);

  return (
    <article>
      <div className="evidence-card-header">
        <strong>{item.tool_name}</strong>
        <span>
          来源 {(item.source_run_ids ?? []).map((id) => compactId(id)).join("、") || "—"}
        </span>
      </div>

      <div className="evidence-json-grid">
        <section
          aria-label={`${item.tool_name} 已验证参数`}
          className="evidence-json-block"
        >
          <span>已验证参数</span>
          <pre>{jsonText(item.validated_arguments)}</pre>
        </section>
        <section aria-label={`${item.tool_name} 汇总`} className="evidence-json-block">
          <span>汇总</span>
          <pre>{jsonText(item.aggregates ?? {})}</pre>
        </section>
        <section aria-label={`${item.tool_name} 单位`} className="evidence-json-block">
          <span>单位</span>
          <pre>{jsonText(units)}</pre>
        </section>
      </div>

      {rows.length && columns.length ? (
        <div className="evidence-table-wrap">
          <table aria-label={`${item.tool_name} 数据明细`} className="evidence-table">
            <thead>
              <tr>
                {columns.map((column) => (
                  <th key={column} scope="col">
                    {column}
                    {units[column] ? ` (${units[column]})` : ""}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {rows.map((row, rowIndex) => (
                <tr key={`${item.tool_name}-row-${rowIndex}`}>
                  {columns.map((column) => (
                    <td key={column}>{valueText(row[column])}</td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      ) : (
        <p className="muted-copy">没有返回数据明细行。</p>
      )}

      {item.chart_url ? (
        safeChartUrl ? (
          <a
            className="evidence-chart-link"
            href={safeChartUrl}
            rel="noreferrer"
            target="_blank"
          >
            <ExternalLink size={13} />
            打开图表制品
          </a>
        ) : (
          <p className="warning-copy" role="status">
            图表地址未通过安全校验，未提供链接。
          </p>
        )
      ) : null}

      {(item.quality_warnings ?? []).map((warning) => (
        <p className="warning-copy" key={warning}>
          {warning}
        </p>
      ))}
    </article>
  );
}

function citationMarker(citationId: string): string {
  return citationId.startsWith("[") && citationId.endsWith("]")
    ? citationId
    : `[${citationId}]`;
}

function safeDomToken(value: string): string {
  return value.replace(/[^A-Za-z0-9_-]/g, "-") || "unknown";
}

function jsonText(value: unknown): string {
  return JSON.stringify(value, null, 2) ?? "—";
}

function valueText(value: unknown): string {
  if (value === undefined) return "—";
  if (typeof value === "string") return value;
  if (typeof value === "number" || typeof value === "boolean" || value === null) {
    return String(value);
  }
  return JSON.stringify(value);
}
