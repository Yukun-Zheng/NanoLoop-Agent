"use client";

import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import {
  CheckCircle2,
  Download,
  Eye,
  FileText,
  RefreshCcw,
  Sparkles
} from "lucide-react";
import { useMemo, useState } from "react";

import { Button } from "@/components/ui/button";
import { RequestError } from "@/components/ui/request-error";
import { StatusBadge } from "@/components/ui/status-badge";
import {
  apiRequest,
  fetchArtifact,
  toBffArtifactUrl
} from "@/lib/api/client";
import { queryKeys } from "@/lib/api/query-keys";
import type {
  ReportArtifactData,
  Run,
  ScientificReportData
} from "@/lib/api/types";
import { sha256Hex } from "@/lib/crypto/sha256";
import { compactId, formatDate, formatNumber } from "@/lib/format/value";

const terminal = new Set(["COMPLETED", "COMPLETED_WITH_WARNINGS"]);

export function ScientificReport({
  jobId,
  runs,
  writeBlocker
}: {
  jobId: string;
  runs: Run[];
  writeBlocker: string | null;
}) {
  const queryClient = useQueryClient();
  const selectedRuns = useMemo(
    () =>
      runs
        .filter(
          (run, index, candidates) =>
            terminal.has(run.status) &&
            candidates.findIndex((candidate) => candidate.run_id === run.run_id) === index
        )
        .slice(0, 20),
    [runs]
  );
  const selectedImageCount = useMemo(
    () => new Set(selectedRuns.map((run) => run.image_id)).size,
    [selectedRuns]
  );
  const selectedRunIds = useMemo(
    () => selectedRuns.map((run) => run.run_id),
    [selectedRuns]
  );
  const reportKey = useMemo(
    () => queryKeys.scientificReport(jobId, selectedRunIds),
    [jobId, selectedRunIds]
  );
  const reportSnapshot = useQuery<ScientificReportData>({
    queryKey: reportKey,
    queryFn: async () => {
      throw new Error("报告只能由用户主动生成");
    },
    enabled: false,
    staleTime: Number.POSITIVE_INFINITY,
    gcTime: Number.POSITIVE_INFINITY
  });
  const report = reportSnapshot.data ?? null;
  const [previewOpen, setPreviewOpen] = useState(false);
  const [downloadStatus, setDownloadStatus] = useState<string | null>(null);

  const generate = useMutation({
    mutationFn: () =>
      apiRequest<ScientificReportData>(
        `analyses/${encodeURIComponent(jobId)}/report`,
        {
          method: "POST",
          body: { run_ids: selectedRunIds }
        }
      ),
    onSuccess(response) {
      queryClient.setQueryData(
        queryKeys.scientificReport(jobId, response.data.selected_run_ids),
        response.data
      );
      setPreviewOpen(true);
      setDownloadStatus(null);
    }
  });

  async function download(artifact: ReportArtifactData) {
    setDownloadStatus(null);
    const response = await fetchArtifact(artifact.download_url);
    const blob = await response.blob();
    const actual = await sha256Hex(blob);
    if (actual !== artifact.sha256) {
      throw new Error(`SHA-256 校验失败：期望 ${artifact.sha256}，实际 ${actual}`);
    }
    const objectUrl = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = objectUrl;
    link.download = artifact.filename;
    link.click();
    setTimeout(() => URL.revokeObjectURL(objectUrl), 0);
    setDownloadStatus(`${artifact.filename} 已通过 SHA-256 校验并下载。`);
  }

  const downloadFile = useMutation({
    mutationFn: download
  });
  const pdfPreviewUrl = toBffArtifactUrl(report?.pdf.download_url, { inline: true });

  return (
    <section className="scientific-report" aria-labelledby="scientific-report-title">
      <div className="report-hero">
        <div className="report-hero-icon">
          <FileText size={22} />
        </div>
        <div>
          <span>SCIENTIFIC REPORT</span>
          <h3 id="scientific-report-title">把分割结果整理成可审阅的科研报告</h3>
          <p>
            {selectedImageCount > 1
              ? `已选择 ${selectedImageCount} 个图像视野；系统将汇总均值、标准差、四分位数、CV 与异常视野。`
              : "确定性数据工具先计算并核验指标，本地 Qwen 再组织技术摘要；"}
            页面预览、DOCX 与 PDF 共用同一份报告快照。
          </p>
        </div>
        <Button
          tone="primary"
          onClick={() => generate.mutate()}
          disabled={
            Boolean(writeBlocker) ||
            !selectedRuns.length ||
            generate.isPending
          }
          title={writeBlocker || undefined}
        >
          {report ? <RefreshCcw size={15} /> : <Sparkles size={15} />}
          {generate.isPending
            ? "正在取证并生成…"
            : report
              ? "重新生成"
              : selectedImageCount > 1
                ? "生成批量报告"
                : "生成系统报告"}
        </Button>
      </div>

      {!report ? (
        <div className="report-pipeline" aria-label="报告生成流程">
          <span><b>1</b> 数据工具取证</span>
          <span><b>2</b> 尺度与质量解释</span>
          <span><b>3</b> 本地模型综合</span>
          <span><b>4</b> DOCX / PDF</span>
        </div>
      ) : (
        <div className="report-preview">
          <header className="report-preview-header">
            <div>
              <span>REPORT SNAPSHOT</span>
              <h4>{report.title}</h4>
              <p>
                {formatDate(report.generated_at)} · {report.selected_run_ids.length} 个运行 ·{" "}
                {report.analysis_mode === "batch"
                  ? `${report.batch_summary?.image_count || selectedImageCount} 个视野 · `
                  : ""}
                {report.synthesis_provider === "local_llm"
                  ? `本地模型 ${report.synthesis_model || ""}`
                  : "可信模板降级"}
              </p>
            </div>
            <div className="report-preview-actions">
              <StatusBadge value={report.quality_status} />
              <StatusBadge
                value={report.scale_status === "physical" ? "healthy" : "degraded"}
                label={
                  report.scale_status === "physical"
                    ? "物理尺度"
                    : report.scale_status === "mixed"
                      ? "混合尺度"
                      : "仅像素尺度"
                }
              />
              <Button
                size="sm"
                tone="ghost"
                onClick={() => setPreviewOpen((open) => !open)}
              >
                <Eye size={14} />
                {previewOpen ? "收起 PDF" : "预览 PDF"}
              </Button>
              <Button
                size="sm"
                onClick={() => downloadFile.mutate(report.docx)}
                disabled={downloadFile.isPending}
              >
                <Download size={14} />
                DOCX
              </Button>
              <Button
                size="sm"
                onClick={() => downloadFile.mutate(report.pdf)}
                disabled={downloadFile.isPending}
              >
                <Download size={14} />
                PDF
              </Button>
            </div>
          </header>

          <section className="report-technical-summary">
            <div>
              <span>TECHNICAL SUMMARY</span>
              <StatusBadge
                value={report.fallback_used ? "degraded" : "healthy"}
                label={report.fallback_used ? "可信降级" : "本地模型综合"}
              />
            </div>
            <p>{report.technical_summary}</p>
          </section>

          <div className="report-metric-grid">
            {report.headline_metrics.map((metric) => (
              <article key={metric.key}>
                <span>{metric.label}</span>
                <strong>{metric.display_value}</strong>
                <p>{metric.definition}</p>
              </article>
            ))}
          </div>

          {report.batch_summary ? (
            <section className="report-section">
              <div className="section-subheading">
                <span>BATCH DISTRIBUTIONS</span>
                <h4>批量分布、离散度与异常视野</h4>
              </div>
              <div className="batch-quality-strip">
                <span>通过 <b>{report.batch_summary.quality_pass_count}</b></span>
                <span>警告 <b>{report.batch_summary.quality_warning_count}</b></span>
                <span>需复核 <b>{report.batch_summary.quality_review_count}</b></span>
                <span>异常组合 <b>{(report.batch_summary.outliers ?? []).length}</b></span>
              </div>
              <div className="report-run-table-wrap">
                <table className="report-run-table batch-distribution-table">
                  <thead>
                    <tr>
                      <th>指标</th>
                      <th>n</th>
                      <th>均值 ± 标准差</th>
                      <th>Q1 / 中位数 / Q3</th>
                      <th>范围</th>
                      <th>CV</th>
                    </tr>
                  </thead>
                  <tbody>
                    {report.batch_summary.distributions.map((item) => (
                      <tr key={item.key}>
                        <td><strong>{item.label}</strong><small>{item.unit}</small></td>
                        <td>{item.sample_count}</td>
                        <td>{formatMetric(item.mean)} ± {formatMetric(item.std_dev)}</td>
                        <td>
                          {formatMetric(item.q1)} / {formatMetric(item.median)} /{" "}
                          {formatMetric(item.q3)}
                        </td>
                        <td>{formatMetric(item.minimum)}–{formatMetric(item.maximum)}</td>
                        <td>
                          {item.coefficient_of_variation == null
                            ? "—"
                            : `${formatMetric(item.coefficient_of_variation * 100)}%`}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
              {(report.batch_summary.outliers ?? []).length ? (
                <div className="batch-outlier-list">
                  {(report.batch_summary.outliers ?? []).map((item) => (
                    <span key={`${item.run_id}-${item.metric_key}`}>
                      {compactId(item.image_id)} · {item.metric_label}{" "}
                      {item.direction === "high" ? "偏高" : "偏低"}：
                      {formatMetric(item.value)} {item.unit}
                    </span>
                  ))}
                </div>
              ) : null}
            </section>
          ) : null}

          <section className="report-section">
            <div className="section-subheading">
              <span>KEY FINDINGS</span>
              <h4>结果、解释与边界</h4>
            </div>
            <div className="report-findings">
              {report.findings.map((finding) => (
                <article className={finding.severity} key={finding.title}>
                  <strong>{finding.title}</strong>
                  <p>{finding.interpretation}</p>
                  <small>
                    {(finding.evidence_ids ?? []).join(" · ") || "结构化运行事实"} ·{" "}
                    {(finding.source_run_ids ?? []).map(compactId).join(" · ")}
                  </small>
                </article>
              ))}
            </div>
          </section>

          <section className="report-section">
            <div className="section-subheading">
              <span>RUN COMPARISON</span>
              <h4>所选运行的可比统计</h4>
            </div>
            <div className="report-run-table-wrap">
              <table className="report-run-table">
                <thead>
                  <tr>
                    <th>{report.analysis_mode === "batch" ? "视野 / 模型" : "模型"}</th>
                    <th>质量</th>
                    <th>颗粒</th>
                    <th>平均粒径</th>
                    <th>覆盖率</th>
                    <th>数密度</th>
                  </tr>
                </thead>
                <tbody>
                  {report.run_summaries.map((item) => (
                    <tr key={item.run_id}>
                      <td>
                        {report.analysis_mode === "batch" ? (
                          <strong>{item.filename || compactId(item.image_id)}</strong>
                        ) : null}
                        <strong>{item.model_id}</strong>
                        <small>{compactId(item.run_id)}</small>
                      </td>
                      <td><StatusBadge value={item.quality_status} /></td>
                      <td>{item.particle_count}</td>
                      <td>{item.mean_equivalent_diameter}</td>
                      <td>{item.coverage}</td>
                      <td>{item.number_density}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </section>

          <section className="report-section">
            <div className="section-subheading">
              <span>NEXT ACTIONS</span>
              <h4>按证据排序的复核动作</h4>
            </div>
            <ol className="report-actions">
              {report.recommendations.map((item) => (
                <li key={`${item.priority}-${item.action}`}>
                  <b>{item.priority}</b>
                  <div>
                    <strong>{item.action}</strong>
                    <p>{item.rationale}</p>
                    <small>验收：{item.verification}</small>
                  </div>
                </li>
              ))}
            </ol>
          </section>

          <details className="report-details">
            <summary>方法、限制与进一步问题</summary>
            <div>
              <ReportList title="方法" items={report.methodology} />
              <ReportList title="限制" items={report.limitations} />
              <ReportList title="进一步问题" items={report.further_questions} />
            </div>
          </details>

          {previewOpen && pdfPreviewUrl ? (
            <div className="report-pdf-preview">
              <iframe src={pdfPreviewUrl} title={`${report.title} PDF 预览`} />
            </div>
          ) : null}
        </div>
      )}

      {downloadStatus ? (
        <div className="verified-message">
          <CheckCircle2 size={16} />
          {downloadStatus}
        </div>
      ) : null}
      {generate.isError ? <RequestError error={generate.error} /> : null}
      {downloadFile.isError ? <RequestError error={downloadFile.error} /> : null}
    </section>
  );
}

function ReportList({ title, items }: { title: string; items: string[] }) {
  return (
    <section>
      <h5>{title}</h5>
      <ul>{items.map((item) => <li key={item}>{item}</li>)}</ul>
    </section>
  );
}

function formatMetric(value: number) {
  const magnitude = Math.abs(value);
  return formatNumber(value, magnitude > 0 && magnitude < 0.01 ? 6 : 3);
}
