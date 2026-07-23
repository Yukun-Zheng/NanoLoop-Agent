"use client";

import { useMutation, useQueryClient } from "@tanstack/react-query";
import {
  CheckCircle2,
  Download,
  FileCheck2,
  GitCompareArrows,
  RotateCcw,
  ShieldAlert,
  Upload
} from "lucide-react";
import { useMemo, useRef, useState } from "react";

import { Button } from "@/components/ui/button";
import { EmptyState } from "@/components/ui/empty-state";
import { RequestError } from "@/components/ui/request-error";
import { StatusBadge } from "@/components/ui/status-badge";
import { apiRequest, fetchArtifact, toBffArtifactUrl } from "@/lib/api/client";
import { sha256Hex } from "@/lib/crypto/sha256";
import { queryKeys } from "@/lib/api/query-keys";
import type {
  CorrectedMaskUpload,
  ExportData,
  ImageAsset,
  ReviewRunData,
  Run
} from "@/lib/api/types";
import { formatNumber } from "@/lib/format/value";

import { ArtifactPreview } from "./artifact-preview";

type LayerKey = "original" | "mask" | "overlay" | "probability" | "labeled";

const terminal = new Set(["COMPLETED", "COMPLETED_WITH_WARNINGS"]);

export function ResultView({
  jobId,
  image,
  run,
  comparisonRuns,
  writeBlocker,
  onChildCreated
}: {
  jobId: string;
  image: ImageAsset | null;
  run: Run | null;
  comparisonRuns: Run[];
  writeBlocker: string | null;
  onChildCreated: (runId: string) => void;
}) {
  const queryClient = useQueryClient();
  const maskInput = useRef<HTMLInputElement>(null);
  const [layer, setLayer] = useState<LayerKey>("overlay");
  const [threshold, setThreshold] = useState("");
  const [minArea, setMinArea] = useState("");
  const [watershed, setWatershed] = useState<boolean | null>(null);
  const [excludeBorder, setExcludeBorder] = useState<boolean | null>(null);
  const [correctedMask, setCorrectedMask] = useState<CorrectedMaskUpload | null>(null);
  const [exportStatus, setExportStatus] = useState<string | null>(null);

  const layers = useMemo(
    () =>
      run
        ? {
            original: image?.original_download_url,
            mask: run.artifacts?.mask_url,
            overlay: run.artifacts?.overlay_url,
            probability: run.artifacts?.probability_url,
            labeled: run.artifacts?.labeled_particles_url
          }
        : {},
    [image, run]
  );

  const uploadMask = useMutation({
    mutationFn: async (file: File) => {
      if (!run) throw new Error("请先选择运行");
      const body = new FormData();
      body.append("file", file, file.name);
      return apiRequest<CorrectedMaskUpload>(
        `runs/${encodeURIComponent(run.run_id)}/corrected-mask`,
        { method: "POST", body }
      );
    },
    onSuccess(response) {
      setCorrectedMask(response.data);
    }
  });

  const review = useMutation({
    mutationFn: async () => {
      if (!run) throw new Error("请先选择运行");
      const payload: Record<string, unknown> = {};
      if (threshold !== "") payload.threshold = Number(threshold);
      if (minArea !== "") payload.min_area_px = Number(minArea);
      if (watershed !== null) payload.watershed_enabled = watershed;
      if (excludeBorder !== null) payload.exclude_border = excludeBorder;
      if (correctedMask) payload.corrected_mask_token = correctedMask.corrected_mask_token;
      if (!Object.keys(payload).length) throw new Error("至少修改一个复核参数或上传修正掩码");
      return apiRequest<ReviewRunData>(`runs/${encodeURIComponent(run.run_id)}/review`, {
        method: "POST",
        body: payload
      });
    },
    async onSuccess(response) {
      await queryClient.invalidateQueries({ queryKey: queryKeys.analysis(jobId) });
      onChildCreated(response.data.run_id);
    }
  });

  const exportRun = useMutation({
    mutationFn: async () => {
      if (!run || !terminal.has(run.status)) throw new Error("只能导出已完成运行");
      const exportRuns = [run, ...comparisonRuns].filter(
        (candidate, index, candidates) =>
          terminal.has(candidate.status) &&
          candidates.findIndex((item) => item.run_id === candidate.run_id) === index
      );
      const query = new URLSearchParams();
      for (const candidate of exportRuns) query.append("run_ids", candidate.run_id);
      const response = await apiRequest<ExportData>(
        `analyses/${encodeURIComponent(jobId)}/export?${query.toString()}`
      );
      const download = await fetchArtifact(response.data.download_url);
      const blob = await download.blob();
      const actual = await sha256Hex(blob);
      if (actual !== response.data.sha256) {
        throw new Error(`SHA-256 校验失败：期望 ${response.data.sha256}，实际 ${actual}`);
      }
      const objectUrl = URL.createObjectURL(blob);
      const link = document.createElement("a");
      link.href = objectUrl;
      link.download = response.data.filename;
      link.click();
      setTimeout(() => URL.revokeObjectURL(objectUrl), 0);
      return response.data;
    },
    onSuccess() {
      setExportStatus("SHA-256 已验证，可信报告已下载。");
    },
    onError() {
      setExportStatus(null);
    }
  });

  if (!run) {
    return (
      <EmptyState
        icon={GitCompareArrows}
        title="选择一个运行查看结果"
        detail="运行结果、质量门控和权威统计均来自后端，不会在浏览器中补算。"
      />
    );
  }

  const summary = run.summary;
  const quality = run.quality;
  const currentLayer = layers[layer];

  return (
    <div className="result-view">
      <div className="result-toolbar">
        <div className="layer-tabs" role="tablist" aria-label="结果图层">
          {(
            [
              ["original", "原图"],
              ["mask", "Mask"],
              ["overlay", "Overlay"],
              ["probability", "Probability"],
              ["labeled", "实例标注"]
            ] as const
          ).map(([key, label]) => (
            <button
              role="tab"
              aria-selected={layer === key}
              className={layer === key ? "active" : undefined}
              key={key}
              onClick={() => setLayer(key)}
            >
              {label}
              {!layers[key] ? <i aria-label="未生成" /> : null}
            </button>
          ))}
        </div>
        <div className="result-download-actions">
          {run.artifacts?.instances_url &&
          toBffArtifactUrl(run.artifacts.instances_url) ? (
            <Button asChild size="sm" tone="ghost">
              <a
                href={toBffArtifactUrl(run.artifacts.instances_url) || "#"}
                download={`${run.run_id}-instances`}
              >
                <Download size={14} />
                实例数据
              </a>
            </Button>
          ) : null}
          {run.artifacts?.particles_csv_url &&
          toBffArtifactUrl(run.artifacts.particles_csv_url) ? (
            <Button asChild size="sm" tone="ghost">
              <a
                href={toBffArtifactUrl(run.artifacts.particles_csv_url) || "#"}
                download={`${run.run_id}-particles.csv`}
              >
                <Download size={14} />
                颗粒 CSV
              </a>
            </Button>
          ) : null}
          <Button
            size="sm"
            onClick={() => exportRun.mutate()}
            disabled={!terminal.has(run.status) || exportRun.isPending}
          >
            <Download size={14} />
            导出当前{comparisonRuns.length ? "及所选" : ""}运行
          </Button>
        </div>
      </div>

      <div className="result-canvas">
        <ArtifactPreview
          url={currentLayer}
          alt={`${layer} 图层`}
          filename={`${run.run_id}-${layer}`}
        />
      </div>

      {exportStatus ? (
        <div className="verified-message">
          <FileCheck2 size={16} />
          {exportStatus}
        </div>
      ) : null}
      {exportRun.isError ? <RequestError error={exportRun.error} /> : null}

      <section className="quality-first">
        <div className="quality-heading">
          <div>
            <span>QUALITY GATE</span>
            <h3>质量门控</h3>
          </div>
          <StatusBadge value={quality?.status || summary?.quality_status || "unavailable"} />
        </div>
        {quality ? (
          <div className="quality-content">
            <div>
              <strong>判断依据</strong>
              {(quality.reasons ?? []).length ? (
                <ul>{(quality.reasons ?? []).map((reason) => <li key={reason}>{reason}</li>)}</ul>
              ) : (
                <p>后端未报告额外风险原因。</p>
              )}
            </div>
            <div>
              <strong>复核建议</strong>
              {(quality.recommendations ?? []).length ? (
                <ul>
                  {(quality.recommendations ?? []).map((item) => <li key={item}>{item}</li>)}
                </ul>
              ) : (
                <p>当前没有额外建议。</p>
              )}
            </div>
          </div>
        ) : (
          <p className="muted-copy">本次运行尚未生成质量门控报告。</p>
        )}
      </section>

      <section className="metrics-section">
        <div className="section-subheading">
          <span>CANONICAL SUMMARY</span>
          <h3>权威统计</h3>
        </div>
        {summary ? (
          <div className="metric-grid">
            <Metric label="颗粒数量" value={formatNumber(summary.particle_count, 0)} />
            <Metric label="ROI 面积" value={`${formatNumber(summary.roi_area_px, 0)} px²`} />
            <Metric
              label="数量密度"
              value={
                summary.number_density_um2 === null
                  ? `${formatNumber(summary.number_density_px2, 6)} px⁻²`
                  : `${formatNumber(summary.number_density_um2)} µm⁻²`
              }
            />
            <Metric
              label="平均等效粒径"
              value={
                summary.mean_equivalent_diameter_nm === null
                  ? `${formatNumber(summary.mean_equivalent_diameter_px)} px`
                  : `${formatNumber(summary.mean_equivalent_diameter_nm)} nm`
              }
            />
            <Metric label="覆盖率" value={`${formatNumber(summary.coverage_ratio * 100, 2)}%`} />
            <Metric
              label="周长密度"
              value={
                summary.perimeter_density_um === null
                  ? `${formatNumber(summary.perimeter_density_px, 6)} px⁻¹`
                  : `${formatNumber(summary.perimeter_density_um)} µm⁻¹`
              }
            />
          </div>
        ) : (
          <p className="muted-copy">运行尚无确定性汇总指标；界面不会自行推测数值。</p>
        )}
      </section>

      {comparisonRuns.length >= 2 ? (
        <section className="comparison-section">
          <div className="section-subheading">
            <span>MODEL COMPARISON</span>
            <h3>同图模型比较</h3>
          </div>
          <div className="comparison-table">
            {comparisonRuns.slice(0, 3).map((candidate) => (
              <article key={candidate.run_id}>
                <strong>{candidate.model_id}</strong>
                <StatusBadge value={candidate.quality?.status || candidate.status} />
                <div className="comparison-artifact">
                  <ArtifactPreview
                    url={artifactForLayer(candidate, image, layer)}
                    alt={`${candidate.model_id} ${layer} 图层`}
                    filename={`${candidate.run_id}-${layer}`}
                  />
                </div>
                <span>颗粒 {formatNumber(candidate.summary?.particle_count, 0)}</span>
                <span>
                  覆盖率{" "}
                  {candidate.summary?.coverage_ratio == null
                    ? "—"
                    : `${formatNumber(candidate.summary.coverage_ratio * 100, 2)}%`}
                </span>
                <span>耗时 {formatNumber(candidate.runtime_ms, 0)} ms</span>
              </article>
            ))}
          </div>
          <p className="muted-copy">界面只并列后端事实，不自行判定“最佳模型”。</p>
        </section>
      ) : null}

      <section className="review-panel">
        <div className="review-heading">
          <div>
            <span>HUMAN IN THE LOOP</span>
            <h3>创建不可变复核子运行</h3>
          </div>
          <RotateCcw size={20} />
        </div>
        <div className="review-grid">
          <label className="field">
            <span>threshold</span>
            <input
              className="input"
              type="number"
              min="0"
              max="1"
              step="0.01"
              value={threshold}
              onChange={(event) => setThreshold(event.target.value)}
              placeholder="保持原值"
            />
          </label>
          <label className="field">
            <span>min_area_px</span>
            <input
              className="input"
              type="number"
              min="0"
              value={minArea}
              onChange={(event) => setMinArea(event.target.value)}
              placeholder="保持原值"
            />
          </label>
          <label className="field">
            <span>watershed</span>
            <select
              className="select"
              value={watershed === null ? "" : String(watershed)}
              onChange={(event) =>
                setWatershed(event.target.value === "" ? null : event.target.value === "true")
              }
            >
              <option value="">保持原值</option>
              <option value="true">启用</option>
              <option value="false">关闭</option>
            </select>
          </label>
          <label className="field">
            <span>exclude border</span>
            <select
              className="select"
              value={excludeBorder === null ? "" : String(excludeBorder)}
              onChange={(event) =>
                setExcludeBorder(
                  event.target.value === "" ? null : event.target.value === "true"
                )
              }
            >
              <option value="">保持原值</option>
              <option value="true">排除边界</option>
              <option value="false">保留边界</option>
            </select>
          </label>
        </div>
        {writeBlocker ? (
          <p className="form-warning" role="status">
            {writeBlocker}
          </p>
        ) : null}
        <div className="review-actions">
          <input
            ref={maskInput}
            className="sr-only"
            type="file"
            accept=".png,.tif,.tiff,.bmp,.npy"
            onChange={(event) => {
              const file = event.target.files?.[0];
              if (file) uploadMask.mutate(file);
            }}
          />
          <Button
            onClick={() => maskInput.current?.click()}
            disabled={Boolean(writeBlocker) || uploadMask.isPending}
            title={writeBlocker || undefined}
          >
            <Upload size={15} />
            {uploadMask.isPending ? "上传中…" : "上传修正掩码"}
          </Button>
          {correctedMask ? (
            <span className="mask-proof">
              <CheckCircle2 size={14} />
              {correctedMask.width}×{correctedMask.height} · {correctedMask.sha256.slice(0, 10)}…
            </span>
          ) : null}
          <Button
            tone="primary"
            onClick={() => review.mutate()}
            disabled={Boolean(writeBlocker) || review.isPending}
            title={writeBlocker || undefined}
          >
            <ShieldAlert size={15} />
            {review.isPending ? "正在创建…" : "创建复核子运行"}
          </Button>
        </div>
        {uploadMask.isError ? <RequestError error={uploadMask.error} /> : null}
        {review.isError ? <RequestError error={review.error} /> : null}
      </section>
    </div>
  );
}

function Metric({ label, value }: { label: string; value: string }) {
  return (
    <div className="metric">
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  );
}

function artifactForLayer(
  run: Run,
  image: ImageAsset | null,
  layer: LayerKey
): string | null | undefined {
  if (layer === "original") return image?.original_download_url;
  if (layer === "mask") return run.artifacts?.mask_url;
  if (layer === "overlay") return run.artifacts?.overlay_url;
  if (layer === "probability") return run.artifacts?.probability_url;
  return run.artifacts?.labeled_particles_url;
}
