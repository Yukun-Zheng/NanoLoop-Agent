import { Database, FileImage, Microscope, Ruler, ShieldCheck } from "lucide-react";

import { StatusBadge } from "@/components/ui/status-badge";
import type { JobDetail } from "@/lib/api/types";
import { compactId, formatDate, formatNumber } from "@/lib/format/value";

export function ProjectOverview({ detail }: { detail: JobDetail }) {
  const images = detail.images ?? [];
  const runs = detail.runs ?? [];
  const failures = detail.partial_failures ?? [];
  return (
    <div className="project-overview">
      <section className="overview-hero">
        <div>
          <span>PROJECT SNAPSHOT</span>
          <h2>{detail.job.name}</h2>
          <p>
            创建于 {formatDate(detail.job.created_at)} · job_id {compactId(detail.job.job_id, 18)}
          </p>
        </div>
        <StatusBadge value={detail.job.status} />
      </section>

      <div className="overview-stats">
        <OverviewStat icon={FileImage} label="显微图像" value={String(images.length)} />
        <OverviewStat icon={Microscope} label="分析运行" value={String(runs.length)} />
        <OverviewStat
          icon={ShieldCheck}
          label="完成运行"
          value={String(
            runs.filter((run) =>
              ["COMPLETED", "COMPLETED_WITH_WARNINGS"].includes(run.status)
            ).length
          )}
        />
        <OverviewStat
          icon={Database}
          label="部分失败"
          value={String(failures.length)}
        />
      </div>

      <section className="image-table-section">
        <div className="section-subheading">
          <span>INPUT ASSETS</span>
          <h3>图像和材料元数据</h3>
        </div>
        <div className="image-table">
          {images.map((image) => (
            <article key={image.image_id}>
              <span className="image-table-icon"><FileImage size={18} /></span>
              <div className="image-table-name">
                <strong>{image.filename}</strong>
                <code>{compactId(image.image_id, 14)}</code>
              </div>
              <div>
                <span>样品</span>
                <strong>{image.sample_id}</strong>
              </div>
              <div>
                <span>材料</span>
                <strong>{image.material_formula || image.material_name || "未填写"}</strong>
              </div>
              <div>
                <span>尺寸</span>
                <strong>{image.width} × {image.height}</strong>
              </div>
              <div>
                <span>尺度</span>
                <strong>
                  {image.scale_nm_per_pixel
                    ? `${formatNumber(image.scale_nm_per_pixel)} nm/px${
                        image.scale_source === "sem_metadata" ? " · 仪器自动读取" : ""
                      }`
                    : "仅像素"}
                </strong>
              </div>
              <StatusBadge
                value={
                  (image.analysis_roi.invalid_rects ?? []).length ? "warn" : "pass"
                }
                label={
                  image.sem_metadata?.footer_detected
                    ? "已排除仪器栏"
                    : (image.analysis_roi.invalid_rects ?? []).length
                      ? "含无效区域"
                      : "全图分析"
                }
              />
            </article>
          ))}
        </div>
        <div className="sem-metadata-list">
          {images
            .filter((image) => image.sem_metadata)
            .map((image) => {
              const sem = image.sem_metadata!;
              return (
                <article className="sem-metadata-card" key={image.image_id}>
                  <div className="sem-metadata-heading">
                    <div>
                      <span>SEM METADATA</span>
                      <strong>{image.filename} · 仪器信息已自动识别</strong>
                    </div>
                    <StatusBadge value={sem.confidence === "high" ? "pass" : "warn"} />
                  </div>
                  <dl>
                    <MetadataItem
                      label="物理尺度"
                      value={
                        image.scale_nm_per_pixel
                          ? `${formatNumber(image.scale_nm_per_pixel, 6)} nm/px（${
                              image.scale_source === "sem_metadata" ? "仪器元数据" : "手动"
                            }）`
                          : "未识别"
                      }
                    />
                    <MetadataItem label="探测器" value={sem.detector} />
                    <MetadataItem
                      label="加速电压"
                      value={
                        sem.accelerating_voltage_kv
                          ? `${formatNumber(sem.accelerating_voltage_kv)} kV`
                          : null
                      }
                    />
                    <MetadataItem
                      label="工作距离"
                      value={
                        sem.working_distance_mm
                          ? `${formatNumber(sem.working_distance_mm)} mm`
                          : null
                      }
                    />
                    <MetadataItem
                      label="放大倍数"
                      value={
                        sem.magnification_x
                          ? `${formatNumber(sem.magnification_x, 0)}×`
                          : null
                      }
                    />
                    <MetadataItem
                      label="孔径"
                      value={
                        sem.aperture_size_um
                          ? `${formatNumber(sem.aperture_size_um)} µm`
                          : null
                      }
                    />
                    <MetadataItem
                      label="仪器"
                      value={
                        [sem.vendor, sem.instrument_model].filter(Boolean).join(" ") || null
                      }
                    />
                    <MetadataItem
                      label="采集时间"
                      value={sem.acquired_at ? formatDate(sem.acquired_at) : null}
                    />
                  </dl>
                  <p>
                    {sem.footer_detected && sem.footer_rect
                      ? `底部 ${sem.footer_rect.y1}–${sem.footer_rect.y2} px 已从推理和统计中排除。`
                      : "未检测到底部仪器栏，分析区域保持原图。"}
                  </p>
                </article>
              );
            })}
        </div>
      </section>

      <section className="project-boundary">
        <Ruler size={18} />
        <div>
          <strong>科学计算边界</strong>
          <p>
            系统优先读取原始显微图的可信仪器元数据并冻结物理尺度；检测到的底部信息栏不会进入
            模型推理或颗粒统计。没有信息栏或可信尺度时保持全图与像素单位，不做猜测。
          </p>
        </div>
      </section>
    </div>
  );
}

function MetadataItem({
  label,
  value
}: {
  label: string;
  value: string | null | undefined;
}) {
  return (
    <div>
      <dt>{label}</dt>
      <dd>{value || "—"}</dd>
    </div>
  );
}

function OverviewStat({
  icon: Icon,
  label,
  value
}: {
  icon: typeof FileImage;
  label: string;
  value: string;
}) {
  return (
    <div>
      <Icon size={16} />
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  );
}
