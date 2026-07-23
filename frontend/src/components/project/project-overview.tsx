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
                    ? `${formatNumber(image.scale_nm_per_pixel)} nm/px`
                    : "仅像素"}
                </strong>
              </div>
              <StatusBadge
                value={
                  (image.analysis_roi.invalid_rects ?? []).length ? "warn" : "pass"
                }
                label={(image.analysis_roi.invalid_rects ?? []).length ? "含无效区域" : "已校验"}
              />
            </article>
          ))}
        </div>
      </section>

      <section className="project-boundary">
        <Ruler size={18} />
        <div>
          <strong>科学计算边界</strong>
          <p>
            前端只展示后端 summary、quality 与 provenance。物理尺度缺失时不会猜测 nm 或 µm，
            也不会从 CSV 重新计算权威统计。
          </p>
        </div>
      </section>
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
