import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { ScientificReport } from "@/components/results/scientific-report";
import { apiRequest, toBffArtifactUrl } from "@/lib/api/client";
import type { Run, ScientificReportData } from "@/lib/api/types";

vi.mock("@/lib/api/client", () => ({
  apiRequest: vi.fn(),
  fetchArtifact: vi.fn(),
  toBffArtifactUrl: vi.fn(() => "/api/nanoloop/files/report?inline=1")
}));

const mockedApi = vi.mocked(apiRequest);
const mockedArtifactUrl = vi.mocked(toBffArtifactUrl);
const now = "2026-07-24T08:00:00Z";
const artifact = {
  filename: "report.pdf",
  download_url: "/api/v1/files/report-token",
  sha256: "a".repeat(64),
  size_bytes: 1024,
  media_type: "application/pdf"
};
const report: ScientificReportData = {
  report_id: "report_1",
  job_id: "job_1",
  title: "BaNi-3 - SEM 纳米颗粒分析报告",
  generated_at: now,
  selected_run_ids: ["run_1"],
  analysis_mode: "single_image",
  quality_status: "WARN",
  scale_status: "physical",
  technical_summary: "当前运行识别 95 个颗粒，物理尺度可用 [D1]。",
  synthesis_provider: "local_llm",
  synthesis_model: "qwen2.5:7b",
  fallback_used: false,
  headline_metrics: [
    {
      key: "mean_diameter",
      label: "平均等效粒径",
      display_value: "59.37 nm",
      unit: "nm",
      definition: "与实例面积相同的圆的直径。",
      source_run_ids: ["run_1"]
    }
  ],
  findings: [
    {
      title: "质量门控要求针对性复核",
      interpretation: "小碎片比例偏高。",
      severity: "review",
      evidence_ids: ["D1"],
      source_run_ids: ["run_1"]
    }
  ],
  run_summaries: [
    {
      run_id: "run_1",
      image_id: "image_1",
      model_id: "unet-specialist",
      quality_status: "WARN",
      scale_nm_per_pixel: 1,
      particle_count: 95,
      roi_area: "2.88 µm²",
      number_density: "0.033 µm⁻²",
      mean_equivalent_diameter: "59.37 nm",
      coverage: "10.69%",
      perimeter_density: "9.16 µm⁻¹",
      quality_reasons: ["small_fragment_ratio_high"]
    }
  ],
  batch_summary: null,
  recommendations: [
    {
      priority: 1,
      action: "单变量上调 min_area_px 并创建复核运行。",
      rationale: "小碎片会抬高计数。",
      verification: "同时比较计数、粒径分布与覆盖率。",
      source_run_ids: ["run_1"]
    }
  ],
  methodology: ["确定性工具负责计算，本地模型只负责语言综合。"],
  limitations: ["单视野不能代表完整样品。"],
  further_questions: ["其他视野是否保持一致？"],
  data_evidence: [],
  knowledge_citations: [],
  provenance: [],
  docx: {
    ...artifact,
    filename: "report.docx",
    media_type:
      "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
  },
  pdf: artifact
};
const run = {
  run_id: "run_1",
  image_id: "image_1",
  status: "COMPLETED_WITH_WARNINGS"
} as Run;
const batchRun = {
  run_id: "run_2",
  image_id: "image_2",
  status: "COMPLETED"
} as Run;
const batchReport: ScientificReportData = {
  ...report,
  title: "BaNi 批次 - SEM 纳米颗粒批量分析报告",
  selected_run_ids: ["run_1", "run_2"],
  analysis_mode: "batch",
  batch_summary: {
    image_count: 2,
    run_count: 2,
    model_count: 1,
    total_particle_count: 215,
    quality_pass_count: 1,
    quality_warning_count: 1,
    quality_review_count: 0,
    distributions: [
      {
        key: "particle_count",
        label: "颗粒数量",
        unit: "个",
        sample_count: 2,
        mean: 107.5,
        std_dev: 12.5,
        minimum: 95,
        q1: 101.25,
        median: 107.5,
        q3: 113.75,
        maximum: 120,
        coefficient_of_variation: 0.1163
      }
    ],
    outliers: []
  },
  run_summaries: [
    report.run_summaries[0]!,
    {
      ...report.run_summaries[0]!,
      run_id: "run_2",
      image_id: "image_2",
      filename: "BaNi-4.tif",
      particle_count: 120
    }
  ]
};

function createQueryClient() {
  return new QueryClient({
    defaultOptions: {
      mutations: { retry: false },
      queries: { retry: false }
    }
  });
}

function renderReport(
  queryClient = createQueryClient(),
  runs: Run[] = [run]
) {
  return {
    queryClient,
    ...render(
      <QueryClientProvider client={queryClient}>
        <ScientificReport jobId="job_1" runs={runs} writeBlocker={null} />
      </QueryClientProvider>
    )
  };
}

beforeEach(() => {
  mockedApi.mockReset();
  mockedArtifactUrl.mockClear();
  mockedApi.mockResolvedValue({
    request_id: "req_report",
    status: "success",
    data: report,
    error: null
  });
});

describe("ScientificReport", () => {
  it("generates one source-backed snapshot and opens its PDF preview", async () => {
    const user = userEvent.setup();
    renderReport();

    await user.click(screen.getByRole("button", { name: "生成系统报告" }));

    await waitFor(() =>
      expect(mockedApi).toHaveBeenCalledWith("analyses/job_1/report", {
        method: "POST",
        body: { run_ids: ["run_1"] }
      })
    );
    expect(
      await screen.findByRole("heading", {
        name: "BaNi-3 - SEM 纳米颗粒分析报告"
      })
    ).toBeVisible();
    expect(screen.getByText("物理尺度")).toBeVisible();
    expect(screen.getAllByText("59.37 nm")).toHaveLength(2);
    expect(screen.getByText("单变量上调 min_area_px 并创建复核运行。")).toBeVisible();
    expect(screen.getByTitle("BaNi-3 - SEM 纳米颗粒分析报告 PDF 预览")).toHaveAttribute(
      "src",
      "/api/nanoloop/files/report?inline=1"
    );
    expect(mockedArtifactUrl).toHaveBeenCalledWith(report.pdf.download_url, {
      inline: true
    });
  });

  it("keeps the generated report when the results stage unmounts and mounts again", async () => {
    const user = userEvent.setup();
    const queryClient = createQueryClient();
    const firstRender = renderReport(queryClient);

    await user.click(screen.getByRole("button", { name: "生成系统报告" }));
    expect(
      await screen.findByRole("heading", {
        name: "BaNi-3 - SEM 纳米颗粒分析报告"
      })
    ).toBeVisible();

    firstRender.unmount();
    renderReport(queryClient);

    expect(
      screen.getByRole("heading", {
        name: "BaNi-3 - SEM 纳米颗粒分析报告"
      })
    ).toBeVisible();
    expect(screen.getByRole("button", { name: "重新生成" })).toBeVisible();
    expect(mockedApi).toHaveBeenCalledTimes(1);
  });

  it("renders high-dimensional distributions for a multi-image report", async () => {
    const user = userEvent.setup();
    mockedApi.mockResolvedValueOnce({
      request_id: "req_batch_report",
      status: "success",
      data: batchReport,
      error: null
    });
    renderReport(createQueryClient(), [run, batchRun]);

    await user.click(screen.getByRole("button", { name: "生成批量报告" }));

    await waitFor(() =>
      expect(mockedApi).toHaveBeenCalledWith("analyses/job_1/report", {
        method: "POST",
        body: { run_ids: ["run_1", "run_2"] }
      })
    );
    expect(
      await screen.findByRole("heading", {
        name: "批量分布、离散度与异常视野"
      })
    ).toBeVisible();
    expect(screen.getByText("107.5 ± 12.5")).toBeVisible();
    expect(screen.getByText("BaNi-4.tif")).toBeVisible();
  });
});
