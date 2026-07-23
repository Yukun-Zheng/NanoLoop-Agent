import { expect, test, type Page, type Route } from "@playwright/test";
import { createHash } from "node:crypto";

const now = "2026-07-23T08:00:00Z";
const jobId = "job-e2e";
const imageId = "image-e2e";
const runId = "run-e2e";
const reviewRunId = "run-review-e2e";
const sha = "a".repeat(64);
const exportToken = "signed-export-token";
const exportBytes = Buffer.from("nanoloop-e2e-trusted-export");
const exportSha = createHash("sha256").update(exportBytes).digest("hex");
const knowledgeDocId = "doc-e2e";

const knowledgeDocument = {
  doc_id: knowledgeDocId,
  title: "NanoLoop 材料笔记",
  source_type: "material_note",
  sha256: "b".repeat(64),
  citation_text: "NanoLoop team, Materials note, 2026",
  material_aliases: ["LaNiO3", "LNO"],
  status: "ready",
  license_note: "Team-authored and approved for internal use",
  allowed_for_demo: false,
  year: 2026,
  created_at: now
};

function envelope(data: unknown, status: "success" | "accepted" = "success") {
  return {
    request_id: `req-${Math.random().toString(16).slice(2)}`,
    status,
    data,
    error: null
  };
}

const job = {
  job_id: jobId,
  name: "LaNiO₃ 颗粒分析",
  status: "COMPLETED",
  created_at: now,
  updated_at: now,
  config: {}
};

const image = {
  image_id: imageId,
  job_id: jobId,
  filename: "sample.png",
  sha256: sha,
  width: 1024,
  height: 768,
  bit_depth: 8,
  sample_id: "sample",
  material_name: "Lanthanum nickelate",
  material_formula: "LaNiO3",
  experiment_conditions: { temperature: "800 °C" },
  scale_nm_per_pixel: 2,
  original_download_url: null,
  analysis_roi: {
    schema_version: 1,
    coordinate_space: "original_px",
    source: "none",
    revision: 1,
    valid_rect: { x1: 0, y1: 0, x2: 1024, y2: 768 },
    invalid_rects: []
  }
};

const model = {
  model_id: "unet-large-a",
  family: "unet",
  variant: "large_particle",
  quality_tier: "validated",
  version: "1.0.0",
  status: "ready",
  supports_box_prompt: true,
  preprocess_profile: "grayscale-v1",
  postprocess_profile: "large-particle-v1",
  default_threshold: 0.55,
  default_min_area_px: 16,
  expected_input_width: 512,
  expected_input_height: 512,
  notes: "E2E fixture",
  metrics: {},
  metric_context: {},
  applicable_materials: [],
  inference_invalid_bottom_px: 0,
  weight_sha256: sha,
  config_sha256: sha,
  adapter_sha256: null,
  model_card_sha256: sha
};

const unavailableModel = {
  ...model,
  model_id: "unet-large-b",
  version: "0.9.0",
  status: "unavailable",
  health_error: "模型权重校验失败：checkpoint 缺失",
  weight_sha256: null
};

const run = {
  run_id: runId,
  job_id: jobId,
  image_id: imageId,
  model_id: model.model_id,
  status: "COMPLETED",
  roi_mode: "full_image",
  threshold: 0.55,
  box_revision: null,
  parent_run_id: null,
  runtime_ms: 428,
  created_at: now,
  updated_at: now,
  inference: {
    threshold: 0.55,
    min_area_px: 16,
    watershed_enabled: false,
    exclude_border: true,
    device: "cpu",
    seed: 42
  },
  configuration: {
    model_id: model.model_id,
    model_version: model.version,
    roi_mode: "full_image",
    analysis_roi: image.analysis_roi,
    inference: {
      threshold: 0.55,
      min_area_px: 16,
      watershed_enabled: false,
      exclude_border: true,
      device: "cpu",
      seed: 42
    },
    preprocess_profile: model.preprocess_profile,
    postprocess_profile: model.postprocess_profile,
    created_at: now,
    image_sha256: sha,
    weight_sha256: sha,
    provenance_status: "complete",
    provenance_warnings: []
  },
  artifacts: {},
  quality: {
    status: "pass",
    reasons: [],
    recommendations: [],
    metrics: { foreground_ratio: 0.12 }
  },
  summary: {
    run_id: runId,
    particle_count: 42,
    roi_area_px: 786432,
    number_density_px2: 0.0000534,
    number_density_um2: 13.35,
    coverage_ratio: 0.12,
    perimeter_density_px: 0.006,
    perimeter_density_um: 3,
    mean_equivalent_diameter_px: 18.5,
    mean_equivalent_diameter_nm: 37,
    quality_status: "pass"
  },
  status_history: []
};

const reviewRun = {
  ...run,
  run_id: reviewRunId,
  status: "COMPLETED",
  threshold: 0.68,
  parent_run_id: runId,
  runtime_ms: 451,
  inference: {
    ...run.inference,
    threshold: 0.68,
    min_area_px: 20,
    watershed_enabled: true
  },
  configuration: {
    ...run.configuration,
    inference: {
      ...run.configuration.inference,
      threshold: 0.68,
      min_area_px: 20,
      watershed_enabled: true
    },
    parent_run_id: runId,
    review_source: "parameter_override"
  },
  summary: {
    ...run.summary,
    run_id: reviewRunId,
    particle_count: 40
  }
};

async function installApiMock(page: Page) {
  let hasRun = false;
  let hasReviewRun = false;
  let savedBoxes: Array<Record<string, unknown>> = [];
  let boxRevision = 0;
  let knowledgeDocuments: Array<typeof knowledgeDocument> = [];
  const browserApiRequests: string[] = [];
  const captured = {
    boxSave: null as Record<string, unknown> | null,
    boxGets: 0,
    createRun: null as Record<string, unknown> | null,
    review: null as Record<string, unknown> | null,
    query: null as Record<string, unknown> | null,
    knowledgeIngestBody: null as string | null,
    knowledgeToggles: [] as Array<Record<string, unknown>>,
    knowledgeReindex: null as Record<string, unknown> | null,
    exportDownloads: 0
  };

  page.on("request", (request) => {
    if (request.resourceType() === "fetch" || request.resourceType() === "xhr") {
      browserApiRequests.push(new URL(request.url()).pathname);
    }
  });

  await page.route("**/api/nanoloop/**", async (route: Route) => {
    const request = route.request();
    const path = new URL(request.url()).pathname.replace("/api/nanoloop/", "");
    const method = request.method();

    if (path === "health" && method === "GET") {
      return route.fulfill({
        contentType: "application/json",
        body: JSON.stringify(
          envelope({
            service: { status: "healthy", detail: "ready" },
            database: { status: "healthy", detail: "ready" },
            model_registry: { status: "healthy", detail: "1 ready" },
            rag_index: { status: "healthy", detail: "ready" },
            version: "1.0.0"
          })
        )
      });
    }
    if (path === "analyses" && method === "POST") {
      return route.fulfill({
        contentType: "application/json",
        body: JSON.stringify(envelope({ job, images: [image], runs: [], partial_failures: [] }))
      });
    }
    if (path === `analyses/${jobId}` && method === "GET") {
      return route.fulfill({
        contentType: "application/json",
        body: JSON.stringify(
          envelope({
            job,
            images: [image],
            runs: hasRun ? [run, ...(hasReviewRun ? [reviewRun] : [])] : [],
            partial_failures: []
          })
        )
      });
    }
    if (path === `analyses/${jobId}/images/${imageId}/boxes` && method === "GET") {
      captured.boxGets += 1;
      return route.fulfill({
        contentType: "application/json",
        body: JSON.stringify(
          envelope({
            image_id: imageId,
            revision: boxRevision,
            boxes: savedBoxes
          })
        )
      });
    }
    if (path === `analyses/${jobId}/images/${imageId}/boxes` && method === "PUT") {
      const payload = request.postDataJSON() as {
        expected_revision: number;
        boxes: Array<Record<string, unknown>>;
      };
      captured.boxSave = payload;
      if (payload.expected_revision !== boxRevision) {
        return route.fulfill({
          status: 409,
          contentType: "application/json",
          body: JSON.stringify({
            request_id: "req-roi-conflict",
            status: "error",
            data: null,
            error: {
              code: "REVISION_CONFLICT",
              message: "ROI revision 冲突",
              details: { current_revision: boxRevision },
              retryable: false
            }
          })
        });
      }
      boxRevision += 1;
      savedBoxes = payload.boxes.map((box, index) => ({
        ...box,
        box_id: `box-${index + 1}`
      }));
      return route.fulfill({
        contentType: "application/json",
        body: JSON.stringify(
          envelope({
            image_id: imageId,
            revision: boxRevision,
            boxes: savedBoxes
          })
        )
      });
    }
    if (path === "models" && method === "GET") {
      return route.fulfill({
        contentType: "application/json",
        body: JSON.stringify(envelope({ models: [model, unavailableModel] }))
      });
    }
    if (path === `analyses/${jobId}/runs` && method === "POST") {
      captured.createRun = request.postDataJSON() as Record<string, unknown>;
      hasRun = true;
      return route.fulfill({
        contentType: "application/json",
        body: JSON.stringify(envelope({ run_ids: [runId] }, "accepted"))
      });
    }
    if (path === `runs/${runId}/review` && method === "POST") {
      captured.review = request.postDataJSON() as Record<string, unknown>;
      hasReviewRun = true;
      return route.fulfill({
        contentType: "application/json",
        body: JSON.stringify(
          envelope({ run_id: reviewRunId, parent_run_id: runId }, "accepted")
        )
      });
    }
    if (path === `analyses/${jobId}/query` && method === "POST") {
      captured.query = request.postDataJSON() as Record<string, unknown>;
      return route.fulfill({
        contentType: "application/json",
        body: JSON.stringify(
          envelope({
            query_type: "mixed",
            answer: "该运行识别到 42 个颗粒；当前知识证据仅支持保守解释。",
            confidence: "medium",
            outcome_code: "OK",
            needs_clarification: false,
            data_evidence: [
              {
                tool_name: "get_run_summary",
                validated_arguments: { run_id: runId },
                source_run_ids: [runId],
                aggregates: { particle_count: 42 },
                rows: [],
                units: { particle_count: "count" },
                quality_warnings: []
              }
            ],
            citations: [],
            limitations: ["仅有单张图像"],
            tool_calls: []
          })
        )
      });
    }
    if (path === `analyses/${jobId}/export` && method === "GET") {
      return route.fulfill({
        contentType: "application/json",
        body: JSON.stringify(
          envelope({
            download_url: `/api/v1/files/${exportToken}`,
            filename: "nanoloop-e2e-export.zip",
            sha256: exportSha,
            size_bytes: exportBytes.byteLength
          })
        )
      });
    }
    if (path === `files/${exportToken}` && method === "GET") {
      captured.exportDownloads += 1;
      return route.fulfill({
        status: 200,
        headers: {
          "content-type": "application/zip",
          "content-disposition": 'attachment; filename="nanoloop-e2e-export.zip"',
          "cache-control": "private, no-store"
        },
        body: exportBytes
      });
    }
    if (path === "knowledge/documents" && method === "GET") {
      return route.fulfill({
        contentType: "application/json",
        body: JSON.stringify(envelope({ documents: knowledgeDocuments }))
      });
    }
    if (path === "knowledge/documents" && method === "POST") {
      captured.knowledgeIngestBody = request.postData();
      knowledgeDocuments = [knowledgeDocument];
      return route.fulfill({
        contentType: "application/json",
        body: JSON.stringify(
          envelope({
            document: knowledgeDocument,
            pages_extracted: 1,
            chunks_indexed: 2,
            warnings: []
          })
        )
      });
    }
    if (path === `knowledge/documents/${knowledgeDocId}` && method === "PATCH") {
      const payload = request.postDataJSON() as { enabled: boolean };
      captured.knowledgeToggles.push(payload);
      knowledgeDocuments = knowledgeDocuments.map((document) => ({
        ...document,
        status: payload.enabled ? "ready" : "disabled"
      }));
      return route.fulfill({
        contentType: "application/json",
        body: JSON.stringify(envelope(knowledgeDocuments[0]))
      });
    }
    if (path === "knowledge/reindex" && method === "POST") {
      captured.knowledgeReindex = request.postDataJSON() as Record<string, unknown>;
      return route.fulfill({
        contentType: "application/json",
        body: JSON.stringify(
          envelope({
            documents_indexed: knowledgeDocuments.filter(
              (document) => document.status !== "disabled"
            ).length,
            chunks_indexed: 2,
            chunks_skipped: 0,
            index_version: "knowledge-e2e-v1",
            warnings: []
          })
        )
      });
    }

    return route.fulfill({
      status: 404,
      contentType: "application/json",
      body: JSON.stringify({
        request_id: "req-unhandled",
        status: "error",
        data: null,
        error: {
          code: "E2E_UNHANDLED",
          message: `${method} ${path}`,
          details: {},
          retryable: false
        }
      })
    });
  });

  return {
    browserApiRequests,
    captured,
    controls: {
      bumpBoxRevision() {
        boxRevision += 1;
      },
      currentBoxRevision() {
        return boxRevision;
      }
    }
  };
}

test("completes the mocked scientific workflow through verified export", async ({ page }) => {
  const { browserApiRequests, captured } = await installApiMock(page);
  await page.goto("/");

  await page.getByLabel("分析任务名称").fill("LaNiO₃ 颗粒分析");
  await page.locator('input[type="file"]').setInputFiles({
    name: "sample.png",
    mimeType: "image/png",
    buffer: Buffer.from("e2e-image")
  });
  await page.getByRole("button", { name: "创建任务" }).click();

  await expect(page).toHaveURL(`/workspace/${jobId}`);
  await expect(page.getByRole("heading", { name: "LaNiO₃ 颗粒分析" })).toBeVisible();

  await page.getByRole("button", { name: "ROI", exact: true }).click();
  await page.getByRole("button", { name: "添加" }).click();
  const roiBox = page.locator(".roi-box").first();
  await roiBox.getByText("x1", { exact: true }).locator("..").getByRole("spinbutton").fill("96");
  await roiBox.getByText("y1", { exact: true }).locator("..").getByRole("spinbutton").fill("80");
  await roiBox.getByText("x2", { exact: true }).locator("..").getByRole("spinbutton").fill("420");
  await roiBox.getByText("y2", { exact: true }).locator("..").getByRole("spinbutton").fill("360");
  await page.getByRole("button", { name: "保存全部 ROI" }).click();
  await expect(page.getByText("将从 revision 1 更新")).toBeVisible();
  expect(captured.boxSave).toMatchObject({
    expected_revision: 0,
    boxes: [
      {
        x1: 96,
        y1: 80,
        x2: 420,
        y2: 360,
        active: true
      }
    ]
  });

  await page.reload();
  await page.getByRole("button", { name: "ROI", exact: true }).click();
  await expect(page.getByText("将从 revision 1 更新")).toBeVisible();
  await expect(page.locator(".roi-box").first().getByRole("spinbutton").nth(0)).toHaveValue("96");
  await expect(page.locator(".roi-box").first().getByRole("spinbutton").nth(3)).toHaveValue("360");

  await page.getByRole("button", { name: "模型与运行" }).click();
  await expect(page.getByRole("heading", { name: model.model_id })).toBeVisible();
  await expect(page.getByRole("heading", { name: unavailableModel.model_id })).toBeVisible();
  await expect(page.getByText("模型权重校验失败：checkpoint 缺失")).toBeVisible();
  await expect(
    page.getByRole("button", { name: `选择 ${unavailableModel.model_id}` })
  ).toBeDisabled();
  await page.getByRole("button", { name: "已保存 ROI" }).click();
  await page.getByRole("button", { name: `选择 ${model.model_id}` }).click();
  await page.getByText("我确认使用所选模型和参数创建不可变运行").click();
  await page.getByRole("button", { name: "创建运行" }).click();

  await expect(page.getByText("智能体执行时间线")).toBeVisible();
  expect(captured.createRun).toMatchObject({
    image_ids: [imageId],
    model_ids: [model.model_id],
    roi_mode: "boxes",
    box_revisions: { [imageId]: 1 }
  });

  await page.getByRole("button", { name: "结果", exact: true }).click();
  await expect(page.getByRole("heading", { name: "质量门控" })).toBeVisible();
  await expect(page.getByText("后端未报告额外风险原因。")).toBeVisible();
  await expect(
    page.locator(".metric").filter({ hasText: "颗粒数量" }).getByText("42", { exact: true })
  ).toBeVisible();
  await page.getByLabel("threshold", { exact: true }).fill("0.68");
  await page.getByLabel("min_area_px", { exact: true }).fill("20");
  await page.locator(".review-panel select").first().selectOption("true");
  await page.getByRole("button", { name: "创建复核子运行" }).click();

  await expect(page.getByText("智能体执行时间线")).toBeVisible();
  await expect(page.getByText(reviewRunId, { exact: true })).toBeVisible();
  expect(captured.review).toEqual({
    threshold: 0.68,
    min_area_px: 20,
    watershed_enabled: true
  });

  await page.getByRole("button", { name: "混合" }).click();
  await page.getByLabel("询问当前实验").fill("这张图识别了多少颗粒？");
  await page.getByRole("button", { name: "发送问题" }).click();

  await expect(page.getByText("该运行识别到 42 个颗粒")).toBeVisible();
  await expect(page.getByText("实验数据证据")).toBeVisible();
  expect(captured.query).toMatchObject({
    question: "这张图识别了多少颗粒？",
    query_type: "mixed",
    image_id: imageId,
    run_ids: [runId]
  });

  await page.getByRole("button", { name: "结果", exact: true }).click();
  const downloadPromise = page.waitForEvent("download");
  await page.getByRole("button", { name: "导出当前及所选运行" }).click();
  const download = await downloadPromise;
  expect(download.suggestedFilename()).toBe("nanoloop-e2e-export.zip");
  await expect(page.getByText("SHA-256 已验证，可信报告已下载。")).toBeVisible();
  expect(captured.exportDownloads).toBe(1);

  if (process.env.NANOLOOP_E2E_SCREENSHOT) {
    await page.screenshot({
      path: process.env.NANOLOOP_E2E_SCREENSHOT,
      fullPage: true
    });
  }
  const browserApiPaths = browserApiRequests.filter((path) => path.startsWith("/api/"));
  expect(browserApiPaths.some((path) => path.startsWith("/api/nanoloop/"))).toBe(true);
  expect(
    browserApiPaths.every(
      (path) => path === "/api/healthz" || path.startsWith("/api/nanoloop/")
    )
  ).toBe(true);
});

test("opens the scientific inspector as a responsive drawer", async ({ page }) => {
  await page.setViewportSize({ width: 1200, height: 800 });
  await installApiMock(page);
  await page.goto(`/workspace/${jobId}`);

  await page.getByRole("button", { name: "打开科学审查器" }).click();
  await expect(page.getByRole("dialog", { name: "科学证据审查器" })).toBeVisible();
  await page.getByRole("button", { name: "关闭科学审查器" }).click();
  await expect(page.getByRole("dialog", { name: "科学证据审查器" })).toBeHidden();
});

test("preserves an unsaved ROI when the server reports a revision conflict", async ({
  page
}) => {
  await page.addInitScript(() => {
    Object.defineProperty(navigator, "clipboard", {
      configurable: true,
      value: {
        writeText(value: string) {
          (
            window as unknown as {
              __copiedRoi?: string;
            }
          ).__copiedRoi = value;
          return Promise.resolve();
        }
      }
    });
  });
  const { captured, controls } = await installApiMock(page);
  await page.goto(`/workspace/${jobId}`);

  await page.getByRole("button", { name: "ROI", exact: true }).click();
  await page.getByRole("button", { name: "添加" }).click();
  const roiBox = page.locator(".roi-box").first();
  await roiBox.getByText("x1", { exact: true }).locator("..").getByRole("spinbutton").fill("96");
  await roiBox.getByText("y1", { exact: true }).locator("..").getByRole("spinbutton").fill("80");
  await roiBox.getByText("x2", { exact: true }).locator("..").getByRole("spinbutton").fill("420");
  await roiBox.getByText("y2", { exact: true }).locator("..").getByRole("spinbutton").fill("360");

  controls.bumpBoxRevision();
  expect(controls.currentBoxRevision()).toBe(1);
  await page.getByRole("button", { name: "保存全部 ROI" }).click();

  await expect(page.getByText("ROI 已被其他会话更新")).toBeVisible();
  await expect(page.getByRole("button", { name: "重新加载服务端" })).toBeVisible();
  await expect(page.getByRole("button", { name: "复制当前坐标" })).toBeVisible();
  expect(captured.boxSave).toMatchObject({
    expected_revision: 0,
    boxes: [{ x1: 96, y1: 80, x2: 420, y2: 360 }]
  });
  await expect(roiBox.getByRole("spinbutton").nth(0)).toHaveValue("96");
  await expect(roiBox.getByRole("spinbutton").nth(3)).toHaveValue("360");

  await page.getByRole("button", { name: "复制当前坐标" }).click();
  await expect
    .poll(() =>
      page.evaluate(
        () =>
          (
            window as unknown as {
              __copiedRoi?: string;
            }
          ).__copiedRoi
      )
    )
    .toContain('"x1": 96');
  const copiedRoi = await page.evaluate(
    () =>
      (
        window as unknown as {
          __copiedRoi?: string;
        }
      ).__copiedRoi
  );
  expect(JSON.parse(copiedRoi || "[]")).toMatchObject([
    { x1: 96, y1: 80, x2: 420, y2: 360, active: true }
  ]);

  const getsBeforeReload = captured.boxGets;
  await page.getByRole("button", { name: "重新加载服务端" }).click();
  await expect(page.getByText("将从 revision 1 更新 · 0/20 个框")).toBeVisible();
  await expect(page.getByText("ROI 已被其他会话更新")).toBeHidden();
  await expect(page.locator(".roi-box")).toHaveCount(0);
  expect(captured.boxGets).toBeGreaterThan(getsBeforeReload);
});

test("ingests, lists, disables, enables, and reindexes a knowledge document", async ({
  page
}) => {
  const { captured } = await installApiMock(page);
  await page.goto("/knowledge");

  await expect(page.getByRole("heading", { name: "材料知识库" })).toBeVisible();
  await expect(page.getByText("知识库尚无文档")).toBeVisible();
  await page.locator('input[type="file"]').setInputFiles({
    name: "nanoloop-note.md",
    mimeType: "text/markdown",
    buffer: Buffer.from("# NanoLoop materials note")
  });
  await page.getByLabel("标题 *").fill(knowledgeDocument.title);
  await page.getByLabel("来源类型").selectOption("material_note");
  await page.getByLabel("年份").fill("2026");
  await page.getByLabel("规范引用 *").fill(knowledgeDocument.citation_text);
  await page.getByLabel("材料别名（逗号分隔）").fill("LaNiO3, LNO");
  await page.getByLabel("许可与来源说明 *").fill(knowledgeDocument.license_note);
  await page.getByRole("button", { name: "导入并建立引用" }).click();

  await expect(page.getByText(knowledgeDocument.title, { exact: true })).toBeVisible();
  await expect(page.getByText("1 份受管文档")).toBeVisible();
  expect(captured.knowledgeIngestBody).toContain('filename="nanoloop-note.md"');
  expect(captured.knowledgeIngestBody).toContain('"title":"NanoLoop 材料笔记"');
  expect(captured.knowledgeIngestBody).toContain('"allowed_for_demo":false');

  await page.getByRole("button", { name: `停用 ${knowledgeDocument.title}` }).click();
  await expect(
    page.getByRole("button", { name: `启用 ${knowledgeDocument.title}` })
  ).toBeVisible();
  await expect(page.getByText("已停用", { exact: true })).toBeVisible();

  await page.getByRole("button", { name: `启用 ${knowledgeDocument.title}` }).click();
  await expect(
    page.getByRole("button", { name: `停用 ${knowledgeDocument.title}` })
  ).toBeVisible();
  expect(captured.knowledgeToggles).toEqual([{ enabled: false }, { enabled: true }]);

  await page.getByRole("button", { name: "强制重建索引" }).click();
  await expect(page.getByText(/knowledge-e2e-v1/)).toBeVisible();
  expect(captured.knowledgeReindex).toEqual({ force: true });
});
