"use client";

import { useMutation, useQueryClient } from "@tanstack/react-query";
import {
  Check,
  Cpu,
  Gauge,
  Lightbulb,
  Play,
  ScanSearch,
  Settings2,
  ShieldCheck
} from "lucide-react";
import { useMemo, useState } from "react";

import { Button } from "@/components/ui/button";
import { EmptyState } from "@/components/ui/empty-state";
import { RequestError } from "@/components/ui/request-error";
import { StatusBadge } from "@/components/ui/status-badge";
import { apiRequest } from "@/lib/api/client";
import { queryKeys } from "@/lib/api/query-keys";
import type {
  BoxSet,
  CreateRunsData,
  ImageAsset,
  ModelList,
  ModelMetadata,
  ModelRecommendation,
  ModelRecommendationRequest
} from "@/lib/api/types";
import { formatNumber } from "@/lib/format/value";
import { runParameterError } from "@/lib/runs/configuration";

const variantLabels: Record<string, string> = {
  general: "通用颗粒",
  small_particle: "小颗粒优化",
  large_particle: "大颗粒优化",
  dense_particle: "高密度/团聚区域",
  low_contrast: "低对比度优化"
};

export function isModelSelectable(model: ModelMetadata) {
  return model.status === "ready" && !model.health_error;
}

export function ModelSelector({
  jobId,
  images,
  image,
  boxSet,
  catalog,
  writeBlocker,
  onRunsCreated
}: {
  jobId: string;
  images?: ImageAsset[];
  image: ImageAsset | null;
  boxSet: BoxSet | null;
  catalog: ModelList;
  writeBlocker: string | null;
  onRunsCreated: (runIds: string[]) => void;
}) {
  const queryClient = useQueryClient();
  const catalogModels = catalog.models ?? [];
  const availableImages = images?.length ? images : image ? [image] : [];
  const batchAvailable = availableImages.length > 1;
  const activeRoiCount = (boxSet?.boxes ?? []).filter((box) => box.active).length;
  const initialRoiMode: "full_image" | "boxes" =
    !batchAvailable && activeRoiCount > 0 ? "boxes" : "full_image";
  const [selected, setSelected] = useState<string[]>(() => {
    const firstReady = catalogModels.find((model) => isModelSelectable(model));
    return firstReady ? [firstReady.model_id] : [];
  });
  const [imageScope, setImageScope] = useState<"current" | "all">(
    batchAvailable ? "all" : "current"
  );
  const [roiMode, setRoiMode] = useState<"full_image" | "boxes">(initialRoiMode);
  const [prefer, setPrefer] = useState<"speed" | "balance" | "accuracy">("accuracy");
  const [threshold, setThreshold] = useState("");
  const [minArea, setMinArea] = useState("");
  const [device, setDevice] = useState<"auto" | "cpu" | "cuda" | "mps">("auto");
  const [watershed, setWatershed] = useState(false);
  const [excludeBorder, setExcludeBorder] = useState(true);
  const selectedImages =
    imageScope === "all" && batchAvailable
      ? availableImages
      : image
        ? [image]
        : [];
  const batchMode = selectedImages.length > 1;

  const recommendation = useMutation({
    mutationFn: () => {
      const recommendationImage = image || selectedImages[0];
      if (!recommendationImage) throw new Error("请先选择图像");
      const payload: ModelRecommendationRequest = {
        image_id: recommendationImage.image_id,
        roi_mode: roiMode,
        target_profile: "general",
        prefer,
        device
      };
      return apiRequest<ModelRecommendation>("models/recommend", {
        method: "POST",
        body: payload
      });
    },
    onSuccess(response) {
      const recommended = (response.data.candidates ?? []).find((candidate) => {
        const model = catalogModels.find((item) => item.model_id === candidate.model_id);
        return Boolean(model && isModelSelectable(model));
      });
      if (recommended) setSelected([recommended.model_id]);
    }
  });

  const recommendedIds = useMemo(
    () =>
      new Set(
        (recommendation.data?.data.candidates ?? []).map((item) => item.model_id)
      ),
    [recommendation.data]
  );

  const createRuns = useMutation({
    mutationFn: () => {
      if (!image) throw new Error("请先选择图像");
      if (!selected.length) throw new Error("至少选择一个就绪模型");
      const invalidParameters = runParameterError(threshold, minArea);
      if (invalidParameters) throw new Error(invalidParameters);
      const invalidSelection = selected.find((modelId) => {
        const selectedModel = (catalog.models ?? []).find(
          (candidate) => candidate.model_id === modelId
        );
        return !selectedModel || !isModelSelectable(selectedModel);
      });
      if (invalidSelection) {
        throw new Error(`模型 ${invalidSelection} 当前未通过运行健康检查`);
      }
      if (roiMode === "boxes" && activeRoiCount === 0) {
        throw new Error("选框模式需要先保存至少一个 ROI");
      }
      const inference: Record<string, unknown> = {
        watershed_enabled: watershed,
        exclude_border: excludeBorder,
        device,
        seed: 42
      };
      if (threshold !== "") inference.threshold = Number(threshold);
      if (minArea !== "") inference.min_area_px = Number(minArea);
      return apiRequest<CreateRunsData>(`analyses/${encodeURIComponent(jobId)}/runs`, {
        method: "POST",
        body: {
          image_ids: selectedImages.map((item) => item.image_id),
          model_ids: selected,
          roi_mode: roiMode,
          ...(roiMode === "boxes" && boxSet
            ? { box_revisions: { [image.image_id]: boxSet.revision } }
            : {}),
          inference
        }
      });
    },
    async onSuccess(response) {
      await queryClient.invalidateQueries({ queryKey: queryKeys.analysis(jobId) });
      onRunsCreated(response.data.run_ids);
    }
  });

  function toggleModel(modelId: string) {
    setSelected((current) =>
      current.includes(modelId)
        ? current.filter((item) => item !== modelId)
        : current.length < 3
          ? [...current, modelId]
          : current
    );
  }

  function changeRoiMode(nextMode: "full_image" | "boxes") {
    setRoiMode(nextMode);
    setSelected((current) => {
      const compatible = current.filter((modelId) => {
        const model = catalogModels.find((candidate) => candidate.model_id === modelId);
        return Boolean(model && isModelSelectable(model));
      });
      if (compatible.length) return compatible;
      const fallback = catalogModels.find((model) => isModelSelectable(model));
      return fallback ? [fallback.model_id] : [];
    });
  }

  function changeImageScope(nextScope: "current" | "all") {
    setImageScope(nextScope);
    if (nextScope === "all") {
      setRoiMode("full_image");
    }
  }

  const readyCount = catalogModels.filter(
    (model) => model.status === "ready" && !model.health_error
  ).length;
  const parameterError = runParameterError(threshold, minArea);
  const roiError =
    roiMode === "boxes" && activeRoiCount === 0
      ? "选框模式需要先到 ROI 页面保存至少一个区域"
      : null;
  const configurationError =
    writeBlocker ||
    (!selectedImages.length ? "请先选择图像" : null) ||
    (!selected.length ? "请选择至少一个就绪模型" : null) ||
    roiError ||
    parameterError;
  const selectedLabel =
    selected.length === 1
      ? selected[0]
      : selected.length > 1
        ? `${selected.length} 个模型并行对比`
        : "尚未选择模型";
  const runCount = selectedImages.length * selected.length;

  return (
    <div className="model-selector">
      <section className="guided-run-card">
        <div>
          <span>本次分析设置</span>
          <h2>确认范围和模型后开始</h2>
          <p>
            分析范围与模型都直接显示在这里；只有阈值和执行设备收在下方可选设置中。
          </p>
        </div>
        <dl>
          <div>
            <dt>图像范围</dt>
            <dd>
              {batchMode
                ? `${selectedImages.length} 张批处理`
                : selectedImages[0]?.filename || "尚未选择"}
            </dd>
          </div>
          <div>
            <dt>分析范围</dt>
            <dd>{roiMode === "full_image" ? "整张图像" : "已保存 ROI"}</dd>
          </div>
          <div>
            <dt>模型</dt>
            <dd>{selectedLabel}</dd>
          </div>
          <div>
            <dt>参数</dt>
            <dd>{parameterError ? "需要修正" : "默认值 / 自动设备"}</dd>
          </div>
        </dl>
        <div className="analysis-scope-block">
          <div className="analysis-step-heading">
            <b>1</b>
            <div>
              <strong>选择图像范围</strong>
              <span>
                {batchAvailable
                  ? `任务中有 ${availableImages.length} 张图，可自动批量创建并行运行。`
                  : "单图任务会直接沿用当前图像，不增加额外操作。"}
              </span>
            </div>
          </div>
          <div className="analysis-scope-options" role="radiogroup" aria-label="图像范围">
            <button
              type="button"
              role="radio"
              aria-checked={imageScope === "current"}
              className={imageScope === "current" ? "active" : undefined}
              onClick={() => changeImageScope("current")}
            >
              <span className="scope-choice-check">
                {imageScope === "current" ? <Check size={14} /> : null}
              </span>
              <strong>仅当前图像</strong>
              <small>{image?.filename || "未选择图像"}</small>
            </button>
            <button
              type="button"
              role="radio"
              aria-checked={imageScope === "all"}
              className={imageScope === "all" ? "active" : undefined}
              disabled={!batchAvailable}
              onClick={() => changeImageScope("all")}
            >
              <span className="scope-choice-check">
                {imageScope === "all" ? <Check size={14} /> : null}
              </span>
              <strong>全部图像批处理</strong>
              <small>
                {batchAvailable
                  ? `${availableImages.length} 张图使用同一冻结模型与参数`
                  : "当前任务只有一张图像"}
              </small>
            </button>
          </div>
        </div>
        <div className="analysis-scope-block">
          <div className="analysis-step-heading">
            <b>2</b>
            <div>
              <strong>选择分析范围</strong>
              <span>
                {activeRoiCount
                  ? `检测到上一步已保存 ${activeRoiCount} 个 ROI，已默认只分析这些区域。`
                  : "没有已保存 ROI 时使用整张图像。"}
              </span>
            </div>
          </div>
          <div className="analysis-scope-options" role="radiogroup" aria-label="分析范围">
            <button
              type="button"
              role="radio"
              aria-checked={roiMode === "full_image"}
              className={roiMode === "full_image" ? "active" : undefined}
              onClick={() => changeRoiMode("full_image")}
            >
              <span className="scope-choice-check">
                {roiMode === "full_image" ? <Check size={14} /> : null}
              </span>
              <strong>整张图像</strong>
              <small>分割并统计整幅图中的颗粒</small>
            </button>
            <button
              type="button"
              role="radio"
              aria-checked={roiMode === "boxes"}
              className={roiMode === "boxes" ? "active" : undefined}
              disabled={batchMode || !activeRoiCount}
              onClick={() => changeRoiMode("boxes")}
            >
              <span className="scope-choice-check">
                {roiMode === "boxes" ? <Check size={14} /> : null}
              </span>
              <strong>只分析已保存 ROI</strong>
              <small>
                {batchMode
                  ? "批处理使用全图；如需 ROI，请切换为仅当前图像"
                  : activeRoiCount
                  ? `${activeRoiCount} 个区域；框外内容不会进入结果统计`
                  : "先到上一步保存至少一个有效区域"}
              </small>
            </button>
          </div>
        </div>
        <div className="guided-run-action">
          {configurationError ? (
            <p className="inline-configuration-error" role="status">
              {configurationError}
            </p>
          ) : (
            <p>点击开始即创建可追溯运行；以后调参会生成新运行，不会覆盖本次结果。</p>
          )}
          <Button
            tone="primary"
            onClick={() => createRuns.mutate()}
            disabled={Boolean(configurationError) || createRuns.isPending}
            title={configurationError || undefined}
          >
            <Play size={16} />
            {createRuns.isPending
              ? "正在提交…"
              : runCount > 1
                ? `批量创建 ${runCount} 个运行`
                : "开始分割"}
          </Button>
        </div>
      </section>

      {recommendation.isError ? <RequestError error={recommendation.error} /> : null}
      {writeBlocker ? (
        <p className="form-warning" role="status">
          {writeBlocker}
        </p>
      ) : null}

      {!readyCount ? (
        <EmptyState
          icon={ScanSearch}
          title="当前没有可运行的真实模型"
          detail="模型目录仍会展示不可用原因；项目创建、ROI 和证据审查不受影响。"
        />
      ) : null}

      <section className="analysis-model-settings" aria-labelledby="model-selection-title">
        <header className="analysis-model-settings-heading">
          <b>3</b>
          <div>
            <strong id="model-selection-title">选择分析模型</strong>
            <span>点击模型卡片即可切换；需要比较差异时可同时选择，最多 3 个。</span>
          </div>
          <small>已选 {selected.length}/3</small>
        </header>
        <div className="advanced-settings-body">
          <div className="model-toolbar">
            <label className="compact-select">
              <span>推荐偏好</span>
              <select
                value={prefer}
                onChange={(event) => setPrefer(event.target.value as typeof prefer)}
              >
                <option value="accuracy">精度</option>
                <option value="balance">平衡</option>
                <option value="speed">速度</option>
              </select>
            </label>
            <Button
              onClick={() => recommendation.mutate()}
              disabled={Boolean(writeBlocker) || !image || recommendation.isPending}
              title={writeBlocker || undefined}
            >
              <Lightbulb size={15} />
              {recommendation.isPending ? "正在推荐…" : "重新推荐并应用"}
            </Button>
          </div>

          <p className="advanced-help">
            一般只运行推荐模型。只有需要比较模型差异时才多选，最多同时运行 3 个。
          </p>
          <div className="model-grid">
            {catalogModels.map((model) => {
              const selectable = isModelSelectable(model);
              const active = selected.includes(model.model_id);
              const candidate = (recommendation.data?.data.candidates ?? []).find(
                (item) => item.model_id === model.model_id
              );
              return (
                <article
                  className={`model-card${active ? " selected" : ""}${!selectable ? " unavailable" : ""}`}
                  key={model.model_id}
                >
                  <button
                    type="button"
                    className="model-select-target"
                    disabled={!selectable}
                    onClick={() => toggleModel(model.model_id)}
                    aria-pressed={active}
                    aria-label={`${active ? "取消选择" : "选择"} ${model.model_id}`}
                  >
                    <span className="model-check">{active ? <Check size={14} /> : null}</span>
                  </button>
                  <div className="model-card-heading">
                    <div>
                      <span>{model.family.replace("_", "-").toUpperCase()}</span>
                      <h3>{model.model_id}</h3>
                    </div>
                    <StatusBadge value={model.status} />
                  </div>
                  <div className="model-tags">
                    <span>{variantLabels[model.variant] || model.variant}</span>
                    <span>{model.quality_tier}</span>
                    <span>v{model.version}</span>
                  </div>
                  <dl className="model-facts">
                    <div>
                      <dt><Gauge size={13} />默认阈值</dt>
                      <dd>{formatNumber(model.default_threshold)}</dd>
                    </div>
                    <div>
                      <dt><ShieldCheck size={13} />最小面积</dt>
                      <dd>{formatNumber(model.default_min_area_px, 0)} px</dd>
                    </div>
                    <div>
                      <dt><Cpu size={13} />输入尺寸</dt>
                      <dd>{model.expected_input_width || "—"}×{model.expected_input_height || "—"}</dd>
                    </div>
                  </dl>
                  {recommendedIds.has(model.model_id) ? (
                    <div className="recommendation-note">
                      推荐分 {formatNumber(candidate?.score, 3)}
                      {(candidate?.reasons ?? []).length
                        ? ` · ${(candidate?.reasons ?? []).join("；")}`
                        : ""}
                    </div>
                  ) : null}
                  {!selectable ? (
                    <p className="model-blocker">
                      {model.health_error || "模型尚未通过运行健康检查"}
                    </p>
                  ) : (
                    <p className="model-note">
                      {roiMode === "boxes" && !model.supports_box_prompt
                        ? "支持 ROI 分析：模型完成推理后，系统只保留并统计选区内结果。"
                        : model.notes || "模型声明不代表跨材料科学性能承诺。"}
                    </p>
                  )}
                </article>
              );
            })}
          </div>

          <details className="optional-run-settings">
            <summary>
              <Settings2 size={17} />
              <div>
                <strong>阈值与运行参数（可选）</strong>
                <span>通常无需修改；留空会使用模型默认值，设备会自动选择。</span>
              </div>
              <small>
                {threshold || minArea || device !== "auto" ? "已有自定义" : "使用默认"}
              </small>
            </summary>
            <section className="run-parameters">
              <div className="section-subheading">
                <span>OPTIONAL OVERRIDES</span>
                <h3>参数覆盖（留空即使用模型默认值）</h3>
              </div>
              <div className="parameter-grid">
                <label className="field">
                  <span>置信度阈值</span>
                  <input
                    className="input"
                    type="number"
                    min="0"
                    max="1"
                    step="0.01"
                    value={threshold}
                    onChange={(event) => setThreshold(event.target.value)}
                    placeholder="模型默认值"
                  />
                </label>
                <label className="field">
                  <span>最小颗粒面积（px）</span>
                  <input
                    className="input"
                    type="number"
                    min="0"
                    step="1"
                    value={minArea}
                    onChange={(event) => setMinArea(event.target.value)}
                    placeholder="模型默认值"
                  />
                </label>
                <label className="field">
                  <span>执行设备</span>
                  <select
                    className="select"
                    value={device}
                    onChange={(event) => setDevice(event.target.value as typeof device)}
                  >
                    <option value="auto">自动（推荐）</option>
                    <option value="cpu">CPU</option>
                    <option value="cuda">CUDA</option>
                    <option value="mps">MPS</option>
                  </select>
                </label>
                <label className="toggle-field">
                  <input
                    type="checkbox"
                    checked={watershed}
                    onChange={(event) => setWatershed(event.target.checked)}
                  />
                  <span>启用 watershed</span>
                </label>
                <label className="toggle-field">
                  <input
                    type="checkbox"
                    checked={excludeBorder}
                    onChange={(event) => setExcludeBorder(event.target.checked)}
                  />
                  <span>排除边界颗粒</span>
                </label>
              </div>
              {parameterError || roiError ? (
                <p className="form-warning" role="status">
                  {parameterError || roiError}
                </p>
              ) : null}
              <div className="run-submit">
                <span>点击开始即确认保存本次不可变配置 · seed 42</span>
                <Button
                  tone="primary"
                  onClick={() => createRuns.mutate()}
                  disabled={Boolean(configurationError) || createRuns.isPending}
                  title={configurationError || undefined}
                >
                  <Play size={16} />
                  {createRuns.isPending ? "正在提交…" : "使用以上设置开始"}
                </Button>
              </div>
            </section>
          </details>
        </div>
      </section>
      {createRuns.isError ? <RequestError error={createRuns.error} /> : null}
    </div>
  );
}
