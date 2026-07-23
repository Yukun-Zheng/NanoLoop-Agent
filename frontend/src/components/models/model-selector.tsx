"use client";

import { useMutation, useQueryClient } from "@tanstack/react-query";
import {
  Check,
  Cpu,
  Gauge,
  Lightbulb,
  Play,
  ScanSearch,
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

const variantLabels: Record<string, string> = {
  general: "通用颗粒",
  small_particle: "小颗粒优化",
  large_particle: "大颗粒优化",
  dense_particle: "高密度/团聚区域",
  low_contrast: "低对比度优化"
};

export function isModelSelectable(
  model: ModelMetadata,
  roiMode: "full_image" | "boxes"
) {
  return (
    model.status === "ready" &&
    !model.health_error &&
    !(roiMode === "boxes" && !model.supports_box_prompt)
  );
}

export function ModelSelector({
  jobId,
  image,
  boxSet,
  catalog,
  writeBlocker,
  onRunsCreated
}: {
  jobId: string;
  image: ImageAsset | null;
  boxSet: BoxSet | null;
  catalog: ModelList;
  writeBlocker: string | null;
  onRunsCreated: (runIds: string[]) => void;
}) {
  const queryClient = useQueryClient();
  const [selected, setSelected] = useState<string[]>([]);
  const [roiMode, setRoiMode] = useState<"full_image" | "boxes">("full_image");
  const [prefer, setPrefer] = useState<"speed" | "balance" | "accuracy">("accuracy");
  const [threshold, setThreshold] = useState("");
  const [minArea, setMinArea] = useState("");
  const [device, setDevice] = useState<"auto" | "cpu" | "cuda" | "mps">("auto");
  const [watershed, setWatershed] = useState(false);
  const [excludeBorder, setExcludeBorder] = useState(true);
  const [confirmed, setConfirmed] = useState(false);

  const recommendation = useMutation({
    mutationFn: () => {
      if (!image) throw new Error("请先选择图像");
      const payload: ModelRecommendationRequest = {
        image_id: image.image_id,
        roi_mode: roiMode,
        target_profile: "general",
        prefer,
        device
      };
      return apiRequest<ModelRecommendation>("models/recommend", {
        method: "POST",
        body: payload
      });
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
      const invalidSelection = selected.find((modelId) => {
        const selectedModel = (catalog.models ?? []).find(
          (candidate) => candidate.model_id === modelId
        );
        return !selectedModel || !isModelSelectable(selectedModel, roiMode);
      });
      if (invalidSelection) {
        throw new Error(`模型 ${invalidSelection} 在当前 ROI 模式或健康状态下不可运行`);
      }
      if (roiMode === "boxes" && (!boxSet || !(boxSet.boxes ?? []).length)) {
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
          image_ids: [image.image_id],
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
    setConfirmed(false);
  }

  const catalogModels = catalog.models ?? [];
  const readyCount = catalogModels.filter(
    (model) => model.status === "ready" && !model.health_error
  ).length;

  return (
    <div className="model-selector">
      <div className="model-toolbar">
        <div className="segmented-control" aria-label="ROI 模式">
          <button
            className={roiMode === "full_image" ? "active" : undefined}
            onClick={() => {
              setRoiMode("full_image");
              setSelected((current) =>
                current.filter((modelId) => {
                  const currentModel = catalogModels.find(
                    (candidate) => candidate.model_id === modelId
                  );
                  return Boolean(
                    currentModel && isModelSelectable(currentModel, "full_image")
                  );
                })
              );
              setConfirmed(false);
            }}
          >
            全图
          </button>
          <button
            className={roiMode === "boxes" ? "active" : undefined}
            onClick={() => {
              setRoiMode("boxes");
              setSelected((current) =>
                current.filter((modelId) => {
                  const currentModel = catalogModels.find(
                    (candidate) => candidate.model_id === modelId
                  );
                  return Boolean(currentModel && isModelSelectable(currentModel, "boxes"));
                })
              );
              setConfirmed(false);
            }}
          >
            已保存 ROI
          </button>
        </div>
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
          {recommendation.isPending ? "正在推荐…" : "获取后端推荐"}
        </Button>
      </div>

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

      <div className="model-grid">
        {catalogModels.map((model) => {
          const selectable = isModelSelectable(model, roiMode);
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
                  {model.status !== "ready" || model.health_error
                    ? model.health_error || "模型尚未通过运行健康检查"
                    : "该模型不支持选框提示"}
                </p>
              ) : (
                <p className="model-note">
                  {model.notes || "模型声明不代表跨材料科学性能承诺。"}
                </p>
              )}
            </article>
          );
        })}
      </div>

      <section className="run-parameters">
        <div className="section-subheading">
          <span>IMMUTABLE RUN</span>
          <h3>确认推理参数</h3>
        </div>
        <div className="parameter-grid">
          <label className="field">
            <span>threshold（留空使用模型默认）</span>
            <input
              className="input"
              type="number"
              min="0"
              max="1"
              step="0.01"
              value={threshold}
              onChange={(event) => {
                setThreshold(event.target.value);
                setConfirmed(false);
              }}
            />
          </label>
          <label className="field">
            <span>min_area_px（留空使用模型默认）</span>
            <input
              className="input"
              type="number"
              min="0"
              value={minArea}
              onChange={(event) => {
                setMinArea(event.target.value);
                setConfirmed(false);
              }}
            />
          </label>
          <label className="field">
            <span>执行设备</span>
            <select
              className="select"
              value={device}
              onChange={(event) => {
                setDevice(event.target.value as typeof device);
                setConfirmed(false);
              }}
            >
              <option value="auto">自动</option>
              <option value="cpu">CPU</option>
              <option value="cuda">CUDA</option>
              <option value="mps">MPS</option>
            </select>
          </label>
          <label className="toggle-field">
            <input
              type="checkbox"
              checked={watershed}
              onChange={(event) => {
                setWatershed(event.target.checked);
                setConfirmed(false);
              }}
            />
            <span>启用 watershed</span>
          </label>
          <label className="toggle-field">
            <input
              type="checkbox"
              checked={excludeBorder}
              onChange={(event) => {
                setExcludeBorder(event.target.checked);
                setConfirmed(false);
              }}
            />
            <span>排除边界颗粒</span>
          </label>
        </div>
        <label className="run-confirmation">
          <input
            type="checkbox"
            checked={confirmed}
            onChange={(event) => setConfirmed(event.target.checked)}
          />
          <span>
            我确认使用所选模型和参数创建不可变运行。后续复核将生成子运行，不覆盖本次配置。
          </span>
        </label>
        <div className="run-submit">
          <span>已选 {selected.length}/3 个模型 · seed 42</span>
          <Button
            tone="primary"
            onClick={() => createRuns.mutate()}
            disabled={
              Boolean(writeBlocker) ||
              !confirmed ||
              !selected.length ||
              createRuns.isPending
            }
            title={writeBlocker || undefined}
          >
            <Play size={16} />
            {createRuns.isPending ? "正在提交…" : "创建运行"}
          </Button>
        </div>
        {createRuns.isError ? <RequestError error={createRuns.error} /> : null}
      </section>
    </div>
  );
}
