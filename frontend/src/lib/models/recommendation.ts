import { apiRequest } from "@/lib/api/client";
import type {
  ImageAsset,
  ModelList,
  ModelRecommendation,
  ModelRecommendationRequest
} from "@/lib/api/types";

export type ImageModelAssignment = {
  imageId: string;
  filename: string;
  modelId: string;
  score: number | null;
  reasons: string[];
  usedFallback: boolean;
};

export async function recommendModelsForImages({
  images,
  models,
  roiMode,
  prefer,
  device
}: {
  images: ImageAsset[];
  models: ModelList["models"];
  roiMode: ModelRecommendationRequest["roi_mode"];
  prefer: ModelRecommendationRequest["prefer"];
  device: ModelRecommendationRequest["device"];
}): Promise<ImageModelAssignment[]> {
  const readyModels = (models ?? []).filter(
    (model) => model.status === "ready" && !model.health_error
  );
  const readyIds = new Set(readyModels.map((model) => model.model_id));
  const fallback = readyModels[0]?.model_id;
  if (!fallback) throw new Error("当前没有可运行模型");

  return Promise.all(
    images.map(async (image) => {
      try {
        const response = await apiRequest<ModelRecommendation>("models/recommend", {
          method: "POST",
          body: {
            job_id: image.job_id,
            image_id: image.image_id,
            roi_mode: roiMode,
            target_profile: "general",
            auto_profile: true,
            prefer,
            device
          }
        });
        const candidate = (response.data.candidates ?? []).find((item) =>
          readyIds.has(item.model_id)
        );
        if (candidate) {
          return {
            imageId: image.image_id,
            filename: image.filename,
            modelId: candidate.model_id,
            score: candidate.score,
            reasons: candidate.reasons ?? [],
            usedFallback: false
          };
        }
      } catch {
        // A catalog-verified model remains available when recommendation is degraded.
      }
      return {
        imageId: image.image_id,
        filename: image.filename,
        modelId: fallback,
        score: null,
        reasons: ["推荐服务不可用，使用首个已验证模型"],
        usedFallback: true
      };
    })
  );
}

export function runAssignmentPayload(assignments: ImageModelAssignment[]) {
  return {
    model_ids: [...new Set(assignments.map((item) => item.modelId))],
    model_assignments: Object.fromEntries(
      assignments.map((item) => [item.imageId, item.modelId])
    )
  };
}
