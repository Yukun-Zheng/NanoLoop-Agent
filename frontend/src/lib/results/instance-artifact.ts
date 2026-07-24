import { fetchArtifact } from "@/lib/api/client";

export type InstanceLabel = {
  instanceIndex: number;
  xPercent: number;
  yPercent: number;
  confidence: number | null;
};

export type InstanceArtifact = {
  width: number;
  height: number;
  labels: InstanceLabel[];
};

export async function loadInstanceArtifact(url: string): Promise<InstanceArtifact> {
  const response = await fetchArtifact(url);
  return parseInstanceArtifact(await response.json());
}

export function parseInstanceArtifact(value: unknown): InstanceArtifact {
  if (!isRecord(value)) throw new Error("实例制品格式无效");
  const width = finitePositive(value.width);
  const height = finitePositive(value.height);
  if (!width || !height || !Array.isArray(value.instances)) {
    throw new Error("实例制品缺少图像尺寸或实例列表");
  }

  const labels = value.instances.slice(0, 5000).map((item) => {
    if (!isRecord(item)) throw new Error("实例记录格式无效");
    const instanceIndex = finiteInteger(item.instance_index);
    const bbox = item.bbox_xyxy;
    if (
      instanceIndex === null ||
      !Array.isArray(bbox) ||
      bbox.length !== 4 ||
      !bbox.every((coordinate) => typeof coordinate === "number" && Number.isFinite(coordinate))
    ) {
      throw new Error("实例记录缺少有效编号或边界框");
    }
    const [x1, y1, x2, y2] = bbox as [number, number, number, number];
    const confidence =
      item.confidence === null || item.confidence === undefined
        ? null
        : finiteProbability(item.confidence);
    if (confidence === undefined) throw new Error("实例置信度格式无效");

    return {
      instanceIndex,
      xPercent: clampPercent(((x1 + x2) / 2 / width) * 100),
      yPercent: clampPercent(((y1 + y2) / 2 / height) * 100),
      confidence
    };
  });

  return { width, height, labels };
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null;
}

function finitePositive(value: unknown): number | null {
  return typeof value === "number" && Number.isFinite(value) && value > 0 ? value : null;
}

function finiteInteger(value: unknown): number | null {
  return typeof value === "number" && Number.isSafeInteger(value) && value > 0
    ? value
    : null;
}

function finiteProbability(value: unknown): number | undefined {
  return typeof value === "number" && Number.isFinite(value) && value >= 0 && value <= 1
    ? value
    : undefined;
}

function clampPercent(value: number) {
  return Math.min(99, Math.max(1, value));
}
