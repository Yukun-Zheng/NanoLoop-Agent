export function defaultAnalysisName(filenames: string[]): string {
  const cleaned = filenames
    .map((filename) => filename.replace(/\.[^.]+$/, "").trim())
    .filter(Boolean);
  if (cleaned.length === 1) return `图像分割 · ${cleaned[0]}`.slice(0, 255);
  if (cleaned.length > 1) return `批量图像分割 · ${cleaned.length} 张图像`;
  return "图像分割任务";
}

export function runParameterError(threshold: string, minArea: string): string | null {
  if (threshold !== "") {
    const value = Number(threshold);
    if (!Number.isFinite(value) || value < 0 || value > 1) {
      return "threshold 必须是 0 到 1 之间的数字";
    }
  }
  if (minArea !== "") {
    const value = Number(minArea);
    if (!Number.isInteger(value) || value < 0) {
      return "min_area_px 必须是大于或等于 0 的整数";
    }
  }
  return null;
}
