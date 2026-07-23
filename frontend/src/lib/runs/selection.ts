type ImageScopedRun = {
  run_id: string;
  image_id: string;
};

export function selectRunForImage<T extends ImageScopedRun>(
  runs: T[],
  activeRunId: string | null,
  activeImageId: string | null
): T | null {
  if (!activeImageId) return null;
  const selected = runs.find((run) => run.run_id === activeRunId);
  if (selected?.image_id === activeImageId) return selected;
  return runs.find((run) => run.image_id === activeImageId) ?? null;
}

export function buildQueryScopeKey(
  jobId: string,
  imageId: string | null,
  runIds: string[]
): string {
  return JSON.stringify([jobId, imageId, runIds]);
}
