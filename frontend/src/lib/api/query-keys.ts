export const queryKeys = {
  health: ["health"] as const,
  analysis: (jobId: string) => ["analysis", jobId] as const,
  queryHistory: (jobId: string) => ["query-history", jobId] as const,
  boxes: (jobId: string, imageId: string) => ["boxes", jobId, imageId] as const,
  models: ["models"] as const,
  run: (runId: string) => ["run", runId] as const,
  knowledge: ["knowledge"] as const
};
