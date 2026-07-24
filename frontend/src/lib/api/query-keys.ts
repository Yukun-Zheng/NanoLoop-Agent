export const queryKeys = {
  health: ["health"] as const,
  analysis: (jobId: string) => ["analysis", jobId] as const,
  boxes: (jobId: string, imageId: string) => ["boxes", jobId, imageId] as const,
  models: ["models"] as const,
  run: (runId: string) => ["run", runId] as const,
  instanceArtifact: (runId: string) => ["instance-artifact", runId] as const,
  knowledge: ["knowledge"] as const
};
