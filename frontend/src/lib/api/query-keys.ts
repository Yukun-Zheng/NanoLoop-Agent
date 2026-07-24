export const queryKeys = {
  health: ["health"] as const,
  analysis: (jobId: string) => ["analysis", jobId] as const,
  queryHistory: (jobId: string) => ["query-history", jobId] as const,
  conversations: (jobId: string) => ["conversations", jobId] as const,
  conversation: (jobId: string, conversationId: string) =>
    ["conversation", jobId, conversationId] as const,
  boxes: (jobId: string, imageId: string) => ["boxes", jobId, imageId] as const,
  models: ["models"] as const,
  run: (runId: string) => ["run", runId] as const,
  instanceArtifact: (runId: string) => ["instance-artifact", runId] as const,
  knowledge: ["knowledge"] as const
};
