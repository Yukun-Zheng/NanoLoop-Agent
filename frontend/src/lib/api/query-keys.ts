export const queryKeys = {
  health: ["health"] as const,
  analysis: (jobId: string) => ["analysis", jobId] as const,
  queryHistory: (jobId: string) => ["query-history", jobId] as const,
  conversations: (jobId: string) => ["conversations", jobId] as const,
  conversation: (jobId: string, conversationId: string) =>
    ["conversation", jobId, conversationId] as const,
  agentTasks: (jobId: string) => ["agent-tasks", jobId] as const,
  agentTask: (taskId: string) => ["agent-task", taskId] as const,
  boxes: (jobId: string, imageId: string) => ["boxes", jobId, imageId] as const,
  models: ["models"] as const,
  run: (runId: string) => ["run", runId] as const,
  instanceArtifact: (runId: string) => ["instance-artifact", runId] as const,
  scientificReport: (jobId: string, runIds: string[]) =>
    ["scientific-report", jobId, ...runIds] as const,
  knowledge: ["knowledge"] as const
};
