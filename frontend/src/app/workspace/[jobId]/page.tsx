import type { Metadata } from "next";

import { WorkspaceCommandCenter } from "@/components/workspace/workspace-command-center";

export const metadata: Metadata = {
  title: "科研任务工作区"
};

export default async function WorkspacePage({
  params
}: {
  params: Promise<{ jobId: string }>;
}) {
  const { jobId } = await params;
  return <WorkspaceCommandCenter jobId={jobId} />;
}
