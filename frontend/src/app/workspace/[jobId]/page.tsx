import type { Metadata } from "next";

import { WorkspaceCommandCenter } from "@/components/workspace/workspace-command-center";

export const metadata: Metadata = {
  title: "科研任务工作区"
};

export default async function WorkspacePage({
  params,
  searchParams
}: {
  params: Promise<{ jobId: string }>;
  searchParams: Promise<Record<string, string | string[] | undefined>>;
}) {
  const { jobId } = await params;
  const query = await searchParams;
  const initialRunId = typeof query.run === "string" ? query.run : null;
  const launchWarning =
    typeof query.autostart_failed === "string" ? query.autostart_failed : null;
  return (
    <WorkspaceCommandCenter
      jobId={jobId}
      initialRunId={initialRunId}
      autoRun={query.autorun === "1"}
      launchWarning={launchWarning}
    />
  );
}
