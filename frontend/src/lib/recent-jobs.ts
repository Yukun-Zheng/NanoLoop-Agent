export type RecentJob = {
  jobId: string;
  name: string;
  openedAt: string;
};

const storageKey = "nanoloop:recent-jobs:v1";
const maximumItems = 8;

export function readRecentJobs(storage: Pick<Storage, "getItem"> = localStorage): RecentJob[] {
  try {
    const value: unknown = JSON.parse(storage.getItem(storageKey) || "[]");
    if (!Array.isArray(value)) return [];
    return value
      .filter(
        (item): item is RecentJob =>
          typeof item === "object" &&
          item !== null &&
          typeof item.jobId === "string" &&
          typeof item.name === "string" &&
          typeof item.openedAt === "string"
      )
      .slice(0, maximumItems);
  } catch {
    return [];
  }
}

export function rememberJob(
  job: Omit<RecentJob, "openedAt"> & { openedAt?: string },
  storage: Pick<Storage, "getItem" | "setItem"> = localStorage
) {
  const next = [
    {
      jobId: job.jobId,
      name: job.name,
      openedAt: job.openedAt || new Date().toISOString()
    },
    ...readRecentJobs(storage).filter((item) => item.jobId !== job.jobId)
  ].slice(0, maximumItems);
  storage.setItem(storageKey, JSON.stringify(next));
  return next;
}
