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

export function removeRecentJob(
  jobId: string,
  storage: Pick<Storage, "getItem" | "setItem"> = localStorage
) {
  const next = readRecentJobs(storage).filter((item) => item.jobId !== jobId);
  storage.setItem(storageKey, JSON.stringify(next));
  return next;
}

export function clearRecentJobs(
  storage: Pick<Storage, "setItem"> = localStorage
) {
  storage.setItem(storageKey, "[]");
  return [] satisfies RecentJob[];
}

export function parseJobReference(value: string): string | null {
  const reference = value.trim();
  if (!reference) return null;
  if (/^job_[A-Za-z0-9_-]{8,255}$/.test(reference)) return reference;

  try {
    const url = new URL(reference, "http://nanoloop.local");
    const match = url.pathname.match(/^\/workspace\/([^/]+)\/?$/);
    if (!match) return null;
    const jobId = decodeURIComponent(match[1] ?? "");
    return /^job_[A-Za-z0-9_-]{8,255}$/.test(jobId) ? jobId : null;
  } catch {
    return null;
  }
}
