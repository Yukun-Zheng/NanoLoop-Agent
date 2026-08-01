import { describe, expect, it } from "vitest";

import {
  isAllowedProxyRequest,
  isKnownProxyPath
} from "@/lib/api/route-policy";

const allowed = [
  ["GET", "health"],
  ["POST", "analyses"],
  ["GET", "analyses/job-1"],
  ["GET", "analyses/job-1/export"],
  ["POST", "analyses/job-1/report"],
  ["GET", "analyses/job-1/images/image-1/boxes"],
  ["PUT", "analyses/job-1/images/image-1/boxes"],
  ["GET", "analyses/job-1/queries"],
  ["POST", "analyses/job-1/query"],
  ["GET", "analyses/job-1/agent-tasks"],
  ["POST", "analyses/job-1/agent-tasks"],
  ["GET", "agent-tasks/agt-1"],
  ["POST", "agent-tasks/agt-1/run"],
  ["POST", "agent-tasks/agt-1/input"],
  ["POST", "agent-tasks/agt-1/cancel"],
  ["POST", "agent-tasks/agt-1/approvals/apr-1"],
  ["POST", "analyses/job-1/runs"],
  ["GET", "files/v2.token.signature"],
  ["GET", "knowledge/documents"],
  ["POST", "knowledge/documents"],
  ["PATCH", "knowledge/documents/doc-1"],
  ["POST", "knowledge/reindex"],
  ["GET", "models"],
  ["POST", "models/recommend"],
  ["GET", "runs/run-1"],
  ["POST", "runs/run-1/corrected-mask"],
  ["POST", "runs/run-1/review"]
] as const;

describe("BFF route policy", () => {
  it.each(allowed)("allows %s %s", (method, path) => {
    expect(isAllowedProxyRequest(path, method)).toBe(true);
  });

  it.each([
    ["DELETE", "analyses/job-1"],
    ["POST", "health"],
    ["POST", "analyses/job-1/queries"],
    ["DELETE", "agent-tasks/agt-1"],
    ["GET", "agent-tasks/agt-1/run"],
    ["GET", "agent-tasks/agt-1/approvals/apr-1"],
    ["GET", "models/recommend"],
    ["PUT", "knowledge/documents/doc-1"]
  ])("recognizes a path but rejects wrong method", (method, path) => {
    expect(isKnownProxyPath(path)).toBe(true);
    expect(isAllowedProxyRequest(path, method)).toBe(false);
  });

  it.each([
    "https://evil.test/api/v1/health",
    "//evil.test/files/token",
    "../health",
    "files/token/extra",
    "files/a%2Fb",
    "openapi.json",
    "metrics",
    "runs/run-1/delete",
    "analyses//job",
    "analyses/job?admin=true",
    "analyses/job#fragment",
    "analyses/\njob"
  ])("rejects an unsafe or unknown path: %s", (path) => {
    expect(isKnownProxyPath(path)).toBe(false);
  });
});
