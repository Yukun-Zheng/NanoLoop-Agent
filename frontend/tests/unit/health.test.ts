import { describe, expect, it } from "vitest";

import { coreMutationBlocker } from "@/lib/health";

const healthy = {
  service: { status: "healthy" as const, detail: "ready" },
  database: { status: "healthy" as const, detail: "ready" },
  model_registry: { status: "degraded" as const, detail: "no ready models" },
  rag_index: { status: "unavailable" as const, detail: "not configured" },
  version: "1.0.0"
};

describe("coreMutationBlocker", () => {
  it("allows core writes even when optional model or RAG components are degraded", () => {
    expect(coreMutationBlocker(healthy)).toBeNull();
  });

  it("fails closed when health is missing or a core component is not healthy", () => {
    expect(coreMutationBlocker(undefined, { pending: true })).toContain("正在确认");
    expect(
      coreMutationBlocker({
        ...healthy,
        database: { status: "degraded", detail: "migration stale" }
      })
    ).toContain("数据库");
    expect(coreMutationBlocker(healthy, { failed: true })).toContain("已暂停");
  });
});
