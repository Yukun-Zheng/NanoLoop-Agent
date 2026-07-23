import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import { QueryAnswer } from "@/components/agent/query-answer";

describe("QueryAnswer", () => {
  it("keeps data evidence, citations, and limitations visibly separated", () => {
    render(
      <QueryAnswer
        response={{
          query_type: "mixed",
          answer: "实验数据显示差异，文献证据有限。",
          confidence: "medium",
          outcome_code: "INSUFFICIENT_EVIDENCE",
          data_evidence: [
            {
              tool_name: "compare_models",
              validated_arguments: {},
              source_run_ids: ["run-1"],
              aggregates: { particle_count_delta: 2 },
              rows: [],
              units: {},
              quality_warnings: []
            }
          ],
          citations: [
            {
              citation_id: "c-1",
              doc_id: "doc-1",
              title: "Team note",
              chunk_id: "chunk-1",
              excerpt: "Evidence excerpt",
              retrieval_score: 0.8
            }
          ],
          limitations: ["样本量有限"],
          needs_clarification: false,
          tool_calls: []
        }}
      />
    );
    expect(screen.getByText("实验数据证据")).toBeVisible();
    expect(screen.getByText("材料知识证据")).toBeVisible();
    expect(screen.getByText("限制")).toBeVisible();
    expect(screen.getByText("证据不足")).toBeVisible();
  });
});
