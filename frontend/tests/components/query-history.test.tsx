import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

import { QueryHistory } from "@/components/agent/query-history";
import type { QueryHistoryData } from "@/lib/api/types";

const record = {
  query_id: "query-1",
  job_id: "job-1",
  image_id: "image-1",
  request: {
    question: "当前周长密度是多少？",
    query_type: "analysis_data",
    image_id: "image-1",
    run_ids: ["run-1"],
    material_context: null
  },
  response: {
    query_type: "analysis_data",
    answer: "周长密度为 30 um^-1。",
    data_evidence: [],
    citations: [],
    tool_calls: [],
    material_context: null,
    confidence: "high",
    limitations: [],
    needs_clarification: false,
    outcome_code: "OK"
  },
  created_at: "2026-07-24T06:00:00Z"
} satisfies NonNullable<QueryHistoryData["items"]>[number];

describe("QueryHistory", () => {
  it("renders scoped audit entries and restores the selected answer", () => {
    const onSelect = vi.fn();
    render(<QueryHistory items={[record]} onSelect={onSelect} />);

    expect(screen.getByText("当前作用域问答历史")).toBeInTheDocument();
    expect(screen.getByText("当前周长密度是多少？")).toBeInTheDocument();
    expect(screen.getByText("置信度 high")).toBeInTheDocument();

    fireEvent.click(
      screen.getByRole("button", {
        name: "查看历史问答：当前周长密度是多少？"
      })
    );
    expect(onSelect).toHaveBeenCalledWith(record);
  });
});
