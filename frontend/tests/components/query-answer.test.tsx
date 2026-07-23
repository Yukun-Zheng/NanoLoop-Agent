import { render, screen, within } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import { QueryAnswer } from "@/components/agent/query-answer";

describe("QueryAnswer", () => {
  it("renders auditable data fields and maps answer markers to citation cards", () => {
    const answer = "实验数据显示差异。[C1]\n文献证据仍然有限。";
    render(
      <QueryAnswer
        response={{
          query_type: "mixed",
          answer,
          confidence: "medium",
          outcome_code: "INSUFFICIENT_EVIDENCE",
          data_evidence: [
            {
              tool_name: "compare_models",
              validated_arguments: {
                metric: "number_density_px2",
                run_ids: ["run-1"]
              },
              source_run_ids: ["run-1"],
              aggregates: { particle_count_delta: 2 },
              rows: [
                {
                  run_id: "run-1",
                  particle_count: 3,
                  number_density_px2: 0.000061035
                }
              ],
              units: {
                number_density_px2: "px^-2",
                particle_count: "count"
              },
              quality_warnings: ["仅比较同一 ROI"],
              chart_url: "/api/v1/files/v2.kid.payload.signature"
            }
          ],
          citations: [
            {
              citation_id: "C1",
              doc_id: "doc-1",
              title: "Team note",
              chunk_id: "chunk-1",
              excerpt: "Evidence excerpt",
              retrieval_score: 0.8,
              source_type: "material_note",
              citation_text: "NanoLoop team, Materials note, 2026."
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
    expect(screen.getByTestId("answer-copy").textContent).toBe(answer);

    const citationLink = screen.getByRole("link", { name: "跳转到引用 C1" });
    expect(citationLink).toHaveAttribute("href", "#citation-C1-1");
    expect(screen.getAllByText("[C1]")).toHaveLength(2);
    expect(screen.getByText("NanoLoop team, Materials note, 2026.")).toBeVisible();
    expect(screen.getByText("来源类型 material_note")).toBeVisible();

    const argumentsBlock = screen.getByLabelText("compare_models 已验证参数");
    expect(argumentsBlock).toHaveTextContent('"metric": "number_density_px2"');
    expect(argumentsBlock).toHaveTextContent('"run-1"');

    const unitsBlock = screen.getByLabelText("compare_models 单位");
    expect(unitsBlock).toHaveTextContent('"number_density_px2": "px^-2"');

    const table = screen.getByRole("table", { name: "compare_models 数据明细" });
    expect(
      within(table).getByRole("columnheader", {
        name: "number_density_px2 (px^-2)"
      })
    ).toBeVisible();
    expect(within(table).getByText("0.000061035")).toBeVisible();
    expect(screen.getByText("仅比较同一 ROI")).toBeVisible();

    const chartLink = screen.getByRole("link", { name: "打开图表制品" });
    expect(chartLink).toHaveAttribute(
      "href",
      "/api/nanoloop/files/v2.kid.payload.signature"
    );
  });

  it("does not expose an unsafe chart URL as a link", () => {
    render(
      <QueryAnswer
        response={{
          query_type: "analysis_data",
          answer: "没有可安全加载的图表。",
          confidence: "low",
          outcome_code: "INSUFFICIENT_EVIDENCE",
          data_evidence: [
            {
              tool_name: "summarize_runs",
              validated_arguments: {},
              rows: [],
              aggregates: {},
              units: {},
              source_run_ids: [],
              quality_warnings: [],
              chart_url: "https://evil.test/api/v1/files/token"
            }
          ],
          citations: [],
          limitations: [],
          needs_clarification: false,
          tool_calls: []
        }}
      />
    );

    expect(screen.queryByRole("link", { name: "打开图表制品" })).not.toBeInTheDocument();
    expect(screen.getByText("图表地址未通过安全校验，未提供链接。")).toBeVisible();
  });
});
