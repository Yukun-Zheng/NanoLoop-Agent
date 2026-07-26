import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { ScientificInspector } from "@/components/shell/scientific-inspector";
import { apiRequest } from "@/lib/api/client";
import type { ImageAsset, Run } from "@/lib/api/types";
import { useWorkspaceStore } from "@/lib/store/workspace";

vi.mock("@/lib/api/client", () => ({
  apiRequest: vi.fn(),
  toBffArtifactUrl: vi.fn(() => null)
}));

const mockedApi = vi.mocked(apiRequest);
const image = {
  image_id: "img_1",
  job_id: "job_1",
  filename: "BaNi-3.tif",
  sha256: "a".repeat(64),
  width: 2048,
  height: 1536,
  bit_depth: 8,
  sample_id: "BaNi-3",
  experiment_conditions: {},
  scale_nm_per_pixel: null,
  scale_source: "none",
  analysis_roi: {
    schema_version: 1,
    coordinate_space: "original_px",
    valid_rect: { x1: 0, y1: 0, x2: 2048, y2: 1536 },
    invalid_rects: [],
    source: "none",
    revision: 1
  }
} as ImageAsset;
const run = {
  run_id: "run_parent",
  job_id: "job_1",
  image_id: "img_1",
  model_id: "unet-large-optimized-v1",
  status: "COMPLETED_WITH_WARNINGS",
  configuration: {
    scale_nm_per_pixel: null,
    scale_calibration: null
  }
} as unknown as Run;

function renderInspector(onChildCreated = vi.fn()) {
  const queryClient = new QueryClient({
    defaultOptions: {
      mutations: { retry: false },
      queries: { retry: false }
    }
  });
  render(
    <QueryClientProvider client={queryClient}>
      <ScientificInspector
        jobId="job_1"
        image={image}
        runIds={["run_parent"]}
        writeBlocker={null}
        health={null}
        model={null}
        run={run}
        answer={null}
        onLatestAnswer={vi.fn()}
        onChildCreated={onChildCreated}
      />
    </QueryClientProvider>
  );
}

beforeEach(() => {
  useWorkspaceStore.setState({ inspectorTab: "scale" });
  mockedApi.mockReset();
  mockedApi.mockResolvedValue({
    request_id: "req_scale",
    status: "success",
    data: { parent_run_id: "run_parent", run_id: "run_calibrated" },
    error: null
  });
});

describe("ScaleCalibrationInspector", () => {
  it("creates an immutable review run from scale-bar evidence", async () => {
    const user = userEvent.setup();
    const onChildCreated = vi.fn();
    renderInspector(onChildCreated);

    expect(screen.getByText("当前运行只有像素尺度")).toBeVisible();
    await user.type(screen.getByLabelText("标尺物理长度（nm）"), "100");
    await user.type(screen.getByLabelText("标尺像素长度（px）"), "184");
    await user.type(screen.getByLabelText("原图标签（用于审计）"), "100 nm");
    expect(screen.getByText(/0.543478 nm\/px/)).toBeVisible();

    await user.click(screen.getByRole("button", { name: "应用尺度并创建复核运行" }));

    await waitFor(() =>
      expect(mockedApi).toHaveBeenCalledWith("runs/run_parent/review", {
        method: "POST",
        body: {
          scale_calibration: {
            physical_length_nm: 100,
            pixel_length_px: 184,
            label_text: "100 nm",
            method: "manual_scale_bar"
          }
        }
      })
    );
    expect(onChildCreated).toHaveBeenCalledWith("run_calibrated");
  });
});
