import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { ModelSelector } from "@/components/models/model-selector";
import { apiRequest } from "@/lib/api/client";
import type { ImageAsset, ModelList } from "@/lib/api/types";

vi.mock("@/lib/api/client", () => ({
  apiRequest: vi.fn()
}));

const mockedApi = vi.mocked(apiRequest);
const images = [
  { image_id: "image_1", filename: "BaNi-1.tif" },
  { image_id: "image_2", filename: "BaNi-2.tif" }
] as ImageAsset[];
const catalog = {
  models: [
    {
      model_id: "unet-batch",
      status: "ready",
      version: "1.0.0",
      family: "unet",
      health_error: null
    }
  ]
} as unknown as ModelList;

function renderSelector(onRunsCreated = vi.fn()) {
  const queryClient = new QueryClient({
    defaultOptions: {
      mutations: { retry: false },
      queries: { retry: false }
    }
  });
  render(
    <QueryClientProvider client={queryClient}>
      <ModelSelector
        jobId="job_batch"
        images={images}
        image={images[0]!}
        boxSet={null}
        catalog={catalog}
        writeBlocker={null}
        onRunsCreated={onRunsCreated}
      />
    </QueryClientProvider>
  );
}

beforeEach(() => {
  mockedApi.mockReset();
  mockedApi.mockResolvedValue({
    request_id: "req_batch",
    status: "accepted",
    data: { run_ids: ["run_1", "run_2"] },
    error: null
  });
});

describe("ModelSelector batch mode", () => {
  it("submits every image through the existing runs API by default", async () => {
    const user = userEvent.setup();
    const onRunsCreated = vi.fn();
    renderSelector(onRunsCreated);

    expect(screen.getByText("2 张批处理")).toBeVisible();
    await user.click(screen.getByRole("button", { name: "批量创建 2 个运行" }));

    await waitFor(() =>
      expect(mockedApi).toHaveBeenCalledWith("analyses/job_batch/runs", {
        method: "POST",
        body: expect.objectContaining({
          image_ids: ["image_1", "image_2"],
          model_ids: ["unet-batch"],
          roi_mode: "full_image"
        })
      })
    );
    expect(onRunsCreated).toHaveBeenCalledWith(["run_1", "run_2"]);
  });

  it("falls back to the single-image path without changing the API shape", async () => {
    const user = userEvent.setup();
    renderSelector();

    await user.click(
      screen.getByRole("radio", { name: "仅当前图像BaNi-1.tif" })
    );
    await user.click(screen.getByRole("button", { name: "开始分割" }));

    await waitFor(() =>
      expect(mockedApi).toHaveBeenCalledWith("analyses/job_batch/runs", {
        method: "POST",
        body: expect.objectContaining({
          image_ids: ["image_1"],
          model_ids: ["unet-batch"]
        })
      })
    );
  });
});
