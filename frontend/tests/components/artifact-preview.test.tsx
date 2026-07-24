import { act, render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, describe, expect, it, vi } from "vitest";

import { ArtifactPreview } from "@/components/results/artifact-preview";

const apiMocks = vi.hoisted(() => ({
  fetchArtifact: vi.fn()
}));

vi.mock("@/lib/api/client", () => ({
  fetchArtifact: apiMocks.fetchArtifact,
  toBffArtifactUrl: (value: string) => value
}));

function deferred<T>() {
  let resolve!: (value: T) => void;
  const promise = new Promise<T>((next) => {
    resolve = next;
  });
  return { promise, resolve };
}

function imageResponse(bytes: string) {
  return new Response(new Blob([bytes], { type: "image/png" }), {
    headers: { "content-type": "image/png" }
  });
}

afterEach(() => {
  apiMocks.fetchArtifact.mockReset();
  vi.restoreAllMocks();
});

describe("ArtifactPreview", () => {
  it("never presents a previous artifact under the next layer label", async () => {
    const first = deferred<Response>();
    const second = deferred<Response>();
    apiMocks.fetchArtifact
      .mockReturnValueOnce(first.promise)
      .mockReturnValueOnce(second.promise);
    const createObjectUrl = vi
      .spyOn(URL, "createObjectURL")
      .mockReturnValueOnce("blob:first")
      .mockReturnValueOnce("blob:second");
    const revokeObjectUrl = vi.spyOn(URL, "revokeObjectURL").mockImplementation(() => {});

    const view = render(
      <ArtifactPreview url="/api/v1/files/first" alt="Mask 图层" filename="first" />
    );
    await act(async () => first.resolve(imageResponse("first")));
    expect(await screen.findByRole("img", { name: "Mask 图层" })).toHaveAttribute(
      "src",
      "blob:first"
    );

    view.rerender(
      <ArtifactPreview url="/api/v1/files/second" alt="Overlay 图层" filename="second" />
    );
    expect(screen.queryByRole("img")).not.toBeInTheDocument();
    expect(screen.getByText("正在验证并载入制品…")).toBeVisible();
    expect(revokeObjectUrl).toHaveBeenCalledWith("blob:first");

    await act(async () => second.resolve(imageResponse("second")));
    await waitFor(() =>
      expect(screen.getByRole("img", { name: "Overlay 图层" })).toHaveAttribute(
        "src",
        "blob:second"
      )
    );
    expect(apiMocks.fetchArtifact).toHaveBeenNthCalledWith(
      1,
      "/api/v1/files/first",
      { preview: true }
    );
    expect(apiMocks.fetchArtifact).toHaveBeenNthCalledWith(
      2,
      "/api/v1/files/second",
      { preview: true }
    );
    expect(createObjectUrl).toHaveBeenCalledTimes(2);
  });

  it("does not allocate an object URL when an obsolete request finishes late", async () => {
    const obsolete = deferred<Response>();
    const current = deferred<Response>();
    apiMocks.fetchArtifact
      .mockReturnValueOnce(obsolete.promise)
      .mockReturnValueOnce(current.promise);
    const createObjectUrl = vi
      .spyOn(URL, "createObjectURL")
      .mockReturnValue("blob:current");
    vi.spyOn(URL, "revokeObjectURL").mockImplementation(() => {});

    const view = render(
      <ArtifactPreview url="/api/v1/files/obsolete" alt="旧图层" filename="old" />
    );
    view.rerender(
      <ArtifactPreview url="/api/v1/files/current" alt="当前图层" filename="current" />
    );

    await act(async () => obsolete.resolve(imageResponse("obsolete")));
    expect(createObjectUrl).not.toHaveBeenCalled();

    await act(async () => current.resolve(imageResponse("current")));
    expect(await screen.findByRole("img", { name: "当前图层" })).toBeVisible();
    expect(createObjectUrl).toHaveBeenCalledOnce();
  });

  it("keeps the raw artifact download available when preview generation fails", async () => {
    apiMocks.fetchArtifact.mockRejectedValueOnce(new Error("preview failed"));

    render(
      <ArtifactPreview
        url="/api/v1/files/tiff"
        alt="TIFF 原图"
        filename="sample.tif"
      />
    );

    expect(await screen.findByText("图层暂时无法载入")).toBeVisible();
    expect(screen.getByRole("link", { name: "下载原始制品" })).toHaveAttribute(
      "href",
      "/api/v1/files/tiff"
    );
  });

  it("shows the fixed confidence legend for probability previews", async () => {
    apiMocks.fetchArtifact.mockResolvedValueOnce(imageResponse("probability"));
    vi.spyOn(URL, "createObjectURL").mockReturnValue("blob:probability");
    vi.spyOn(URL, "revokeObjectURL").mockImplementation(() => {});

    render(
      <ArtifactPreview
        url="/api/v1/files/probability"
        alt="置信度图层"
        filename="probability.npy"
        mode="probability"
      />
    );

    expect(await screen.findByRole("img", { name: "置信度图层" })).toBeVisible();
    expect(screen.getByLabelText("置信度色阶")).toHaveTextContent("低 0高 1");
  });

  it("enlarges instance targets through CSS and copies the authoritative number", async () => {
    const user = userEvent.setup();
    const writeText = vi.fn().mockResolvedValue(undefined);
    Object.defineProperty(navigator, "clipboard", {
      configurable: true,
      value: { writeText }
    });
    apiMocks.fetchArtifact.mockResolvedValueOnce(imageResponse("instances"));
    vi.spyOn(URL, "createObjectURL").mockReturnValue("blob:instances");
    vi.spyOn(URL, "revokeObjectURL").mockImplementation(() => {});

    render(
      <ArtifactPreview
        url="/api/v1/files/labeled"
        alt="实例编号图层"
        filename="labeled.png"
        mode="instances"
        instances={{
          width: 200,
          height: 100,
          labels: [
            {
              instanceIndex: 17,
              xPercent: 20,
              yPercent: 30,
              confidence: 0.875
            }
          ]
        }}
      />
    );

    const target = await screen.findByRole("button", { name: "复制实例编号 17" });
    expect(target).toHaveAttribute("title", "实例 17 · 点击复制");
    expect(target).toHaveTextContent("置信度 87.5%");
    await user.click(target);
    expect(writeText).toHaveBeenCalledWith("17");
  });
});
