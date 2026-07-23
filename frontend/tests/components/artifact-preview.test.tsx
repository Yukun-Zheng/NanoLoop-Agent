import { act, render, screen, waitFor } from "@testing-library/react";
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
});
