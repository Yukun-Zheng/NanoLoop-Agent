"use client";

import { Download, FileQuestion, ImageOff, Minus, Plus, ScanLine } from "lucide-react";
import { useEffect, useState } from "react";

import { Button } from "@/components/ui/button";
import { fetchArtifact, toBffArtifactUrl } from "@/lib/api/client";
import { errorMessage } from "@/lib/api/errors";

type PreviewState =
  | { source: string | null; status: "loading" }
  | { source: string; status: "ready"; objectUrl: string; contentType: string }
  | { source: string; status: "unsupported"; contentType: string }
  | { source: string; status: "error"; message: string };

export function ArtifactPreview({
  url,
  alt,
  filename
}: {
  url: string | null | undefined;
  alt: string;
  filename: string;
}) {
  const [state, setState] = useState<PreviewState>({
    source: null,
    status: "loading"
  });
  const [fit, setFit] = useState(true);
  const [zoom, setZoom] = useState(1);

  useEffect(() => {
    if (!url) return;
    let active = true;
    let objectUrl: string | null = null;
    void fetchArtifact(url)
      .then(async (response) => {
        const contentType = response.headers.get("content-type") || "application/octet-stream";
        if (!contentType.startsWith("image/") || contentType.includes("tiff")) {
          if (active) setState({ source: url, status: "unsupported", contentType });
          return;
        }
        const blob = await response.blob();
        if (!active) return;
        objectUrl = URL.createObjectURL(blob);
        setState({ source: url, status: "ready", objectUrl, contentType });
      })
      .catch((error: unknown) => {
        if (active) {
          setState({ source: url, status: "error", message: errorMessage(error) });
        }
      });
    return () => {
      active = false;
      if (objectUrl) URL.revokeObjectURL(objectUrl);
    };
  }, [url]);

  if (!url) {
    return (
      <div className="artifact-fallback">
        <ImageOff size={22} />
        <strong>本次运行未生成该图层</strong>
      </div>
    );
  }

  const visibleState: PreviewState =
    state.source === url ? state : { source: url, status: "loading" };

  if (visibleState.status === "ready") {
    return (
      <div className="artifact-preview-ready">
        <div className="artifact-view-controls" aria-label="图层视图控制">
          <button
            type="button"
            className={fit ? "active" : undefined}
            onClick={() => {
              setFit(true);
              setZoom(1);
            }}
          >
            <ScanLine size={13} />适应
          </button>
          <button
            type="button"
            className={!fit && zoom === 1 ? "active" : undefined}
            onClick={() => {
              setFit(false);
              setZoom(1);
            }}
          >
            1:1
          </button>
          <button
            type="button"
            aria-label="缩小图层"
            onClick={() => {
              setFit(false);
              setZoom((value) => Math.max(0.5, value - 0.25));
            }}
          >
            <Minus size={13} />
          </button>
          <span>{Math.round(zoom * 100)}%</span>
          <button
            type="button"
            aria-label="放大图层"
            onClick={() => {
              setFit(false);
              setZoom((value) => Math.min(3, value + 0.25));
            }}
          >
            <Plus size={13} />
          </button>
        </div>
        <div className={`artifact-scroll${fit ? " is-fit" : " is-actual"}`}>
          {/* Signed bytes render only from a short-lived same-origin object URL. */}
          {/* eslint-disable-next-line @next/next/no-img-element */}
          <img
            className="artifact-image"
            src={visibleState.objectUrl}
            alt={alt}
            style={{ transform: `scale(${zoom})` }}
          />
        </div>
      </div>
    );
  }

  if (visibleState.status === "unsupported") {
    return (
      <div className="artifact-fallback">
        <FileQuestion size={22} />
        <strong>此制品仅支持下载审查</strong>
        <p>
          {visibleState.contentType.includes("tiff")
            ? "浏览器没有统一的 TIFF 预览能力。"
            : "概率数组或结构化制品不是可直接显示的图片。"}
        </p>
        <Button asChild size="sm">
          <a href={toBffArtifactUrl(url) || "#"} download={filename}>
            <Download size={14} />下载原始制品
          </a>
        </Button>
      </div>
    );
  }

  if (visibleState.status === "error") {
    return (
      <div className="artifact-fallback">
        <ImageOff size={22} />
        <strong>图层暂时无法载入</strong>
        <p>{visibleState.message}</p>
      </div>
    );
  }

  return (
    <div className="artifact-fallback">
      <span className="status-spinner" />
      <strong>正在验证并载入制品…</strong>
    </div>
  );
}
