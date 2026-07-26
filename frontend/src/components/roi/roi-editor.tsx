"use client";

import { useMutation, useQueryClient } from "@tanstack/react-query";
import { Clipboard, MousePointer2, Plus, RefreshCcw, Save, Trash2 } from "lucide-react";
import type Konva from "konva";
import { useEffect, useMemo, useRef, useState } from "react";
import {
  Image as KonvaImage,
  Layer,
  Rect,
  Stage,
  Text,
  Transformer
} from "react-konva";

import { Button } from "@/components/ui/button";
import { RequestError } from "@/components/ui/request-error";
import {
  apiRequest,
  artifactPreviewIdentity,
  toBffArtifactUrl
} from "@/lib/api/client";
import { NanoLoopApiError } from "@/lib/api/errors";
import { queryKeys } from "@/lib/api/query-keys";
import type { BoxSet, ImageAsset, RoiBox } from "@/lib/api/types";
import {
  canvasTransform,
  displayToOriginal,
  originalToDisplay,
  validateRoiRect,
  type PixelRect
} from "@/lib/roi/geometry";

const canvasHeight = 440;

function artifactUrls(source: string | null | undefined) {
  return {
    identity: artifactPreviewIdentity(source),
    previewUrl: toBffArtifactUrl(source, { preview: true }),
    rawUrl: toBffArtifactUrl(source)
  };
}

function useStableArtifactUrls(source: string | null | undefined) {
  const identity = artifactPreviewIdentity(source);
  const [urls, setUrls] = useState(() => artifactUrls(source));

  useEffect(() => {
    if (urls.identity === identity) return;
    const frame = requestAnimationFrame(() => setUrls(artifactUrls(source)));
    return () => cancelAnimationFrame(frame);
  }, [identity, source, urls.identity]);

  return urls;
}

function useHtmlImage(src: string | null) {
  const [state, setState] = useState<{
    image: HTMLImageElement | null;
    status: "loading" | "ready" | "error";
  }>({ image: null, status: "loading" });
  useEffect(() => {
    let active = true;
    if (!src) {
      queueMicrotask(() => {
        if (active) setState({ image: null, status: "error" });
      });
      return () => {
        active = false;
      };
    }
    queueMicrotask(() => {
      if (active) {
        setState((current) => ({ image: current.image, status: "loading" }));
      }
    });
    const next = new window.Image();
    next.onload = () => {
      if (active) setState({ image: next, status: "ready" });
    };
    next.onerror = () => {
      if (active) {
        setState((current) => ({ image: current.image, status: "error" }));
      }
    };
    next.src = src;
    return () => {
      active = false;
      next.onload = null;
      next.onerror = null;
    };
  }, [src]);
  return state;
}

export function RoiEditor({
  jobId,
  image,
  serverBoxes,
  writeBlocker
}: {
  jobId: string;
  image: ImageAsset;
  serverBoxes: BoxSet;
  writeBlocker: string | null;
}) {
  const queryClient = useQueryClient();
  const container = useRef<HTMLDivElement>(null);
  const [canvasWidth, setCanvasWidth] = useState(760);
  const [boxes, setBoxes] = useState<RoiBox[]>(serverBoxes.boxes || []);
  const [baseRevision, setBaseRevision] = useState(serverBoxes.revision);
  const [dirty, setDirty] = useState(false);
  const [draft, setDraft] = useState<PixelRect | null>(null);
  const [drawingFrom, setDrawingFrom] = useState<{ x: number; y: number } | null>(null);
  const [selectedBoxIndex, setSelectedBoxIndex] = useState<number | null>(null);
  const [conflict, setConflict] = useState(false);
  const loadedImageId = useRef(image.image_id);
  const selectedRect = useRef<Konva.Rect>(null);
  const transformer = useRef<Konva.Transformer>(null);
  const { previewUrl, rawUrl } = useStableArtifactUrls(image.original_download_url);
  const imagePreview = useHtmlImage(previewUrl);

  useEffect(() => {
    const frame = requestAnimationFrame(() => {
      if (loadedImageId.current !== image.image_id) {
        loadedImageId.current = image.image_id;
        setBoxes(serverBoxes.boxes || []);
        setBaseRevision(serverBoxes.revision);
        setDirty(false);
        setConflict(false);
        setSelectedBoxIndex(null);
        return;
      }
      if (serverBoxes.revision === baseRevision) return;
      if (dirty) {
        setConflict(true);
        return;
      }
      setBoxes(serverBoxes.boxes || []);
      setBaseRevision(serverBoxes.revision);
      setConflict(false);
      setSelectedBoxIndex((current) =>
        current !== null && current < (serverBoxes.boxes ?? []).length
          ? current
          : null
      );
    });
    return () => cancelAnimationFrame(frame);
  }, [baseRevision, dirty, image.image_id, serverBoxes]);

  useEffect(() => {
    if (!container.current) return;
    const observer = new ResizeObserver(([entry]) => {
      if (entry) setCanvasWidth(Math.max(320, Math.floor(entry.contentRect.width)));
    });
    observer.observe(container.current);
    return () => observer.disconnect();
  }, []);

  const transform = useMemo(
    () => canvasTransform(canvasWidth, canvasHeight, image.width, image.height),
    [canvasWidth, image.height, image.width]
  );
  const validRect = image.analysis_roi.valid_rect;
  const invalidRects = image.analysis_roi.invalid_rects || [];
  const displayValidRect = useMemo(
    () => originalToDisplay(validRect, transform),
    [transform, validRect]
  );
  const errors = boxes.map((box) =>
    validateRoiRect(box, validRect, invalidRects, 32)
  );

  useEffect(() => {
    const selectedNode = selectedRect.current;
    const transformerNode = transformer.current;
    if (!selectedNode || !transformerNode || selectedBoxIndex === null) return;
    transformerNode.nodes([selectedNode]);
    transformerNode.getLayer()?.batchDraw();
  }, [boxes.length, selectedBoxIndex, transform]);

  const save = useMutation({
    mutationFn: () =>
      apiRequest<BoxSet>(
        `analyses/${encodeURIComponent(jobId)}/images/${encodeURIComponent(image.image_id)}/boxes`,
        {
          method: "PUT",
          body: {
            expected_revision: baseRevision,
            boxes
          }
        }
      ),
    onSuccess: (response) => {
      setBoxes(response.data.boxes || []);
      setBaseRevision(response.data.revision);
      setDirty(false);
      setConflict(false);
      setSelectedBoxIndex((current) =>
        current !== null && current < (response.data.boxes ?? []).length
          ? current
          : null
      );
      queryClient.setQueryData(queryKeys.boxes(jobId, image.image_id), response.data);
    },
    onError(error) {
      if (error instanceof NanoLoopApiError && error.status === 409) setConflict(true);
    }
  });

  const reload = useMutation({
    mutationFn: () =>
      apiRequest<BoxSet>(
        `analyses/${encodeURIComponent(jobId)}/images/${encodeURIComponent(image.image_id)}/boxes`
      ),
    onSuccess: (response) => {
      setBoxes(response.data.boxes || []);
      setBaseRevision(response.data.revision);
      setDirty(false);
      setConflict(false);
      setSelectedBoxIndex(null);
      queryClient.setQueryData(queryKeys.boxes(jobId, image.image_id), response.data);
    }
  });

  function startDrawing(event: Konva.KonvaEventObject<MouseEvent | TouchEvent>) {
    if (boxes.length >= 20) return;
    const stage = event.target.getStage();
    if (!stage || event.target !== stage) return;
    const position = stage.getPointerPosition();
    if (!position) return;
    setSelectedBoxIndex(null);
    setDrawingFrom(position);
    setDraft({ x1: position.x, y1: position.y, x2: position.x, y2: position.y });
  }

  function continueDrawing(event: Konva.KonvaEventObject<MouseEvent | TouchEvent>) {
    if (!drawingFrom) return;
    const position = event.target.getStage()?.getPointerPosition();
    if (!position) return;
    setDraft({
      x1: Math.min(drawingFrom.x, position.x),
      y1: Math.min(drawingFrom.y, position.y),
      x2: Math.max(drawingFrom.x, position.x),
      y2: Math.max(drawingFrom.y, position.y)
    });
  }

  function finishDrawing() {
    if (!draft) return;
    const original = displayToOriginal(draft, transform, image.width, image.height);
    if (!validateRoiRect(original, validRect, invalidRects, 32)) {
      setSelectedBoxIndex(boxes.length);
      setBoxes((current) => [
        ...current,
        {
          box_id: null,
          label: `ROI ${current.length + 1}`,
          active: true,
          ...original
        }
      ]);
      setDirty(true);
    }
    setDraft(null);
    setDrawingFrom(null);
  }

  function updateBox(index: number, patch: Partial<RoiBox>) {
    setDirty(true);
    setBoxes((current) =>
      current.map((box, boxIndex) => (boxIndex === index ? { ...box, ...patch } : box))
    );
  }

  function clamp(value: number, minimum: number, maximum: number) {
    return Math.min(Math.max(value, minimum), Math.max(minimum, maximum));
  }

  function commitSelectedTransform(index: number) {
    const node = selectedRect.current;
    if (!node) return;
    const width = Math.max(1, node.width() * node.scaleX());
    const height = Math.max(1, node.height() * node.scaleY());
    const display: PixelRect = {
      x1: node.x(),
      y1: node.y(),
      x2: node.x() + width,
      y2: node.y() + height
    };
    node.scaleX(1);
    node.scaleY(1);
    node.width(width);
    node.height(height);
    updateBox(
      index,
      displayToOriginal(display, transform, image.width, image.height)
    );
  }

  const selectedBox =
    selectedBoxIndex === null ? null : boxes[selectedBoxIndex] ?? null;
  const selectedDisplay = selectedBox
    ? originalToDisplay(selectedBox, transform)
    : null;

  return (
    <div className="roi-editor">
      <div className="roi-canvas" ref={container}>
        <Stage
          width={canvasWidth}
          height={canvasHeight}
          onMouseDown={startDrawing}
          onMouseMove={continueDrawing}
          onMouseUp={finishDrawing}
          onTouchStart={startDrawing}
          onTouchMove={continueDrawing}
          onTouchEnd={finishDrawing}
        >
          <Layer>
            <Rect
              width={canvasWidth}
              height={canvasHeight}
              fill="#eef0f5"
              listening={false}
            />
            {imagePreview.image ? (
              <KonvaImage
                image={imagePreview.image}
                x={transform.offsetX}
                y={transform.offsetY}
                width={image.width * transform.scale}
                height={image.height * transform.scale}
                listening={false}
              />
            ) : (
              <Text
                x={Math.max(20, transform.offsetX + 20)}
                y={canvasHeight / 2 - 12}
                width={Math.max(240, image.width * transform.scale - 40)}
                align="center"
                text={
                  imagePreview.status === "loading"
                    ? "正在生成原图预览…"
                    : "原图预览加载失败；可下载原图检查，或继续使用右侧坐标精调"
                }
                fontSize={12}
                fill="#676d7a"
                listening={false}
              />
            )}
            {invalidRects.map((rect, index) => {
              const display = originalToDisplay(rect, transform);
              return (
                <Rect
                  key={`invalid-${index}`}
                  x={display.x1}
                  y={display.y1}
                  width={display.x2 - display.x1}
                  height={display.y2 - display.y1}
                  fill="rgba(200,70,70,.18)"
                  listening={false}
                />
              );
            })}
            {(() => {
              const display = originalToDisplay(validRect, transform);
              return (
                <Rect
                  x={display.x1}
                  y={display.y1}
                  width={display.x2 - display.x1}
                  height={display.y2 - display.y1}
                  stroke="#5f6ff5"
                  strokeWidth={1.5}
                  dash={[7, 5]}
                  listening={false}
                />
              );
            })()}
            {boxes.map((box, index) => {
              if (index === selectedBoxIndex) return null;
              const display = originalToDisplay(box, transform);
              const hasError = Boolean(errors[index]);
              return (
                <Rect
                  key={`${box.box_id || "draft"}-${index}`}
                  x={display.x1}
                  y={display.y1}
                  width={display.x2 - display.x1}
                  height={display.y2 - display.y1}
                  fill={hasError ? "rgba(200,70,70,.12)" : "rgba(95,111,245,.12)"}
                  stroke={hasError ? "#c84646" : "#5f6ff5"}
                  strokeWidth={2}
                  listening={false}
                />
              );
            })}
            {boxes.map((box, index) => {
              if (index === selectedBoxIndex) return null;
              const display = originalToDisplay(box, transform);
              return (
                <Text
                  key={`label-${index}`}
                  x={display.x1 + 5}
                  y={display.y1 + 5}
                  text={box.label || `ROI ${index + 1}`}
                  fontSize={12}
                  fill="#ffffff"
                  padding={4}
                  cornerRadius={4}
                  background="#4656db"
                  listening={false}
                />
              );
            })}
            {selectedBox && selectedDisplay ? (
              <>
                <Rect
                  ref={selectedRect}
                  x={selectedDisplay.x1}
                  y={selectedDisplay.y1}
                  width={selectedDisplay.x2 - selectedDisplay.x1}
                  height={selectedDisplay.y2 - selectedDisplay.y1}
                  fill={
                    errors[selectedBoxIndex ?? -1]
                      ? "rgba(200,70,70,.16)"
                      : "rgba(95,111,245,.18)"
                  }
                  stroke={
                    errors[selectedBoxIndex ?? -1] ? "#c84646" : "#3143df"
                  }
                  strokeWidth={2.5}
                  draggable
                  dragBoundFunc={(position) => {
                    const width = selectedDisplay.x2 - selectedDisplay.x1;
                    const height = selectedDisplay.y2 - selectedDisplay.y1;
                    return {
                      x: clamp(
                        position.x,
                        displayValidRect.x1,
                        displayValidRect.x2 - width
                      ),
                      y: clamp(
                        position.y,
                        displayValidRect.y1,
                        displayValidRect.y2 - height
                      )
                    };
                  }}
                  onDragEnd={() => commitSelectedTransform(selectedBoxIndex!)}
                  onTransformEnd={() => commitSelectedTransform(selectedBoxIndex!)}
                />
                <Transformer
                  ref={transformer}
                  rotateEnabled={false}
                  flipEnabled={false}
                  enabledAnchors={[
                    "top-left",
                    "top-center",
                    "top-right",
                    "middle-left",
                    "middle-right",
                    "bottom-left",
                    "bottom-center",
                    "bottom-right"
                  ]}
                  anchorSize={9}
                  borderStroke="#3143df"
                  anchorStroke="#3143df"
                  anchorFill="#ffffff"
                  boundBoxFunc={(oldBox, nextBox) => {
                    const minimum = 32 * transform.scale;
                    if (nextBox.width < minimum || nextBox.height < minimum) {
                      return oldBox;
                    }
                    const width = Math.min(
                      nextBox.width,
                      displayValidRect.x2 - displayValidRect.x1
                    );
                    const height = Math.min(
                      nextBox.height,
                      displayValidRect.y2 - displayValidRect.y1
                    );
                    return {
                      ...nextBox,
                      x: clamp(
                        nextBox.x,
                        displayValidRect.x1,
                        displayValidRect.x2 - width
                      ),
                      y: clamp(
                        nextBox.y,
                        displayValidRect.y1,
                        displayValidRect.y2 - height
                      ),
                      width,
                      height
                    };
                  }}
                />
              </>
            ) : null}
            {draft ? (
              <Rect
                x={draft.x1}
                y={draft.y1}
                width={draft.x2 - draft.x1}
                height={draft.y2 - draft.y1}
                stroke="#5f6ff5"
                strokeWidth={2}
                dash={[5, 4]}
                listening={false}
              />
            ) : null}
          </Layer>
        </Stage>
        <div className="roi-legend">
          <span><i className="legend-valid" />有效区域</span>
          <span><i className="legend-invalid" />无效区域</span>
          <span>{image.width} × {image.height} px</span>
          {rawUrl ? (
            <a href={rawUrl} download={image.filename}>
              下载原图
            </a>
          ) : null}
        </div>
      </div>

      <div className="roi-side">
        <div className="roi-side-heading">
          <div>
            <strong>选择与精调</strong>
            <span>先从右侧选中，再在画布拖动或缩放</span>
          </div>
          <Button
            size="sm"
            onClick={() => {
              setDirty(true);
              setSelectedBoxIndex(boxes.length);
              setBoxes((current) => [
                ...current,
                {
                  box_id: null,
                  label: `ROI ${current.length + 1}`,
                  active: true,
                  x1: validRect.x1,
                  y1: validRect.y1,
                  x2: Math.min(validRect.x1 + 128, validRect.x2),
                  y2: Math.min(validRect.y1 + 128, validRect.y2)
                }
              ]);
            }}
            disabled={boxes.length >= 20}
          >
            <Plus size={14} />添加
          </Button>
        </div>

        <div className="roi-box-list">
          {boxes.map((box, index) => (
            <article
              className={[
                "roi-box",
                errors[index] ? "invalid" : "",
                selectedBoxIndex === index ? "selected" : ""
              ].filter(Boolean).join(" ")}
              key={index}
            >
              <button
                type="button"
                className="roi-adjust-select"
                aria-pressed={selectedBoxIndex === index}
                onClick={() =>
                  setSelectedBoxIndex((current) => (current === index ? null : index))
                }
              >
                <MousePointer2 size={14} />
                {selectedBoxIndex === index ? "正在画布中调整" : "在画布中调整此 ROI"}
              </button>
              <div className="roi-box-title">
                <input
                  className="input"
                  aria-label={`ROI ${index + 1} 标签`}
                  value={box.label}
                  maxLength={120}
                  onChange={(event) => updateBox(index, { label: event.target.value })}
                />
                <label className="roi-active-toggle">
                  <input
                    type="checkbox"
                    checked={box.active}
                    onChange={(event) =>
                      updateBox(index, { active: event.target.checked })
                    }
                  />
                  <span>启用</span>
                </label>
                <button
                  type="button"
                  aria-label={`删除 ROI ${index + 1}`}
                  onClick={() => {
                    setDirty(true);
                    setBoxes((current) =>
                      current.filter((_, boxIndex) => boxIndex !== index)
                    );
                    setSelectedBoxIndex((current) => {
                      if (current === null) return null;
                      if (current === index) return null;
                      return current > index ? current - 1 : current;
                    });
                  }}
                >
                  <Trash2 size={15} />
                </button>
              </div>
              <div className="roi-coordinates">
                {(["x1", "y1", "x2", "y2"] as const).map((key) => (
                  <label key={key}>
                    <span>{key}</span>
                    <input
                      className="input"
                      type="number"
                      min="0"
                      value={box[key]}
                      onChange={(event) => updateBox(index, { [key]: Number(event.target.value) })}
                    />
                  </label>
                ))}
              </div>
              {errors[index] ? <p>{errors[index]}</p> : null}
            </article>
          ))}
          {!boxes.length ? (
            <p className="roi-empty">在画布拖拽，或使用“添加”创建第一个 ROI。</p>
          ) : null}
        </div>

        {conflict ? (
          <div className="roi-conflict" role="alert">
            <strong>ROI 已被其他会话更新</strong>
            <p>当前未保存坐标仍保留，不会自动覆盖服务端 revision。</p>
            <div className="inline-actions">
              <Button
                size="sm"
                onClick={() => reload.mutate()}
                disabled={reload.isPending}
              >
                <RefreshCcw size={14} />
                {reload.isPending ? "重新加载中…" : "重新加载服务端"}
              </Button>
              <Button
                size="sm"
                tone="ghost"
                onClick={() => void navigator.clipboard.writeText(JSON.stringify(boxes, null, 2))}
              >
                <Clipboard size={14} />复制当前坐标
              </Button>
            </div>
          </div>
        ) : null}

        {save.isError && !conflict ? <RequestError error={save.error} /> : null}
        {reload.isError ? <RequestError error={reload.error} /> : null}

        <div className="roi-save">
          <span>
            将从 revision {baseRevision} 更新 · {boxes.length}/20 个框
          </span>
          {writeBlocker ? (
            <span className="form-warning" role="status">
              {writeBlocker}
            </span>
          ) : null}
          <Button
            tone="primary"
            onClick={() => save.mutate()}
            disabled={Boolean(writeBlocker) || save.isPending || errors.some(Boolean)}
            title={writeBlocker || undefined}
          >
            <Save size={15} />
            {save.isPending ? "保存中…" : "保存全部 ROI"}
          </Button>
        </div>
      </div>
    </div>
  );
}
