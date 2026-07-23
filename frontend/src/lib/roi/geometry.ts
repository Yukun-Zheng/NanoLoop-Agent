export type PixelRect = {
  x1: number;
  y1: number;
  x2: number;
  y2: number;
};

export type CanvasTransform = {
  scale: number;
  offsetX: number;
  offsetY: number;
};

export function canvasTransform(
  canvasWidth: number,
  canvasHeight: number,
  imageWidth: number,
  imageHeight: number
): CanvasTransform {
  const scale = Math.min(canvasWidth / imageWidth, canvasHeight / imageHeight);
  return {
    scale,
    offsetX: (canvasWidth - imageWidth * scale) / 2,
    offsetY: (canvasHeight - imageHeight * scale) / 2
  };
}

export function displayToOriginal(
  rect: PixelRect,
  transform: CanvasTransform,
  imageWidth: number,
  imageHeight: number
): PixelRect {
  return {
    x1: clamp(Math.floor((rect.x1 - transform.offsetX) / transform.scale), 0, imageWidth),
    y1: clamp(Math.floor((rect.y1 - transform.offsetY) / transform.scale), 0, imageHeight),
    x2: clamp(Math.ceil((rect.x2 - transform.offsetX) / transform.scale), 0, imageWidth),
    y2: clamp(Math.ceil((rect.y2 - transform.offsetY) / transform.scale), 0, imageHeight)
  };
}

export function originalToDisplay(rect: PixelRect, transform: CanvasTransform): PixelRect {
  return {
    x1: rect.x1 * transform.scale + transform.offsetX,
    y1: rect.y1 * transform.scale + transform.offsetY,
    x2: rect.x2 * transform.scale + transform.offsetX,
    y2: rect.y2 * transform.scale + transform.offsetY
  };
}

export function rectIntersects(a: PixelRect, b: PixelRect): boolean {
  return a.x1 < b.x2 && a.x2 > b.x1 && a.y1 < b.y2 && a.y2 > b.y1;
}

export function validateRoiRect(
  rect: PixelRect,
  validRect: PixelRect,
  invalidRects: PixelRect[],
  minEdge = 32
): string | null {
  if (rect.x2 - rect.x1 < minEdge || rect.y2 - rect.y1 < minEdge) {
    return `每条边至少需要 ${minEdge} px`;
  }
  if (
    rect.x1 < validRect.x1 ||
    rect.y1 < validRect.y1 ||
    rect.x2 > validRect.x2 ||
    rect.y2 > validRect.y2
  ) {
    return "选框必须完全位于有效分析区域内";
  }
  if (invalidRects.some((candidate) => rectIntersects(rect, candidate))) {
    return "选框与仪器栏或其他无效区域相交";
  }
  return null;
}

function clamp(value: number, minimum: number, maximum: number) {
  return Math.max(minimum, Math.min(maximum, value));
}
