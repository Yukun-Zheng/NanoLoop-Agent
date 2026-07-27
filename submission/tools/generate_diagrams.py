#!/usr/bin/env python3
"""Generate submission diagrams and video cards with no external network assets."""

from __future__ import annotations

import math
from itertools import pairwise
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter, ImageFont

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "submission" / "assets" / "diagrams"
OUT.mkdir(parents=True, exist_ok=True)

INK = "#141620"
MUTED = "#697386"
BLUE = "#596EF7"
BLUE_DARK = "#3348D8"
BLUE_PALE = "#EDF0FF"
CYAN = "#36B8CF"
GREEN = "#11875D"
GREEN_PALE = "#E8F7F1"
AMBER = "#B86800"
AMBER_PALE = "#FFF4E3"
LINE = "#DDE2EF"
WHITE = "#FFFFFF"


def font_path() -> str:
    candidates = [
        Path(
            "/System/Library/AssetsV2/com_apple_MobileAsset_Font7/"
            "3419f2a427639ad8c8e139149a287865a90fa17e.asset/"
            "AssetData/PingFang.ttc"
        ),
        Path("/System/Library/Fonts/PingFang.ttc"),
        Path("/System/Library/Fonts/STHeiti Light.ttc"),
        Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"),
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    raise FileNotFoundError("No suitable CJK font found")


FONT_PATH = font_path()


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    # PingFang is a variable TTC; layout remains stable even when the requested face
    # falls back to the collection's default. Stroke is used sparingly for headings.
    return ImageFont.truetype(FONT_PATH, size=size, index=0)


def gradient(size: tuple[int, int], left: str, right: str) -> Image.Image:
    canvas = Image.new("RGB", size, left)
    draw = ImageDraw.Draw(canvas)
    lrgb = tuple(int(left[i : i + 2], 16) for i in (1, 3, 5))
    rrgb = tuple(int(right[i : i + 2], 16) for i in (1, 3, 5))
    for x in range(size[0]):
        ratio = x / max(1, size[0] - 1)
        color = tuple(round(a + (b - a) * ratio) for a, b in zip(lrgb, rrgb, strict=True))
        draw.line((x, 0, x, size[1]), fill=color)
    return canvas


def wrap(
    draw: ImageDraw.ImageDraw, text: str, text_font: ImageFont.FreeTypeFont, width: int
) -> list[str]:
    lines: list[str] = []
    current = ""
    for char in text:
        candidate = current + char
        if draw.textbbox((0, 0), candidate, font=text_font)[2] <= width:
            current = candidate
        else:
            if current:
                lines.append(current)
            current = char
    if current:
        lines.append(current)
    return lines


def text_block(
    draw: ImageDraw.ImageDraw,
    xy: tuple[int, int],
    text: str,
    *,
    size: int,
    color: str,
    max_width: int,
    line_gap: int = 8,
    anchor: str | None = None,
) -> int:
    text_font = font(size)
    lines = wrap(draw, text, text_font, max_width)
    x, y = xy
    for line in lines:
        draw.text((x, y), line, font=text_font, fill=color, anchor=anchor)
        y += size + line_gap
    return y


def shadow_card(
    canvas: Image.Image,
    box: tuple[int, int, int, int],
    *,
    fill: str = WHITE,
    outline: str = LINE,
    radius: int = 26,
) -> ImageDraw.ImageDraw:
    shadow = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
    shadow_draw = ImageDraw.Draw(shadow)
    sx1, sy1, sx2, sy2 = box
    shadow_draw.rounded_rectangle(
        (sx1 + 5, sy1 + 14, sx2 + 5, sy2 + 14),
        radius=radius,
        fill=(51, 72, 130, 38),
    )
    shadow = shadow.filter(ImageFilter.GaussianBlur(18))
    canvas.paste(shadow, (0, 0), shadow)
    draw = ImageDraw.Draw(canvas)
    draw.rounded_rectangle(box, radius=radius, fill=fill, outline=outline, width=2)
    return draw


def arrow(
    draw: ImageDraw.ImageDraw,
    start: tuple[int, int],
    end: tuple[int, int],
    *,
    color: str = BLUE,
    width: int = 6,
) -> None:
    draw.line((*start, *end), fill=color, width=width)
    angle = math.atan2(end[1] - start[1], end[0] - start[0])
    length = 18
    spread = math.pi / 7
    points = [
        end,
        (
            int(end[0] - length * math.cos(angle - spread)),
            int(end[1] - length * math.sin(angle - spread)),
        ),
        (
            int(end[0] - length * math.cos(angle + spread)),
            int(end[1] - length * math.sin(angle + spread)),
        ),
    ]
    draw.polygon(points, fill=color)


def heading(draw: ImageDraw.ImageDraw, title: str, subtitle: str, *, width: int) -> None:
    draw.text((90, 62), title, font=font(52), fill=INK, stroke_width=1, stroke_fill=INK)
    draw.text((90, 128), subtitle, font=font(24), fill=MUTED)
    draw.rounded_rectangle((90, 174, width - 90, 179), radius=2, fill=BLUE)


def make_workflow() -> None:
    canvas = gradient((1800, 980), "#F8FAFF", "#F3F0FF")
    draw = ImageDraw.Draw(canvas)
    heading(draw, "从原始图像到实验判断", "七步闭环，每一步都留下可复核的输入与证据", width=1800)
    cards = [
        ("01", "批量上传", "1–20 张 TIF / PNG / JPG\n样品信息可后补", BLUE_PALE),
        ("02", "识别 SEM 信息", "比例尺 · 仪器字段\n定位有效成像区", GREEN_PALE),
        ("03", "确认 ROI", "全图可直接跳过\n多框按 revision 保存", AMBER_PALE),
        ("04", "逐图选模型", "推荐可解释\n人工确认后冻结", BLUE_PALE),
        ("05", "不可变运行", "原图 · 模型 · 参数\n设备与代码摘要", GREEN_PALE),
        ("06", "统计与质量", "canonical 实例\n物理量纲与 Gate", AMBER_PALE),
        ("07", "复核与导出", "叠加 · 比较 · 助手\nCSV / DOCX / PDF / ZIP", BLUE_PALE),
    ]
    positions = [
        (80, 245, 440, 505),
        (510, 245, 870, 505),
        (940, 245, 1300, 505),
        (1370, 245, 1730, 505),
        (295, 635, 655, 895),
        (725, 635, 1085, 895),
        (1155, 635, 1515, 895),
    ]
    for (number, title, detail, pale), box in zip(cards, positions, strict=True):
        card_draw = shadow_card(canvas, box, fill=WHITE)
        x1, y1, x2, _ = box
        card_draw.rounded_rectangle((x1 + 24, y1 + 24, x1 + 96, y1 + 62), radius=19, fill=pale)
        card_draw.text((x1 + 60, y1 + 43), number, font=font(19), fill=BLUE_DARK, anchor="mm")
        card_draw.text(
            (x1 + 28, y1 + 92), title, font=font(31), fill=INK, stroke_width=1, stroke_fill=INK
        )
        line_y = y1 + 145
        for line in detail.split("\n"):
            card_draw.text((x1 + 28, line_y), line, font=font(22), fill=MUTED)
            line_y += 39
        card_draw.rounded_rectangle((x1 + 28, y1 + 218, x2 - 28, y1 + 224), radius=3, fill=pale)
    for left, right in pairwise(positions[:4]):
        arrow(
            draw,
            (left[2] + 12, (left[1] + left[3]) // 2),
            (right[0] - 12, (right[1] + right[3]) // 2),
        )
    arrow(
        draw,
        (positions[3][2] - 80, positions[3][3] + 20),
        (positions[4][0] + 110, positions[4][1] - 18),
    )
    arrow(
        draw,
        (positions[4][2] + 12, (positions[4][1] + positions[4][3]) // 2),
        (positions[5][0] - 12, (positions[5][1] + positions[5][3]) // 2),
    )
    arrow(
        draw,
        (positions[5][2] + 12, (positions[5][1] + positions[5][3]) // 2),
        (positions[6][0] - 12, (positions[6][1] + positions[6][3]) // 2),
    )
    canvas.save(OUT / "workflow.png", quality=95)


def make_architecture() -> None:
    canvas = gradient((1800, 1080), "#FBFCFF", "#F1F4FF")
    draw = ImageDraw.Draw(canvas)
    heading(
        draw,
        "NanoLoop 科学工作台架构",
        "前端不重算科学结果；模型、统计与助手通过清晰边界协作",
        width=1800,
    )

    layers = [
        (
            "交互层",
            "Next.js Command Center",
            "上传 · ROI · 模型选择 · 结果复核 · 报告",
            BLUE_PALE,
            BLUE,
        ),
        ("服务层", "FastAPI 与契约", "任务编排 · 鉴权边界 · 队列 · 导出 · 审计", GREEN_PALE, GREEN),
        (
            "计算层",
            "模型网关 + 确定性分析",
            "内容寻址 bundle · Adapter · canonical 实例 · 形貌统计 · Gate",
            AMBER_PALE,
            AMBER,
        ),
        (
            "事实层",
            "SQLite WAL + 制品存储",
            "任务 · ROI revision · 不可变运行 · 状态事件 · SHA-256",
            "#F1EDFF",
            "#7153C8",
        ),
    ]
    top = 245
    for index, (label, title, detail, pale, accent) in enumerate(layers):
        y1 = top + index * 185
        y2 = y1 + 135
        card_draw = shadow_card(canvas, (80, y1, 1230, y2), fill=WHITE)
        card_draw.rounded_rectangle((80, y1, 105, y2), radius=13, fill=accent)
        card_draw.rounded_rectangle((130, y1 + 28, 300, y1 + 66), radius=19, fill=pale)
        card_draw.text((215, y1 + 47), label, font=font(20), fill=accent, anchor="mm")
        card_draw.text(
            (330, y1 + 35), title, font=font(31), fill=INK, stroke_width=1, stroke_fill=INK
        )
        card_draw.text((330, y1 + 83), detail, font=font(21), fill=MUTED)
        if index < len(layers) - 1:
            arrow(draw, (655, y2 + 6), (655, y2 + 44), color=BLUE_DARK, width=5)

    side_draw = shadow_card(canvas, (1320, 245, 1725, 875), fill=WHITE)
    side_draw.rounded_rectangle((1360, 280, 1685, 334), radius=27, fill=BLUE_PALE)
    side_draw.text((1522, 307), "科研助手", font=font(27), fill=BLUE_DARK, anchor="mm")
    items = [
        ("通用对话", "本地 Qwen / 兼容模型"),
        ("实验工具", "只读取确定性数据"),
        ("材料知识", "本地受管文档与引用"),
        ("在线研究", "Crossref / 可选网页"),
        ("安全回退", "超时、引用、单位均校验"),
    ]
    y = 375
    for title, detail in items:
        side_draw.ellipse((1365, y + 6, 1383, y + 24), fill=BLUE)
        side_draw.text((1402, y), title, font=font(23), fill=INK)
        side_draw.text((1402, y + 38), detail, font=font(18), fill=MUTED)
        y += 96
    side_draw.rounded_rectangle((1355, 790, 1690, 837), radius=20, fill=GREEN_PALE)
    side_draw.text((1522, 813), "大模型不计算实验数字", font=font(19), fill=GREEN, anchor="mm")
    arrow(draw, (1238, 492), (1305, 492), color=CYAN)
    arrow(draw, (1305, 675), (1238, 675), color=CYAN)
    draw.text(
        (90, 1015),
        "默认本地回环网络 · 非 root 容器 · 只读根文件系统 · 健康检查 · Docker Compose",
        font=font(21),
        fill=MUTED,
    )
    canvas.save(OUT / "architecture.png", quality=95)


def make_provenance() -> None:
    canvas = gradient((1800, 880), "#FAFBFF", "#F2F5FF")
    draw = ImageDraw.Draw(canvas)
    heading(
        draw,
        "一次运行，保存一条不可改写的证据链",
        "调整 ROI、模型或阈值会创建新运行，旧结果继续可审查",
        width=1800,
    )
    nodes = [
        ("原始输入", "图像 SHA-256\n尺寸 · 元数据 · 有效区", BLUE_PALE, BLUE_DARK),
        ("科学配置", "ROI revision\n模型 bundle · 阈值 · 后处理", GREEN_PALE, GREEN),
        ("实际执行", "设备 · seed · Adapter\n源码与依赖摘要", AMBER_PALE, AMBER),
        ("可信输出", "canonical 实例\n质量 · 统计 · 制品摘要", "#F1EDFF", "#7153C8"),
    ]
    boxes = [
        (70, 280, 420, 555),
        (500, 280, 850, 555),
        (930, 280, 1280, 555),
        (1360, 280, 1710, 555),
    ]
    for (title, detail, pale, accent), box in zip(nodes, boxes, strict=True):
        card_draw = shadow_card(canvas, box, fill=WHITE)
        x1, y1, x2, _ = box
        card_draw.rounded_rectangle((x1 + 26, y1 + 28, x2 - 26, y1 + 84), radius=27, fill=pale)
        card_draw.text(((x1 + x2) // 2, y1 + 56), title, font=font(26), fill=accent, anchor="mm")
        line_y = y1 + 120
        for line in detail.split("\n"):
            card_draw.text(((x1 + x2) // 2, line_y), line, font=font(21), fill=MUTED, anchor="ma")
            line_y += 47
        card_draw.text(
            ((x1 + x2) // 2, y1 + 238), "✓ 已冻结", font=font(18), fill=GREEN, anchor="mm"
        )
    for left, right in pairwise(boxes):
        arrow(draw, (left[2] + 12, 417), (right[0] - 12, 417))
    draw.rounded_rectangle((250, 675, 1550, 795), radius=36, fill=INK)
    draw.text(
        (900, 712),
        "相同快照 → 相同内容地址；科学输入变化 → 新运行、新导出",
        font=font(27),
        fill=WHITE,
        anchor="ma",
    )
    draw.text(
        (900, 757), "截图只是显示，运行记录才是事实", font=font(20), fill="#C8D0FF", anchor="ma"
    )
    canvas.save(OUT / "provenance.png", quality=95)


def make_video_cards() -> None:
    for name, title, subtitle in [
        (
            "video-title.png",
            "从显微图像到实验洞察",
            "NanoLoop · 可追溯 SEM 颗粒分析智能体",
        ),
        (
            "video-end.png",
            "让每一个结果都能回到实验现场",
            "纳米颗粒图像识别工具开发小组",
        ),
    ]:
        canvas = gradient((1920, 1080), "#EEF3FF", "#F5ECFF")
        draw = ImageDraw.Draw(canvas)
        for cx, cy, radius, color in [
            (1750, 160, 220, "#DDE6FF"),
            (140, 980, 300, "#E7DAFF"),
            (1580, 940, 160, "#DDF6F2"),
        ]:
            draw.ellipse((cx - radius, cy - radius, cx + radius, cy + radius), fill=color)
        draw.rounded_rectangle(
            (170, 160, 1750, 920), radius=52, fill=WHITE, outline="#D9DFF2", width=3
        )
        draw.rounded_rectangle((230, 230, 590, 286), radius=28, fill=BLUE_PALE)
        draw.text((410, 258), "NANOLOOP · AI4S", font=font(24), fill=BLUE_DARK, anchor="mm")
        draw.text(
            (960, 430), title, font=font(70), fill=INK, anchor="mm", stroke_width=1, stroke_fill=INK
        )
        draw.text((960, 540), subtitle, font=font(34), fill=MUTED, anchor="mm")
        draw.rounded_rectangle((590, 665, 1330, 671), radius=3, fill=BLUE)
        draw.text(
            (960, 780),
            "C 赛道 · AI4S 湿闭环（Wet Lab）",
            font=font(27),
            fill=BLUE_DARK,
            anchor="mm",
        )
        canvas.save(OUT / name, quality=95)


def main() -> None:
    make_workflow()
    make_architecture()
    make_provenance()
    make_video_cards()
    for path in sorted(OUT.glob("*.png")):
        print(f"{path.relative_to(ROOT)}\t{path.stat().st_size} bytes")


if __name__ == "__main__":
    main()
