"""Render a figure walkthrough from storyboard.json and retime frames from real sentence audio.

A storyboard (see references/storyboard.md) lists scenes; each scene covers a
number of reviewed sentences and is split into frames. A frame highlights parts
of one figure (named region, pixel box, grid cells or full rows), may dim the
rest, may swap in another image (formula, schematic, crop), and shows cards in
the right-hand panel.

Subcommands:

- ``render``: draw every frame PNG and write ``video-project.json``. Each frame
  records how many sentences it covers; ``fraction`` is a placeholder until retime.
- ``retime``: read the alignment JSON from ``sentence_timeline.py build`` and set
  each frame's ``fraction`` to the real time of the sentences it covers, so the
  highlight changes when the matching sentence starts.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parent))
from align_subtitles import split_caption_units  # noqa: E402

ACCENT = (230, 97, 0)  # 高亮框与当前卡片的默认强调色（橙）
TEXT_DARK = (34, 34, 34)
TEXT_MUTED = (150, 150, 150)
CARD_BG = (253, 240, 228)
RULE_GREY = (210, 210, 210)
REFERENCE_WIDTH, REFERENCE_HEIGHT = 2560, 1440
DIM_STRENGTH = 0.68  # 非焦点区域向白色混合的比例

# 常见系统里的 CJK 字体：(常规, 粗体, TTC 内简体中文索引)
FONT_CANDIDATES: tuple[tuple[str, str, int], ...] = (
    ("/usr/share/fonts/google-noto-cjk/NotoSansCJK-Regular.ttc",
     "/usr/share/fonts/google-noto-cjk/NotoSansCJK-Bold.ttc", 2),
    ("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
     "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc", 2),
    ("/usr/share/fonts/noto-cjk/NotoSansCJK-Regular.ttc",
     "/usr/share/fonts/noto-cjk/NotoSansCJK-Bold.ttc", 2),
    ("/System/Library/Fonts/PingFang.ttc", "/System/Library/Fonts/PingFang.ttc", 0),
    ("C:/Windows/Fonts/msyh.ttc", "C:/Windows/Fonts/msyhbd.ttc", 0),
)
FONTCONFIG_FAMILY = "Noto Sans CJK SC"


@dataclass(frozen=True)
class FontSpec:
    """A font file plus the face index inside a TTC collection."""

    path: Path
    index: int = 0


@dataclass(frozen=True)
class Fonts:
    regular: FontSpec
    bold: FontSpec

    def get(self, size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
        spec = self.bold if bold else self.regular
        return ImageFont.truetype(str(spec.path), size, index=spec.index)


def _fontconfig_match(pattern: str) -> FontSpec | None:
    if shutil.which("fc-match") is None:
        return None
    result = subprocess.run(
        ["fc-match", "-f", "%{file}|%{index}", pattern], capture_output=True, text=True, check=False
    )
    if result.returncode != 0 or "|" not in result.stdout:
        return None
    path, index = result.stdout.strip().rsplit("|", 1)
    return FontSpec(Path(path), int(index or 0)) if Path(path).is_file() else None


def find_fonts(regular: Path | None = None, bold: Path | None = None, index: int | None = None) -> Fonts:
    """Locate a CJK-capable font pair.

    Order: explicit arguments, ``SPV_FONT_REGULAR`` / ``SPV_FONT_BOLD`` /
    ``SPV_FONT_INDEX`` environment variables, a list of common install paths,
    then fontconfig (``fc-match``) for Noto Sans CJK SC.
    """

    regular = regular or (Path(os.environ["SPV_FONT_REGULAR"]) if os.environ.get("SPV_FONT_REGULAR") else None)
    bold = bold or (Path(os.environ["SPV_FONT_BOLD"]) if os.environ.get("SPV_FONT_BOLD") else None)
    if index is None and os.environ.get("SPV_FONT_INDEX"):
        index = int(os.environ["SPV_FONT_INDEX"])
    if regular is not None:
        if not regular.is_file():
            raise FileNotFoundError(f"Font not found: {regular}")
        bold = bold or regular
        return Fonts(FontSpec(regular, index or 0), FontSpec(bold, index or 0))
    for reg_path, bold_path, face in FONT_CANDIDATES:
        if Path(reg_path).is_file() and Path(bold_path).is_file():
            face = face if index is None else index
            return Fonts(FontSpec(Path(reg_path), face), FontSpec(Path(bold_path), face))
    found_regular = _fontconfig_match(FONTCONFIG_FAMILY)
    found_bold = _fontconfig_match(f"{FONTCONFIG_FAMILY}:style=Bold") or found_regular
    if found_regular is not None and found_bold is not None:
        return Fonts(found_regular, found_bold)
    raise FileNotFoundError(
        "No CJK font found; pass --font-regular/--font-bold or set SPV_FONT_REGULAR."
    )


@dataclass(frozen=True)
class Layout:
    """Title on top, figure on the left, cards on the right, bottom strip left free for subtitles."""

    width: int
    height: int
    scale: float
    title_box: tuple[int, int, int, int]
    figure_box: tuple[int, int, int, int]
    panel_box: tuple[int, int, int, int]

    @classmethod
    def for_canvas(cls, width: int, height: int) -> Layout:
        sx, sy = width / REFERENCE_WIDTH, height / REFERENCE_HEIGHT

        def box(x0: int, y0: int, x1: int, y1: int) -> tuple[int, int, int, int]:
            return round(x0 * sx), round(y0 * sy), round(x1 * sx), round(y1 * sy)

        return cls(
            width=width,
            height=height,
            scale=sy,
            title_box=box(60, 36, 2500, 140),
            figure_box=box(40, 165, 1540, 1250),
            panel_box=box(1600, 185, 2520, 1250),
        )

    def px(self, value: float) -> int:
        """Scale a size defined on the 2560x1440 reference canvas."""

        return max(1, round(value * self.scale))


def hex_to_rgb(value: str) -> tuple[int, int, int]:
    """Convert ``#RRGGBB`` to an RGB tuple."""

    value = value.lstrip("#")
    if len(value) != 6:
        raise ValueError(f"Expected #RRGGBB colour, got '{value}'")
    return int(value[0:2], 16), int(value[2:4], 16), int(value[4:6], 16)


def wrap_text(draw: ImageDraw.ImageDraw, text: str, fnt: ImageFont.FreeTypeFont, max_width: int) -> list[str]:
    """Wrap by pixel width without breaking runs of Latin letters, digits or symbols."""

    tokens = re.findall(r"[A-Za-z0-9.+\-–/%α]+|\s|.", text)
    lines: list[str] = []
    current = ""
    for token in tokens:
        trial = current + token
        if current and draw.textlength(trial, font=fnt) > max_width:
            lines.append(current.rstrip())
            current = token.lstrip()
        else:
            current = trial
    if current.strip():
        lines.append(current.rstrip())
    return lines


def resolve_box(spec: dict[str, Any], storyboard: dict[str, Any]) -> tuple[int, int, int, int]:
    """Turn one highlight spec into figure pixel coordinates ``(x0, y0, x1, y1)``."""

    regions = storyboard.get("figure_regions", {})
    if "region" in spec:
        x0, y0, x1, y1 = regions[spec["region"]]
        return int(x0), int(y0), int(x1), int(y1)
    if "box" in spec:
        x0, y0, x1, y1 = spec["box"]
        return int(x0), int(y0), int(x1), int(y1)
    grid = storyboard.get("grid")
    if not grid:
        raise KeyError(f"Highlight {spec} needs a 'grid' in the storyboard.")
    rows = grid["rows"]
    col_width = (grid["x1"] - grid["x0"]) / len(grid["markers"])
    if "cells" in spec:
        row_ids = spec["cells"]["rows"]
        cols = [grid["markers"].index(col) for col in spec["cells"]["cols"]]
        x0 = grid["x0"] + min(cols) * col_width
        x1 = grid["x0"] + (max(cols) + 1) * col_width
    elif "rows_full" in spec:
        row_ids = spec["rows_full"]
        # 整行从行标签左缘起；没有 row_labels 区域时从网格左缘起
        x0 = regions["row_labels"][0] if "row_labels" in regions else grid["x0"]
        x1 = regions["bars"][2] if spec.get("with_bars") else grid["x1"]
    else:
        raise KeyError(f"Unknown highlight spec: {spec}")
    y0 = min(rows[row][0] for row in row_ids)
    y1 = max(rows[row][1] for row in row_ids)
    return round(x0), int(y0), round(x1), int(y1) + 1


def place_image(img: Image.Image, box: tuple[int, int, int, int]) -> tuple[float, int, int]:
    """Fit ``img`` into ``box`` keeping aspect ratio; return (scale, left, top)."""

    box_w, box_h = box[2] - box[0], box[3] - box[1]
    scale = min(box_w / img.width, box_h / img.height)
    width, height = round(img.width * scale), round(img.height * scale)
    return scale, box[0] + (box_w - width) // 2, box[1] + (box_h - height) // 2


def draw_cards(
    canvas: Image.Image,
    layout: Layout,
    fonts: Fonts,
    cards: dict[str, dict[str, str]],
    shown: list[str],
    active: str | None,
) -> None:
    """Right-hand cards; the active card gets a tinted background and accent bar, the rest turn grey."""

    draw = ImageDraw.Draw(canvas)
    x0, y, x1, y_max = layout.panel_box
    head_font, body_font = fonts.get(layout.px(40), bold=True), fonts.get(layout.px(34))
    pad, bar, head_h, line_h, space = layout.px(22), layout.px(10), layout.px(52), layout.px(48), layout.px(18)
    for card_id in shown:
        card = cards[card_id]
        is_active = card_id == active
        body_lines = wrap_text(draw, card["body"], body_font, x1 - x0 - 2 * pad - layout.px(12))
        height = pad + head_h + len(body_lines) * line_h + pad
        if y + height > y_max:
            raise ValueError(f"Cards overflow the panel at card '{card_id}'; show fewer cards in this frame.")
        # 卡片可带 color（与公式变量同色）：标题始终用该色当图例，当前卡片的强调条也用该色
        card_colour = hex_to_rgb(card["color"]) if card.get("color") else None
        accent = card_colour or ACCENT
        if is_active:
            draw.rounded_rectangle((x0, y, x1, y + height), radius=layout.px(14), fill=CARD_BG)
            draw.rectangle((x0, y, x0 + bar, y + height), fill=accent)
        head_colour = accent if (is_active or card_colour) else TEXT_MUTED
        body_colour = TEXT_DARK if is_active else TEXT_MUTED
        text_x = x0 + pad + layout.px(12)
        draw.text((text_x, y + pad - layout.px(4)), card["head"], font=head_font, fill=head_colour)
        for i, line in enumerate(body_lines):
            draw.text((text_x, y + pad + head_h + i * line_h), line, font=body_font, fill=body_colour)
        y += height + space


def frame_figure(storyboard: dict[str, Any], base: Path, frame: dict[str, Any]) -> tuple[Path, bool]:
    """Pick the image for one frame; return (path, whether highlight boxes apply)."""

    if frame.get("image"):
        return base / frame["image"], False
    key = "next_figure" if frame.get("next_figure") else "figure"
    return base / storyboard["source"][key], key == "figure"


def render_frame(
    storyboard: dict[str, Any],
    base: Path,
    layout: Layout,
    fonts: Fonts,
    scene: dict[str, Any],
    frame: dict[str, Any],
) -> Image.Image:
    canvas = Image.new("RGB", (layout.width, layout.height), "white")
    draw = ImageDraw.Draw(canvas)
    tx0, ty0, tx1, ty1 = layout.title_box
    draw.text((tx0, ty0), scene["title"], font=fonts.get(layout.px(58), bold=True), fill=TEXT_DARK)
    draw.line((tx0, ty1, tx1, ty1), fill=RULE_GREY, width=layout.px(3))

    figure_path, boxes_apply = frame_figure(storyboard, base, frame)
    figure = Image.open(figure_path).convert("RGB")
    scale, fx, fy = place_image(figure, layout.figure_box)
    scaled = figure.resize((round(figure.width * scale), round(figure.height * scale)), Image.LANCZOS)

    boxes = [resolve_box(spec, storyboard) for spec in frame.get("boxes", [])] if boxes_apply else []
    if frame.get("dim", False):
        # 调暗非焦点区域，再把焦点区域按原样贴回，让视线落在正在讲的地方；没有框时整张图调暗
        dimmed = Image.blend(scaled, Image.new("RGB", scaled.size, "white"), DIM_STRENGTH)
        for x0, y0, x1, y1 in boxes:
            crop = (round(x0 * scale), round(y0 * scale), round(x1 * scale), round(y1 * scale))
            dimmed.paste(scaled.crop(crop), crop[:2])
        scaled = dimmed
    canvas.paste(scaled, (fx, fy))
    pad, width = layout.px(5), layout.px(6)
    for x0, y0, x1, y1 in boxes:
        draw.rectangle(
            (fx + x0 * scale - pad, fy + y0 * scale - pad, fx + x1 * scale + pad, fy + y1 * scale + pad),
            outline=ACCENT,
            width=width,
        )
    draw_cards(canvas, layout, fonts, storyboard.get("cards", {}), frame.get("cards", []), frame.get("active"))
    return canvas


def check_sentence_counts(scene: dict[str, Any]) -> int:
    """Display, TTS and frame sentence counts must agree; return the count."""

    n_display = len(split_caption_units(scene["narration_display"]))
    n_tts = len(split_caption_units(scene["tts_text"]))
    n_frames = sum(int(frame["sentences"]) for frame in scene["frames"])
    if not n_display == n_tts == n_frames:
        raise ValueError(
            f"{scene['id']}: sentence counts differ (display={n_display}, tts={n_tts}, frames={n_frames})."
        )
    return n_display


def render_storyboard(storyboard_path: Path, out_dir: Path, fonts: Fonts | None = None) -> dict[str, Any]:
    """Render all frames into ``out_dir/frames`` and write ``out_dir/video-project.json``."""

    storyboard = json.loads(storyboard_path.read_text(encoding="utf-8"))
    base = storyboard_path.parent
    canvas_cfg = storyboard["canvas"]
    layout = Layout.for_canvas(int(canvas_cfg["width"]), int(canvas_cfg["height"]))
    fonts = fonts or find_fonts()
    frames_dir = out_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    scenes_out = []
    for scene in storyboard["scenes"]:
        check_sentence_counts(scene)
        frames_out = []
        for i, frame in enumerate(scene["frames"], 1):
            path = frames_dir / f"{scene['id']}_{i:02d}.png"
            render_frame(storyboard, base, layout, fonts, scene, frame).save(path)
            sentences = int(frame["sentences"])
            frames_out.append({"image": path.relative_to(out_dir).as_posix(), "fraction": sentences,
                               "sentences": sentences})
        scenes_out.append({
            "id": scene["id"],
            "slide_summary": scene["title"],
            "narration_display": scene["narration_display"],
            "tts_text": scene["tts_text"],
            "audio_file": f"audio/{scene['id']}.wav",
            "frames": frames_out,
        })

    source = storyboard.get("source", {})
    project = {
        "schema_version": 1,
        "source_bundle": {
            "producer": source.get("producer", "reviewed-report"),
            "article": source.get("report"),
            "figures_verified": True,
            "storyboard": storyboard_path.name,
        },
        "canvas": canvas_cfg,
        "scenes": scenes_out,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "video-project.json").write_text(json.dumps(project, ensure_ascii=False, indent=2), encoding="utf-8")
    return project


def frame_fractions(frame_sentences: list[int], cues: list[dict[str, Any]], duration: float, gap: float) -> list[float]:
    """Frame durations from real cue times.

    A frame switches when the first sentence of the next frame starts; the
    scene's tail silence and the inter-scene gap go to the last frame.
    """

    if sum(frame_sentences) != len(cues):
        raise ValueError("Frame sentence counts do not match aligned cues.")
    bounds, cursor = [0.0], 0
    for count in frame_sentences[:-1]:
        cursor += count
        bounds.append(float(cues[cursor]["start"]))
    bounds.append(duration + gap)
    fractions = [round(b - a, 3) for a, b in zip(bounds[:-1], bounds[1:])]
    if any(value <= 0 for value in fractions):
        raise ValueError("Non-positive frame duration after retiming; a frame must cover at least one sentence.")
    return fractions


def retime_project(project: dict[str, Any], alignment: dict[str, Any], gap: float) -> dict[str, Any]:
    """Set every frame ``fraction`` from the aligned cues; ``gap`` must match compose_video."""

    segments = {item["scene_id"]: item for item in alignment["segments"]}
    for scene in project["scenes"]:
        segment = segments[scene["id"]]
        counts = [int(frame.get("sentences", 0)) for frame in scene["frames"]]
        if any(count <= 0 for count in counts):
            raise ValueError(f"{scene['id']}: every frame needs a positive 'sentences' count to retime.")
        fractions = frame_fractions(counts, segment["cues"], float(segment["duration_seconds"]), gap)
        for frame, value in zip(scene["frames"], fractions):
            frame["fraction"] = value
    return project


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    render = sub.add_parser("render", help="Render frames and write video-project.json.")
    render.add_argument("storyboard", type=Path)
    render.add_argument("--out-dir", type=Path, required=True)
    render.add_argument("--font-regular", type=Path, help="CJK font file for body text.")
    render.add_argument("--font-bold", type=Path, help="CJK font file for titles and card heads.")
    render.add_argument("--font-index", type=int, help="Face index inside a TTC collection.")
    retime = sub.add_parser("retime", help="Set frame fractions from per-sentence cue times.")
    retime.add_argument("project", type=Path)
    retime.add_argument("alignment", type=Path)
    retime.add_argument("--gap", type=float, default=0.7, help="Must equal the gap passed to compose_video.py.")
    args = parser.parse_args()

    if args.command == "render":
        fonts = find_fonts(args.font_regular, args.font_bold, args.font_index)
        project = render_storyboard(args.storyboard.resolve(), args.out_dir.resolve(), fonts)
        frames = sum(len(scene["frames"]) for scene in project["scenes"])
        print(json.dumps({"scenes": len(project["scenes"]), "frames": frames}))
        return
    project = json.loads(args.project.read_text(encoding="utf-8"))
    alignment = json.loads(args.alignment.read_text(encoding="utf-8"))
    retime_project(project, alignment, args.gap)
    args.project.write_text(json.dumps(project, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"retimed_scenes": len(project["scenes"])}))


if __name__ == "__main__":
    main()
