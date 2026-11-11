"""Compose verified frames, per-scene audio, and aligned cues into a video package."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path


def build_global_timeline(project: dict, aligned: dict, gap: float = 0.7) -> tuple[list[dict], float]:
    by_id = {item["scene_id"]: item for item in aligned["segments"]}
    cursor = 0.0
    cues: list[dict] = []
    for scene in project["scenes"]:
        item = by_id[scene["id"]]
        for cue in item["cues"]:
            global_cue = {
                "start": round(cursor + float(cue["start"]), 3),
                "end": round(cursor + float(cue["end"]), 3),
                "text": cue["text"],
            }
            if "lines" in cue:
                global_cue["lines"] = list(cue["lines"])
            cues.append(global_cue)
        cursor += float(item["duration_seconds"]) + gap
    return cues, round(cursor, 3)


def _srt_time(seconds: float) -> str:
    millis = round(seconds * 1000)
    hours, millis = divmod(millis, 3_600_000)
    minutes, millis = divmod(millis, 60_000)
    secs, millis = divmod(millis, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def _ass_time(seconds: float) -> str:
    centis = round(seconds * 100)
    hours, centis = divmod(centis, 360_000)
    minutes, centis = divmod(centis, 6_000)
    secs, centis = divmod(centis, 100)
    return f"{hours}:{minutes:02d}:{secs:02d}.{centis:02d}"


def to_srt(cues: list[dict]) -> str:
    blocks = []
    for index, cue in enumerate(cues, 1):
        blocks.append(
            f"{index}\n{_srt_time(cue['start'])} --> {_srt_time(cue['end'])}\n{cue['text']}"
        )
    return "\n\n".join(blocks) + "\n"


DEFAULT_SUBTITLE_FONT = "Noto Sans CJK SC"
SUBTITLE_STYLES = ("outline", "box")
# 2026-11-11 在 2560x1440 实测可读的透明描边样式；其他分辨率按高度等比缩放
_REFERENCE_HEIGHT = 1440
_OUTLINE_FONT_SIZE = 64
_OUTLINE_COMPACT_SIZE = 60
_OUTLINE_WIDTH = 2.5
_OUTLINE_SHADOW = 1.0


def _scaled(value: float, height: int) -> float:
    return value * height / _REFERENCE_HEIGHT


def subtitle_style_line(
    height: int,
    font: str = DEFAULT_SUBTITLE_FONT,
    font_size: int | None = None,
    style: str = "outline",
) -> str:
    """Return the ASS Default style: transparent outline (default) or translucent box."""

    if style not in SUBTITLE_STYLES:
        raise ValueError(f"Unknown subtitle style {style!r}; choose one of {SUBTITLE_STYLES}.")
    margin = round(height * 0.038)
    if style == "outline":
        # 白字粗体，纯黑描边，半透明黑阴影，BorderStyle=1 无底框；描边过粗会糊字
        size = font_size or round(_scaled(_OUTLINE_FONT_SIZE, height))
        outline = round(_scaled(_OUTLINE_WIDTH, height), 2)
        shadow = round(_scaled(_OUTLINE_SHADOW, height), 2)
        colours, bold, border = "&H00FFFFFF,&H000000FF,&H00000000,&H96000000", -1, f"1,{outline:g},{shadow:g}"
    else:
        size = font_size or max(38, round(height * 0.036))
        colours, bold, border = "&H00FFFFFF,&H000000FF,&H00101010,&H8A000000", 0, "3,6,0"
    return (
        f"Style: Default,{font},{size},{colours},{bold},0,0,0,100,100,0,0,{border},2,"
        f"{margin},{margin},{margin},1"
    )


def default_compact_size(height: int, style: str = "outline") -> int:
    if style == "outline":
        return round(_scaled(_OUTLINE_COMPACT_SIZE, height))
    return 74 if height >= 2160 else max(36, round(height * 0.034))


def to_ass(
    cues: list[dict],
    width: int,
    height: int,
    font: str = DEFAULT_SUBTITLE_FONT,
    font_size: int | None = None,
    compact_size: int | None = None,
    style: str = "outline",
) -> str:
    compact = compact_size or default_compact_size(height, style)
    output = [
        "[Script Info]",
        "ScriptType: v4.00+",
        f"PlayResX: {width}",
        f"PlayResY: {height}",
        "ScaledBorderAndShadow: yes",
        "WrapStyle: 2",
        "",
        "[V4+ Styles]",
        "Format: Name,Fontname,Fontsize,PrimaryColour,SecondaryColour,OutlineColour,BackColour,Bold,Italic,Underline,StrikeOut,ScaleX,ScaleY,Spacing,Angle,BorderStyle,Outline,Shadow,Alignment,MarginL,MarginR,MarginV,Encoding",
        subtitle_style_line(height, font=font, font_size=font_size, style=style),
        "",
        "[Events]",
        "Format: Layer,Start,End,Style,Name,MarginL,MarginR,MarginV,Effect,Text",
    ]
    for cue in cues:
        lines = cue.get("lines") or [cue["text"]]
        if len(lines) > 2 or "".join(lines) != cue["text"]:
            raise ValueError("Subtitle visual lines must preserve cue text in at most two lines.")
        escaped = [
            str(line).replace("{", r"\{").replace("}", r"\}").replace("\n", r"\N")
            for line in lines
        ]
        prefix = rf"{{\fs{compact}}}" if len(lines) == 2 else ""
        text = prefix + r"\N".join(escaped)
        output.append(
            f"Dialogue: 0,{_ass_time(cue['start'])},{_ass_time(cue['end'])},Default,,0,0,0,,{text}"
        )
    return "\n".join(output) + "\n"


def run(command: list[str], cwd: Path | None = None) -> None:
    subprocess.run(command, cwd=cwd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("project", type=Path)
    parser.add_argument("alignment", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--name", default="paper-video")
    parser.add_argument("--gap", type=float, default=0.7)
    parser.add_argument("--crf", type=int, default=18)
    parser.add_argument("--preset", default="medium")
    parser.add_argument("--subtitle-font", default=DEFAULT_SUBTITLE_FONT)
    parser.add_argument("--subtitle-size", type=int, help="One-line font size; default scales 64 px at 1440p.")
    parser.add_argument("--subtitle-compact-size", type=int, help="Two-line font size; default scales 60 px at 1440p.")
    parser.add_argument(
        "--subtitle-style", choices=SUBTITLE_STYLES, default="outline",
        help="outline: transparent background with outline and shadow (default); box: translucent box.",
    )
    args = parser.parse_args()

    project = json.loads(args.project.read_text(encoding="utf-8-sig"))
    aligned = json.loads(args.alignment.read_text(encoding="utf-8-sig"))
    base = args.project.parent.resolve()
    output = args.output_dir.resolve()
    work = output / f".{args.name}-work"
    clips, audio_parts = work / "clips", work / "audio"
    clips.mkdir(parents=True, exist_ok=True)
    audio_parts.mkdir(parents=True, exist_ok=True)
    output.mkdir(parents=True, exist_ok=True)

    width = int(project["canvas"]["width"])
    height = int(project["canvas"]["height"])
    fps = int(project["canvas"].get("fps", 30))
    by_id = {item["scene_id"]: item for item in aligned["segments"]}
    clip_manifest: list[str] = []
    audio_manifest: list[str] = []
    frame_index = 0
    timeline = []

    for scene in project["scenes"]:
        item = by_id[scene["id"]]
        scene_duration = float(item["duration_seconds"]) + args.gap
        fractions = sum(float(frame["fraction"]) for frame in scene["frames"])
        for frame in scene["frames"]:
            frame_index += 1
            duration = scene_duration * float(frame["fraction"]) / fractions
            clip = clips / f"frame-{frame_index:04d}.mp4"
            run(
                [
                    "ffmpeg", "-y", "-loglevel", "error", "-loop", "1", "-i", str((base / frame["image"]).resolve()),
                    "-t", f"{duration:.3f}", "-r", str(fps), "-vf", f"scale={width}:{height}:flags=lanczos,format=yuv420p",
                    "-c:v", "libx264", "-preset", args.preset, "-tune", "stillimage", "-crf", str(args.crf), "-an", str(clip),
                ]
            )
            clip_manifest.append(f"file '{clip.as_posix()}'")
        padded = audio_parts / f"{scene['id']}.wav"
        run(
            [
                "ffmpeg", "-y", "-loglevel", "error", "-i", str((base / scene["audio_file"]).resolve()),
                "-af", f"apad=pad_dur={args.gap}", "-t", f"{scene_duration:.3f}", "-ar", "48000", "-ac", "1", str(padded),
            ]
        )
        audio_manifest.append(f"file '{padded.as_posix()}'")
        timeline.append({"scene_id": scene["id"], "duration_seconds": round(scene_duration, 3)})

    (work / "clips.txt").write_text("\n".join(clip_manifest), encoding="utf-8")
    (work / "audio.txt").write_text("\n".join(audio_manifest), encoding="utf-8")
    silent = work / "silent.mp4"
    narration = output / f"{args.name}-narration.wav"
    run(["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", str(work / "clips.txt"), "-c", "copy", str(silent)])
    run(["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", str(work / "audio.txt"), "-af", "loudnorm=I=-16:LRA=11:TP=-1.5", str(narration)])

    cues, total = build_global_timeline(project, aligned, args.gap)
    srt = output / f"{args.name}.srt"
    ass = output / f"{args.name}.ass"
    srt.write_text(to_srt(cues), encoding="utf-8")
    ass.write_text(
        to_ass(
            cues, width, height, font=args.subtitle_font, font_size=args.subtitle_size,
            compact_size=args.subtitle_compact_size, style=args.subtitle_style,
        ),
        encoding="utf-8",
    )
    final = output / f"{args.name}.mp4"
    run(
        [
            "ffmpeg", "-y", "-loglevel", "error", "-i", str(silent), "-i", str(narration),
            "-vf", f"ass={ass.name}", "-map", "0:v:0", "-map", "1:a:0", "-shortest",
            "-c:v", "libx264", "-preset", args.preset, "-crf", str(args.crf), "-c:a", "aac", "-b:a", "192k", "-ar", "48000", "-movflags", "+faststart", final.name,
        ],
        cwd=output,
    )
    (output / f"{args.name}-timeline.json").write_text(
        json.dumps({"duration_seconds": total, "scenes": timeline}, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({"video": str(final), "duration_seconds": total, "subtitle_cues": len(cues)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
