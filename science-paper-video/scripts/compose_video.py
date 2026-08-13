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
            cues.append(
                {
                    "start": round(cursor + float(cue["start"]), 3),
                    "end": round(cursor + float(cue["end"]), 3),
                    "text": cue["text"],
                }
            )
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


def to_ass(cues: list[dict], width: int, height: int) -> str:
    font_size = max(38, round(height * 0.036))
    margin = round(height * 0.038)
    header = [
        "[Script Info]",
        "ScriptType: v4.00+",
        f"PlayResX: {width}",
        f"PlayResY: {height}",
        "ScaledBorderAndShadow: yes",
        "WrapStyle: 2",
        "",
        "[V4+ Styles]",
        "Format: Name,Fontname,Fontsize,PrimaryColour,SecondaryColour,OutlineColour,BackColour,Bold,Italic,Underline,StrikeOut,ScaleX,ScaleY,Spacing,Angle,BorderStyle,Outline,Shadow,Alignment,MarginL,MarginR,MarginV,Encoding",
        f"Style: Default,Microsoft YaHei,{font_size},&H00FFFFFF,&H000000FF,&H00101010,&H8A000000,0,0,0,0,100,100,0,0,1,5,2,2,{margin},{margin},{margin},1",
        "",
        "[Events]",
        "Format: Layer,Start,End,Style,Name,MarginL,MarginR,MarginV,Effect,Text",
    ]
    for cue in cues:
        text = str(cue["text"]).replace("{", r"\{").replace("}", r"\}").replace("\n", r"\N")
        header.append(
            f"Dialogue: 0,{_ass_time(cue['start'])},{_ass_time(cue['end'])},Default,,0,0,0,,{text}"
        )
    return "\n".join(header) + "\n"


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
    ass.write_text(to_ass(cues, width, height), encoding="utf-8")
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
