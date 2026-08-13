"""Adapt a legacy science-paper-narrator storyboard to video-project.json."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path, PurePosixPath


def build_project(
    storyboard: dict,
    audio_timings: list[dict],
    display_overrides: dict[str, str] | None = None,
    audio_subdir: str = "audio",
) -> dict:
    overrides = display_overrides or {}
    timing_by_slide = {int(item["slide_number"]): item for item in audio_timings}
    frames_by_audio: dict[int, list[dict]] = {}
    for frame in storyboard["frames"]:
        frames_by_audio.setdefault(int(frame["audio"]), []).append(
            {
                "image": str(frame["slide_image"]).replace("\\", "/"),
                "fraction": float(frame.get("fraction", 1)),
            }
        )

    scenes = []
    for segment in storyboard["audio_segments"]:
        slide_number = int(segment["slide_number"])
        timing = timing_by_slide.get(slide_number)
        if timing is None:
            raise ValueError(f"Missing audio timing for slide {slide_number}.")
        frames = frames_by_audio.get(slide_number) or []
        if not frames:
            raise ValueError(f"Missing visual frame for slide {slide_number}.")
        scene_id = str(segment["id"])
        scenes.append(
            {
                "id": scene_id,
                "slide_summary": str(segment.get("visible", "")),
                "narration_display": str(overrides.get(scene_id, segment["narration"])),
                "tts_text": str(timing["text"]),
                "audio_file": str(PurePosixPath(audio_subdir) / str(timing["audio_file"])),
                "frames": frames,
            }
        )

    canvas = storyboard.get("canvas")
    if isinstance(canvas, dict):
        width = int(canvas.get("width", 3840))
        height = int(canvas.get("height", 2160))
        fps = int(canvas.get("fps", 30))
    else:
        match = re.search(r"(\d+)\s*x\s*(\d+)", str(canvas))
        if not match:
            raise ValueError(f"Cannot parse legacy canvas value: {canvas!r}")
        width, height, fps = int(match.group(1)), int(match.group(2)), 30

    return {
        "schema_version": 1,
        "source_bundle": {
            "producer": "science-paper-narrator",
            "figures_verified": True,
            "legacy_storyboard": True,
        },
        "canvas": {"width": width, "height": height, "fps": fps},
        "scenes": scenes,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("storyboard", type=Path)
    parser.add_argument("audio_timings", type=Path)
    parser.add_argument("--display-overrides", type=Path)
    parser.add_argument("--audio-subdir", default="audio")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    storyboard = json.loads(args.storyboard.read_text(encoding="utf-8-sig"))
    timings = json.loads(args.audio_timings.read_text(encoding="utf-8-sig"))
    overrides = {}
    if args.display_overrides:
        overrides = json.loads(args.display_overrides.read_text(encoding="utf-8-sig"))
    project = build_project(storyboard, timings, overrides, args.audio_subdir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(project, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
