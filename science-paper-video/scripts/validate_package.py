"""Validate a science-paper-video project, alignment, and optional rendered MP4."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path


SENTENCE_END = re.compile(r".+?(?:[\u3002\uff01\uff1f!?]+|$)", re.DOTALL)
DEFAULT_PRODUCERS = ("science-paper-narrator",)


def validate_project_contract(
    project: dict,
    base: Path,
    require_files: bool = False,
    allowed_producers: tuple[str, ...] = DEFAULT_PRODUCERS,
) -> None:
    if project.get("schema_version") != 1:
        raise ValueError("Unsupported video-project schema_version.")
    source = project.get("source_bundle", {})
    producer = source.get("producer")
    if producer not in allowed_producers:
        raise ValueError(
            f"source_bundle producer {producer!r} is not allowed; expected one of "
            f"{list(allowed_producers)}. For an already reviewed in-house report, set the producer "
            "(e.g. 'reviewed-report') and pass --allow-producer explicitly."
        )
    if source.get("figures_verified") is not True:
        raise ValueError("Verified high-resolution figure crops are required.")
    canvas = project.get("canvas", {})
    if min(int(canvas.get("width", 0)), int(canvas.get("height", 0))) <= 0:
        raise ValueError("Canvas dimensions must be positive.")
    scenes = project.get("scenes") or []
    if not scenes:
        raise ValueError("At least one scene is required.")
    for scene in scenes:
        for field in ("id", "narration_display", "tts_text", "audio_file"):
            if not str(scene.get(field, "")).strip():
                raise ValueError(f"Scene is missing {field}.")
        frames = scene.get("frames") or []
        if not frames or any(float(frame.get("fraction", 0)) <= 0 for frame in frames):
            raise ValueError(f"Scene {scene['id']} needs positive frame fractions.")
        if require_files:
            for relative in [scene["audio_file"], *(frame["image"] for frame in frames)]:
                if not (base / relative).is_file():
                    raise ValueError(f"Missing media input: {relative}")


def validate_alignment(project: dict, aligned: dict) -> dict:
    by_id = {item["scene_id"]: item for item in aligned.get("segments", [])}
    cue_count = 0
    for scene in project["scenes"]:
        item = by_id.get(scene["id"])
        if item is None:
            raise ValueError(f"Missing aligned scene {scene['id']}.")
        if item.get("display_text") != scene["narration_display"]:
            raise ValueError(f"Aligned text differs from reviewed narration for {scene['id']}.")
        cues = item.get("cues") or []
        if "".join(cue["text"] for cue in cues) != scene["narration_display"]:
            raise ValueError(f"Subtitle cues differ from reviewed narration for {scene['id']}.")
        expected_units = SENTENCE_END.findall(scene["narration_display"])
        if [cue["text"] for cue in cues] != expected_units:
            raise ValueError(f"Subtitle cues do not follow reviewed sentence boundaries for {scene['id']}.")
        previous_end = -1.0
        for cue in cues:
            start, end = float(cue["start"]), float(cue["end"])
            if start < previous_end or end <= start:
                raise ValueError(f"Invalid subtitle timing for {scene['id']}.")
            if end > float(item["duration_seconds"]) + 0.25:
                raise ValueError(f"Subtitle exceeds audio for {scene['id']}.")
            lines = cue.get("lines") or [cue["text"]]
            if len(lines) > 2 or "".join(lines) != cue["text"]:
                raise ValueError(f"Invalid visual subtitle lines for {scene['id']}.")
            previous_end = end
            cue_count += 1
    return {"scenes": len(project["scenes"]), "subtitle_cues": cue_count}


def probe_video(path: Path) -> dict:
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration,size:stream=codec_type,codec_name,width,height,sample_rate,channels", "-of", "json", str(path)],
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(result.stdout)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("project", type=Path)
    parser.add_argument("alignment", type=Path)
    parser.add_argument("--video", type=Path)
    parser.add_argument("--report", type=Path)
    parser.add_argument(
        "--allow-producer", action="append", default=[], metavar="NAME",
        help="Also accept this source_bundle.producer (repeatable), e.g. reviewed-report.",
    )
    args = parser.parse_args()
    allowed = (*DEFAULT_PRODUCERS, *args.allow_producer)
    project = json.loads(args.project.read_text(encoding="utf-8-sig"))
    aligned = json.loads(args.alignment.read_text(encoding="utf-8-sig"))
    validate_project_contract(project, args.project.parent, require_files=True, allowed_producers=allowed)
    report = validate_alignment(project, aligned)
    report["source_producer"] = project["source_bundle"]["producer"]
    report["timing_policy"] = aligned.get("timing_policy")
    if args.video:
        report["media"] = probe_video(args.video)
    output = json.dumps(report, ensure_ascii=False, indent=2)
    if args.report:
        args.report.write_text(output, encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
