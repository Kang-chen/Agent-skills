"""Time subtitles from per-sentence TTS clips instead of forced alignment.

Default timing path (2026-11-11). Forced alignment with FunASR fa-zh compressed
Doubao TTS 2.0 audio by about 9%, so this script derives every cue from the
real length of its own sentence clip:

1. ``plan``: split each scene's ``tts_text`` and ``narration_display`` into
   reviewed sentences and write ``sentences.jsonl`` (one TTS request per line,
   id ``<scene_id>_<NN>``).
2. Synthesize each line to ``<sentence-dir>/<id>.wav`` with any TTS engine.
3. ``build``: concatenate the clips per scene (fixed pause between sentences),
   write each scene's ``audio_file``, and write ``alignment.json`` in the same
   structure as ``align_subtitles.py``.
"""

from __future__ import annotations

import argparse
import json
import sys
import wave
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from align_subtitles import (
    read_mono_samples,
    speech_bounds,
    split_caption_units,
    split_display_lines,
)

TIMING_POLICY = "per-sentence-tts-audio"


def sentence_id(scene_id: str, index: int) -> str:
    return f"{scene_id}_{index:02d}"


def scene_sentences(scene: dict) -> list[tuple[str, str]]:
    """Pair reviewed display sentences with their TTS sentences."""

    display = split_caption_units(scene["narration_display"])
    spoken = split_caption_units(scene["tts_text"])
    if len(display) != len(spoken):
        raise ValueError(
            f"{scene['id']}: narration_display has {len(display)} sentences but tts_text has "
            f"{len(spoken)}; keep sentence-ending punctuation identical in both fields."
        )
    return list(zip(display, spoken))


def plan_sentences(project: dict) -> list[dict]:
    rows = []
    for scene in project["scenes"]:
        for index, (display, spoken) in enumerate(scene_sentences(scene), 1):
            rows.append(
                {
                    "id": sentence_id(scene["id"], index),
                    "scene_id": scene["id"],
                    "index": index,
                    "tts_text": spoken,
                    "display_text": display,
                }
            )
    return rows


def read_pcm16(path: Path) -> tuple[bytes, int]:
    with wave.open(str(path), "rb") as handle:
        if handle.getsampwidth() != 2 or handle.getnchannels() != 1:
            raise ValueError(
                f"{path} must be 16-bit mono PCM WAV; convert with "
                "`ffmpeg -i in.wav -ac 1 -c:a pcm_s16le out.wav`."
            )
        return handle.readframes(handle.getnframes()), handle.getframerate()


def build_scene(
    scene: dict,
    sentence_dir: Path,
    pause: float,
    max_chars: int,
) -> tuple[bytes, int, dict]:
    """Concatenate one scene's sentence clips and time each cue from its clip."""

    pcm_all, cues, rate, cursor_frames = b"", [], None, 0
    for index, (display, _) in enumerate(scene_sentences(scene), 1):
        clip = sentence_dir / f"{sentence_id(scene['id'], index)}.wav"
        if not clip.is_file():
            raise FileNotFoundError(f"Missing sentence audio: {clip}")
        pcm, clip_rate = read_pcm16(clip)
        if rate is None:
            rate = clip_rate
        elif clip_rate != rate:
            raise ValueError(f"{scene['id']}: sentence clips use different sample rates.")
        if index > 1:
            # 句间插入固定静音，按采样点计数避免累计舍入误差
            pause_frames = round(pause * rate)
            pcm_all += b"\x00\x00" * pause_frames
            cursor_frames += pause_frames
        frames = len(pcm) // 2
        if frames == 0:
            raise ValueError(f"Empty sentence audio: {clip}")
        samples, _ = read_mono_samples(clip)
        voice_start, voice_end = speech_bounds(samples, rate)
        start = cursor_frames / rate
        cues.append(
            {
                "start": round(start, 3),
                "end": round((cursor_frames + frames) / rate, 3),
                "speech_start": round(start + voice_start, 3),
                "speech_end": round(start + voice_end, 3),
                "text": display,
                "lines": split_display_lines(display, max_chars=max_chars),
                "timing_source": "per-sentence TTS audio length",
            }
        )
        pcm_all += pcm
        cursor_frames += frames
    if "".join(cue["text"] for cue in cues) != scene["narration_display"]:
        raise AssertionError("Sentence split changed reviewed narration.")
    segment = {
        "scene_id": scene["id"],
        "display_text": scene["narration_display"],
        "tts_text": scene["tts_text"],
        "audio_file": scene["audio_file"],
        "duration_seconds": round(cursor_frames / rate, 3),
        "cues": cues,
    }
    return pcm_all, rate, segment


def write_wav(path: Path, pcm: bytes, rate: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes(pcm)


def build_timeline(
    project: dict,
    base: Path,
    sentence_dir: Path,
    pause: float = 0.3,
    max_chars: int = 45,
) -> dict:
    segments = []
    for scene in project["scenes"]:
        pcm, rate, segment = build_scene(scene, sentence_dir, pause, max_chars)
        write_wav(base / scene["audio_file"], pcm, rate)
        segments.append(segment)
    return {
        "schema_version": 2,
        "source_of_truth": "scenes[].narration_display",
        "alignment_model": None,
        "timing_policy": TIMING_POLICY,
        "sentence_pause_seconds": pause,
        "segments": segments,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    plan = sub.add_parser("plan", help="Write one TTS request per reviewed sentence.")
    plan.add_argument("project", type=Path)
    plan.add_argument("--output", type=Path, required=True)
    build = sub.add_parser("build", help="Concatenate sentence clips and write alignment JSON.")
    build.add_argument("project", type=Path)
    build.add_argument("--sentence-dir", type=Path, required=True)
    build.add_argument("--output", type=Path, required=True)
    build.add_argument("--pause", type=float, default=0.3, help="Silence between sentences, seconds.")
    build.add_argument("--max-chars", type=int, default=45, help="Maximum visual width per subtitle line.")
    args = parser.parse_args()

    project = json.loads(args.project.read_text(encoding="utf-8-sig"))
    if args.command == "plan":
        rows = plan_sentences(project)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8"
        )
        print(json.dumps({"sentences": len(rows), "output": str(args.output)}, ensure_ascii=False))
        return
    payload = build_timeline(
        project, args.project.parent, args.sentence_dir, pause=args.pause, max_chars=args.max_chars
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    cue_count = sum(len(item["cues"]) for item in payload["segments"])
    print(json.dumps({"scenes": len(payload["segments"]), "subtitle_cues": cue_count}, ensure_ascii=False))


if __name__ == "__main__":
    main()
