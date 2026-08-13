"""Align reviewed subtitle text to final per-scene audio with FunASR fa-zh."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Sequence


CLAUSE_END = re.compile(r".+?[，。！？；：]|.+$", re.DOTALL)
ALIGN_TOKEN = re.compile(r"[A-Za-z0-9]+|[\u3400-\u9fff]")


def tokenize_for_alignment(text: str) -> list[str]:
    return ALIGN_TOKEN.findall(re.sub(r"\s+", "", text))


def split_caption_units(text: str, max_chars: int = 24) -> list[str]:
    units: list[str] = []
    for clause in CLAUSE_END.findall(text):
        cursor = 0
        while cursor < len(clause):
            units.append(clause[cursor : cursor + max_chars])
            cursor += max_chars
    merged: list[str] = []
    for unit in units:
        if merged and len(merged[-1]) < 8 and len(merged[-1]) + len(unit) <= max_chars:
            merged[-1] += unit
        else:
            merged.append(unit)
    return merged


def _boundaries(weights: Sequence[int], count: int) -> list[tuple[int, int]]:
    if not weights or count <= 0:
        raise ValueError("Alignment requires text and timestamps.")
    total = max(1, sum(weights))
    result: list[tuple[int, int]] = []
    cursor = cumulative = 0
    for index, weight in enumerate(weights):
        cumulative += weight
        if index == len(weights) - 1:
            end = count
        else:
            end = round(cumulative / total * count)
            end = max(cursor + 1, min(end, count - (len(weights) - index - 1)))
        result.append((cursor, end))
        cursor = end
    return result


def build_cues(
    display_text: str,
    spoken_text: str,
    aligned_tokens: Sequence[str],
    timestamps_ms: Sequence[Sequence[int]],
    max_chars: int = 24,
) -> list[dict]:
    if len(aligned_tokens) != len(timestamps_ms):
        raise ValueError("FunASR token and timestamp counts differ.")
    display_units = split_caption_units(display_text, max_chars)
    spoken_units = split_caption_units(spoken_text, max_chars)
    if len(display_units) != len(spoken_units):
        display_units, spoken_units = [display_text], [spoken_text]
    weights = [max(1, len(tokenize_for_alignment(unit))) for unit in spoken_units]
    raw: list[dict] = []
    for text, (begin, end) in zip(display_units, _boundaries(weights, len(timestamps_ms))):
        raw.append(
            {
                "start": max(0.0, int(timestamps_ms[begin][0]) / 1000 - 0.06),
                "end": int(timestamps_ms[end - 1][1]) / 1000 + 0.10,
                "text": text,
            }
        )
    cues: list[dict] = []
    for index, cue in enumerate(raw):
        start = max(cue["start"], cues[-1]["end"] + 0.01 if cues else 0.0)
        end = cue["end"]
        if index + 1 < len(raw):
            end = min(end, max(start + 0.20, raw[index + 1]["start"] - 0.01))
        cues.append({"start": round(start, 3), "end": round(max(end, start + 0.20), 3), "text": cue["text"]})
    if "".join(cue["text"] for cue in cues) != display_text:
        raise AssertionError("Caption segmentation changed reviewed narration.")
    return cues


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("project", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default="fa-zh")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-chars", type=int, default=24)
    args = parser.parse_args()
    project = json.loads(args.project.read_text(encoding="utf-8-sig"))
    base = args.project.parent

    from funasr import AutoModel
    import soundfile

    model = AutoModel(model=args.model, device=args.device, disable_update=True)
    segments = []
    for scene in project["scenes"]:
        audio = (base / scene["audio_file"]).resolve()
        result = model.generate(
            input=(str(audio), scene["tts_text"]), data_type=("sound", "text")
        )[0]
        tokens = str(result["text"]).split()
        cues = build_cues(
            scene["narration_display"],
            scene["tts_text"],
            tokens,
            result["timestamp"],
            args.max_chars,
        )
        segments.append(
            {
                "scene_id": scene["id"],
                "display_text": scene["narration_display"],
                "tts_text": scene["tts_text"],
                "audio_file": scene["audio_file"],
                "duration_seconds": round(float(soundfile.info(str(audio)).duration), 3),
                "cues": cues,
            }
        )
    payload = {
        "schema_version": 1,
        "source_of_truth": "scenes[].narration_display",
        "alignment_model": args.model,
        "segments": segments,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
