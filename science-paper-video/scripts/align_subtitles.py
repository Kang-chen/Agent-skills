"""Align reviewed subtitle text to final per-scene audio with FunASR fa-zh."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Sequence


SENTENCE_END = re.compile(r".+?(?:[\u3002\uff01\uff1f!?]+|$)", re.DOTALL)
ALIGN_TOKEN = re.compile(r"[A-Za-z0-9]+|[\u3400-\u9fff]")
PREFERRED_LINE_BREAK = set("\uff0c\uff1b\uff1a\u3001,;: ")


def tokenize_for_alignment(text: str) -> list[str]:
    return ALIGN_TOKEN.findall(re.sub(r"\s+", "", text))


def split_caption_units(text: str, max_chars: int = 45) -> list[str]:
    """Return complete sentences; max_chars controls layout, not cue timing."""

    del max_chars
    units = SENTENCE_END.findall(text)
    if not units or "".join(units) != text:
        raise ValueError("Unable to preserve reviewed sentence boundaries.")
    return units


def _visual_width(text: str) -> float:
    width = 0.0
    for char in text:
        if char.isspace():
            width += 0.35
        elif char.isascii():
            width += 0.56
        else:
            width += 1.0
    return width


def split_display_lines(text: str, max_chars: int = 45) -> list[str]:
    """Wrap one timed sentence into two lines, preferring semantic punctuation."""

    total_width = _visual_width(text)
    if total_width <= max_chars:
        return [text]
    if total_width > max_chars * 2:
        raise ValueError(
            "One sentence exceeds two subtitle lines; revise the reviewed narration "
            "instead of creating mid-sentence timed cues."
        )

    preferred: list[tuple[float, int]] = []
    fallback: list[tuple[float, int]] = []
    target = total_width / 2
    for index in range(1, len(text)):
        if (
            text[index - 1].isascii()
            and text[index - 1].isalnum()
            and text[index].isascii()
            and text[index].isalnum()
        ):
            continue
        left = _visual_width(text[:index])
        right = _visual_width(text[index:])
        if left > max_chars or right > max_chars:
            continue
        candidate = (abs(left - target), index)
        if text[index - 1] in PREFERRED_LINE_BREAK:
            preferred.append(candidate)
        else:
            fallback.append(candidate)
    pool = preferred or fallback
    if not pool:
        raise ValueError("No safe two-line layout exists for a complete sentence.")
    _, split_at = min(pool)
    lines = [text[:split_at], text[split_at:]]
    if "".join(lines) != text:
        raise AssertionError("Visual wrapping changed reviewed narration.")
    return lines


def _alignment_atoms(text: str) -> list[str]:
    atoms: list[str] = []
    for token in tokenize_for_alignment(text):
        atoms.extend(char.casefold() for char in token)
    return atoms


def _sentence_token_spans(
    spoken_units: Sequence[str],
    aligned_tokens: Sequence[str],
) -> list[tuple[int, int]]:
    expected = _alignment_atoms("".join(spoken_units))
    observed_groups = [_alignment_atoms(str(token)) for token in aligned_tokens]
    observed = [atom for group in observed_groups for atom in group]
    if expected != observed:
        mismatch = next(
            (index for index, pair in enumerate(zip(expected, observed)) if pair[0] != pair[1]),
            min(len(expected), len(observed)),
        )
        raise ValueError(
            "Forced-alignment token content differs from tts_text at atom "
            f"{mismatch}; refusing proportional timing fallback."
        )

    token_end_by_atom: dict[int, int] = {}
    atom_cursor = 0
    for token_index, group in enumerate(observed_groups, 1):
        if not group:
            raise ValueError("Forced alignment returned an empty token.")
        atom_cursor += len(group)
        token_end_by_atom[atom_cursor] = token_index

    spans: list[tuple[int, int]] = []
    token_cursor = atom_cursor = 0
    for unit in spoken_units:
        atom_cursor += len(_alignment_atoms(unit))
        token_end = token_end_by_atom.get(atom_cursor)
        if token_end is None:
            raise ValueError("A reviewed sentence boundary falls inside one forced-alignment token.")
        spans.append((token_cursor, token_end))
        token_cursor = token_end
    return spans


def build_cues(
    display_text: str,
    spoken_text: str,
    aligned_tokens: Sequence[str],
    timestamps_ms: Sequence[Sequence[int]],
    max_chars: int = 45,
    audio_duration: float | None = None,
    lead_seconds: float = 0.08,
    tail_seconds: float = 0.18,
) -> list[dict]:
    """Build sentence cues from atom-verified token spans without cumulative shifts."""

    if len(aligned_tokens) != len(timestamps_ms):
        raise ValueError("FunASR token and timestamp counts differ.")
    display_units = split_caption_units(display_text)
    spoken_units = split_caption_units(spoken_text)
    if len(display_units) != len(spoken_units):
        raise ValueError(
            "Display and TTS texts have different sentence boundaries; provide an explicit "
            "pronunciation-only tts_text that preserves punctuation."
        )
    spans = _sentence_token_spans(spoken_units, aligned_tokens)
    speech_spans: list[tuple[float, float]] = []
    for begin, end in spans:
        start = int(timestamps_ms[begin][0]) / 1000.0
        finish = int(timestamps_ms[end - 1][1]) / 1000.0
        if finish <= start:
            raise ValueError("Forced-alignment timestamps are not positive.")
        speech_spans.append((start, finish))

    limit = float(audio_duration) if audio_duration is not None else speech_spans[-1][1] + tail_seconds
    starts = [max(0.0, start - lead_seconds) for start, _ in speech_spans]
    ends = [min(limit, finish + tail_seconds) for _, finish in speech_spans]
    for index in range(len(speech_spans) - 1):
        if ends[index] > starts[index + 1]:
            boundary = (speech_spans[index][1] + speech_spans[index + 1][0]) / 2
            ends[index] = boundary
            starts[index + 1] = boundary

    cues: list[dict] = []
    for index, (text, (begin, end), speech) in enumerate(zip(display_units, spans, speech_spans)):
        start = max(0.0, starts[index])
        finish = min(limit, ends[index])
        if finish <= start:
            raise ValueError("A sentence received no positive subtitle duration.")
        cues.append(
            {
                "start": round(start, 3),
                "end": round(finish, 3),
                "speech_start": round(speech[0], 3),
                "speech_end": round(speech[1], 3),
                "text": text,
                "lines": split_display_lines(text, max_chars=max_chars),
                "token_start": begin,
                "token_end": end,
            }
        )
    if "".join(cue["text"] for cue in cues) != display_text:
        raise AssertionError("Caption segmentation changed reviewed narration.")
    return cues


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("project", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default="fa-zh")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-chars", type=int, default=45, help="Maximum visual width per subtitle line.")
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
        duration = float(soundfile.info(str(audio)).duration)
        cues = build_cues(
            scene["narration_display"],
            scene["tts_text"],
            tokens,
            result["timestamp"],
            args.max_chars,
            audio_duration=duration,
        )
        segments.append(
            {
                "scene_id": scene["id"],
                "display_text": scene["narration_display"],
                "tts_text": scene["tts_text"],
                "audio_file": scene["audio_file"],
                "duration_seconds": round(duration, 3),
                "cues": cues,
            }
        )
    payload = {
        "schema_version": 2,
        "source_of_truth": "scenes[].narration_display",
        "alignment_model": args.model,
        "timing_policy": "exact-token-sentence-boundaries",
        "segments": segments,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
if __name__ == "__main__":
    main()
