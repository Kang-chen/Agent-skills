"""Align reviewed subtitle text to final per-scene audio with FunASR fa-zh."""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
import wave
from array import array
from pathlib import Path
from typing import Sequence


SENTENCE_END = re.compile(r".+?(?:[\u3002\uff01\uff1f!?]+|$)", re.DOTALL)
ALIGN_TOKEN = re.compile(r"[A-Za-z0-9]+|[\u3400-\u9fff]")
PREFERRED_LINE_BREAK = set("\uff0c\uff1b\uff1a\u3001,;: ")
# fa-zh 时间轴漂移的容忍上限：超过任一阈值即判定对齐失败
MAX_DRIFT_SECONDS = 0.5
MAX_DRIFT_RATIO = 0.03


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


def split_display_lines(text: str, max_chars: float = 45) -> list[str]:
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


def read_mono_samples(path: Path) -> tuple[list[float], int]:
    """Read audio as mono floats in [-1, 1]; soundfile if installed, else 16-bit mono WAV."""

    try:
        import soundfile
    except ImportError:
        soundfile = None
    if soundfile is not None:
        # 多声道取均值，与 FunASR 读到的单声道内容一致
        data, rate = soundfile.read(str(path), dtype="float32", always_2d=True)
        return data.mean(axis=1).tolist(), int(rate)
    with wave.open(str(path), "rb") as handle:
        if handle.getsampwidth() != 2 or handle.getnchannels() != 1:
            raise ValueError(
                f"{path} must be 16-bit mono PCM WAV; convert with "
                "`ffmpeg -i in.wav -ac 1 -c:a pcm_s16le out.wav`."
            )
        pcm = array("h")
        pcm.frombytes(handle.readframes(handle.getnframes()))
        rate = handle.getframerate()
    return [value / 32768.0 for value in pcm], rate


def speech_bounds(
    samples: Sequence[float],
    rate: int,
    threshold: float = 0.02,
    window_seconds: float = 0.02,
) -> tuple[float, float]:
    """Return first and last voiced time (seconds) by short-time RMS energy."""

    # 短时能量：窗口 RMS 不低于最大 RMS 的 threshold 倍即视为有声
    window = max(1, int(rate * window_seconds))
    count = len(samples) // window
    if count == 0:
        return 0.0, len(samples) / rate
    rms = [
        math.sqrt(sum(value * value for value in samples[i * window:(i + 1) * window]) / window)
        for i in range(count)
    ]
    peak = max(rms)
    voiced = [i for i, value in enumerate(rms) if peak > 0 and value >= threshold * peak]
    if not voiced:
        return 0.0, len(samples) / rate
    step = window / rate
    return round(voiced[0] * step, 3), round((voiced[-1] + 1) * step, 3)


def check_alignment_drift(
    last_token_end: float,
    energy_speech_end: float,
    audio_duration: float,
    max_seconds: float = MAX_DRIFT_SECONDS,
    max_ratio: float = MAX_DRIFT_RATIO,
) -> float:
    """Reject a forced alignment whose time axis disagrees with the audio energy.

    Measured on 2026-11-11: fa-zh compressed Doubao TTS 2.0 audio by about 9%
    (last token end at 0.90-0.93 of the audio length), so later subtitles led
    the voice. There is no safe proportional fix; use sentence_timeline.py.
    """

    drift = energy_speech_end - last_token_end
    if abs(drift) > max_seconds or abs(drift) > max_ratio * audio_duration:
        raise ValueError(
            f"Forced-alignment drift {drift:+.2f} s: last token ends at {last_token_end:.2f} s "
            f"but speech energy ends at {energy_speech_end:.2f} s (audio {audio_duration:.2f} s; "
            f"limits {max_seconds} s or {max_ratio:.0%}). Do not rescale by character count; "
            "synthesize one clip per sentence and run scripts/sentence_timeline.py instead."
        )
    return drift


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
    max_chars: float = 45,
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
    parser.add_argument(
        "--max-chars", type=float, default=None,
        help="Maximum visual width per subtitle line (CJK = 1, ASCII = 0.56); "
        "default derives from canvas width, subtitle font size and margins.",
    )
    parser.add_argument("--max-drift-seconds", type=float, default=MAX_DRIFT_SECONDS)
    parser.add_argument("--max-drift-ratio", type=float, default=MAX_DRIFT_RATIO)
    args = parser.parse_args()
    project = json.loads(args.project.read_text(encoding="utf-8-sig"))
    base = args.project.parent
    if args.max_chars is None:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from compose_video import max_line_units

        canvas = project["canvas"]
        args.max_chars = max_line_units(int(canvas["width"]), int(canvas["height"]))

    from funasr import AutoModel

    model = AutoModel(model=args.model, device=args.device, disable_update=True)
    segments = []
    for scene in project["scenes"]:
        audio = (base / scene["audio_file"]).resolve()
        result = model.generate(
            input=(str(audio), scene["tts_text"]), data_type=("sound", "text")
        )[0]
        tokens = str(result["text"]).split()
        samples, rate = read_mono_samples(audio)
        duration = len(samples) / rate
        # 漂移自检：fa-zh 末 token 结束时间须与能量检测的语音结束时间一致
        _, energy_end = speech_bounds(samples, rate)
        last_token_end = int(result["timestamp"][-1][1]) / 1000.0
        try:
            drift = check_alignment_drift(
                last_token_end, energy_end, duration, args.max_drift_seconds, args.max_drift_ratio
            )
        except ValueError as error:
            raise SystemExit(f"{scene['id']}: {error}") from error
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
                "alignment_drift_seconds": round(drift, 3),
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
