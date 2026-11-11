"""Read back per-sentence TTS clips with ASR and list likely misreadings.

QA only; rendering never depends on it. For each sentence row (``id`` plus
``tts_text`` or ``text``, as written by ``sentence_timeline.py plan``) the clip
``<sentence-dir>/<id>.wav`` is transcribed and compared with the TTS text after
removing punctuation and spaces. Differences whose pinyin (without tones) is the
same are homophones from the recogniser, not TTS errors, and are hidden, so what
remains are candidates for real misreadings (wrong polyphone, skipped word,
letters read as a word). Listen to each one before changing ``tts_text``.

Homophone filtering needs ``pypinyin``; without it every difference is listed
and marked ``unfiltered``.

Engines:

- ``funasr`` (default): FunASR ``AutoModel(model=--model)``, default ``paraformer-zh``.
- ``--transcripts <jsonl>``: rows ``{"id", "text"}`` from any other engine; no model is loaded.
"""

from __future__ import annotations

import argparse
import difflib
import json
import re
import sys
from pathlib import Path
from typing import Callable, Iterable

PUNCTUATION = re.compile(r"[\s，。；？！、：“”‘’（）《》,.;?!:'\"()\-–—]")


def normalize(text: str) -> str:
    return PUNCTUATION.sub("", text).lower()


def load_pinyin() -> Callable[[str], list[str]] | None:
    try:
        from pypinyin import lazy_pinyin
    except ImportError:
        return None
    return lazy_pinyin


def is_homophone(ref: str, hyp: str, to_pinyin: Callable[[str], list[str]] | None) -> bool:
    """True when both strings read the same in toneless pinyin."""

    if to_pinyin is None or not ref or not hyp:
        return False
    return "".join(to_pinyin(ref)) == "".join(to_pinyin(hyp))


def diff_sentence(
    ref_text: str, hyp_text: str, to_pinyin: Callable[[str], list[str]] | None
) -> list[dict[str, str]]:
    """Differences between TTS text and ASR text, without same-pinyin substitutions."""

    ref, hyp = normalize(ref_text), normalize(hyp_text)
    issues = []
    for op, a0, a1, b0, b1 in difflib.SequenceMatcher(None, ref, hyp, autojunk=False).get_opcodes():
        if op == "equal":
            continue
        if op == "replace" and is_homophone(ref[a0:a1], hyp[b0:b1], to_pinyin):
            continue
        issues.append({"op": op, "expected": ref[a0:a1], "heard": hyp[b0:b1]})
    return issues


def read_rows(path: Path) -> list[dict]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    for row in rows:
        if "id" not in row or not (row.get("tts_text") or row.get("text")):
            raise ValueError(f"{path}: each row needs 'id' and 'tts_text' (or 'text').")
    return rows


def funasr_transcriber(model_name: str, device: str | None) -> Callable[[Path], str]:
    from funasr import AutoModel

    kwargs = {"model": model_name, "disable_update": True}
    if device:
        kwargs["device"] = device
    model = AutoModel(**kwargs)
    return lambda wav: str(model.generate(input=str(wav))[0]["text"])


def readback(
    rows: Iterable[dict],
    transcripts: dict[str, str],
    to_pinyin: Callable[[str], list[str]] | None,
) -> list[dict]:
    """One report entry per sentence that still differs after homophone filtering."""

    report = []
    for row in rows:
        ref = row.get("tts_text") or row["text"]
        issues = diff_sentence(ref, transcripts[row["id"]], to_pinyin)
        if issues:
            report.append({"id": row["id"], "tts_text": ref, "asr_text": transcripts[row["id"]], "issues": issues})
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("sentences", type=Path, help="JSONL from sentence_timeline.py plan.")
    parser.add_argument("--sentence-dir", type=Path, help="Directory with <id>.wav clips (funasr engine).")
    parser.add_argument("--transcripts", type=Path, help="JSONL {id, text} from another ASR engine.")
    parser.add_argument("--model", default="paraformer-zh")
    parser.add_argument("--device", default=None)
    parser.add_argument("--output", type=Path, help="Write the full report as JSON.")
    args = parser.parse_args()

    rows = read_rows(args.sentences)
    if args.transcripts:
        transcripts = {row["id"]: row["text"] for row in read_rows(args.transcripts)}
    else:
        if args.sentence_dir is None:
            parser.error("--sentence-dir is required unless --transcripts is given.")
        transcribe = funasr_transcriber(args.model, args.device)
        transcripts = {row["id"]: transcribe(args.sentence_dir / f"{row['id']}.wav") for row in rows}
    to_pinyin = load_pinyin()
    if to_pinyin is None:
        print("pypinyin not installed: homophones are not filtered (pip install pypinyin).", file=sys.stderr)
    report = readback(rows, transcripts, to_pinyin)
    for entry in report:
        flag = "" if to_pinyin else " [unfiltered]"
        pairs = "; ".join(f"{item['op']} '{item['expected']}' -> '{item['heard']}'" for item in entry["issues"])
        print(f"{entry['id']}{flag}: {pairs}")
    summary = {"checked": len(rows), "flagged": len(report), "homophones_filtered": to_pinyin is not None}
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps({**summary, "sentences": report}, ensure_ascii=False, indent=2),
                               encoding="utf-8")
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
