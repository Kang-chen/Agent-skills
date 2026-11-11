"""Build chapter videos from storyboards and join them into one video.

Each chapter directory holds ``storyboard.json``. ``chapter`` runs, per chapter:

1. ``figure_walkthrough.py render`` -> ``build/frames/`` and ``build/video-project.json``;
2. ``sentence_timeline.py plan`` -> ``build/sentences.jsonl``;
3. per-sentence TTS into ``build/sentences/<id>.wav``; a sentence is synthesized
   again only if its ``tts_text`` changed or its clip is missing
   (``build/sentences/texts.json`` records the text behind each clip);
4. ``sentence_timeline.py build`` -> scene audio and ``build/alignment.json``;
5. ``figure_walkthrough.py retime`` so each highlight follows its sentence;
6. ``compose_video.py`` -> ``output/<chapter>.mp4``, SRT/ASS and 48 kHz narration.

``concat`` joins built chapters by stream copy, merges the SRT files with time
offsets and writes ``<out>.chapters.json`` with chapter start times.

TTS is an external program given as a command template with ``{jsonl}`` and
``{outdir}`` placeholders. It reads JSONL rows ``{"id", "text"}`` and must write
16-bit mono PCM ``<outdir>/<id>.wav``. Example (Doubao TTS 2.0 via zgcai-tts)::

    --tts-command "python3 ~/skills-src/zgcai-tts/scripts/zgcai_tts.py batch \
        --jsonl {jsonl} --outdir {outdir} --format wav --speech-rate 5 --workers 4"

The TTS program reads its own credentials; this script never reads or prints them.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS))
from figure_walkthrough import render_storyboard, retime_project  # noqa: E402
from sentence_timeline import build_timeline, plan_sentences  # noqa: E402

GAP_SECONDS = 0.7
PAUSE_SECONDS = 0.3
TEXT_RECORD = "texts.json"


def run(cmd: list[str], cwd: Path | None = None) -> str:
    """Run a command; on failure raise with its output."""

    result = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"Command failed ({result.returncode}): {' '.join(cmd)}\n{result.stdout}\n{result.stderr}")
    return result.stdout


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def pending_sentences(rows: list[dict], sentence_dir: Path) -> list[dict]:
    """Sentences whose clip is missing or was synthesized from different text."""

    record_path = sentence_dir / TEXT_RECORD
    record = json.loads(record_path.read_text(encoding="utf-8")) if record_path.is_file() else {}
    return [
        row for row in rows
        if record.get(row["id"]) != row["tts_text"] or not (sentence_dir / f"{row['id']}.wav").is_file()
    ]


def tts_command(template: str, jsonl: Path, outdir: Path) -> list[str]:
    """Expand ``{jsonl}`` / ``{outdir}`` and ``~`` in a TTS command template."""

    if "{jsonl}" not in template or "{outdir}" not in template:
        raise ValueError("TTS command template needs both {jsonl} and {outdir} placeholders.")
    return [
        os.path.expanduser(token.replace("{jsonl}", str(jsonl)).replace("{outdir}", str(outdir)))
        for token in shlex.split(template)
    ]


def synthesize_pending(rows: list[dict], sentence_dir: Path, template: str | None) -> int:
    """Synthesize only new or changed sentences; return how many were sent to TTS."""

    sentence_dir.mkdir(parents=True, exist_ok=True)
    pending = pending_sentences(rows, sentence_dir)
    if not pending:
        return 0
    if not template:
        ids = ", ".join(row["id"] for row in pending)
        raise RuntimeError(f"{len(pending)} sentences need TTS but no --tts-command was given: {ids}")
    jsonl = sentence_dir.parent / "tts_pending.jsonl"
    write_jsonl(jsonl, [{"id": row["id"], "text": row["tts_text"]} for row in pending])
    run(tts_command(template, jsonl, sentence_dir))
    missing = [row["id"] for row in pending if not (sentence_dir / f"{row['id']}.wav").is_file()]
    if missing:
        raise RuntimeError(f"TTS did not produce: {missing}")
    record_path = sentence_dir / TEXT_RECORD
    record = json.loads(record_path.read_text(encoding="utf-8")) if record_path.is_file() else {}
    record.update({row["id"]: row["tts_text"] for row in pending})
    record_path.write_text(json.dumps(record, ensure_ascii=False, indent=1), encoding="utf-8")
    jsonl.unlink()
    return len(pending)


def build_chapter(
    chapter: Path,
    tts_template: str | None,
    gap: float = GAP_SECONDS,
    pause: float = PAUSE_SECONDS,
    max_chars: float | None = None,
    compose_args: list[str] | None = None,
    keep_work: bool = False,
) -> dict[str, object]:
    """Build one chapter directory into ``output/<name>.mp4``; return a summary."""

    name = chapter.name
    build, output = chapter / "build", chapter / "output"
    project = render_storyboard(chapter / "storyboard.json", build)
    rows = plan_sentences(project)
    write_jsonl(build / "sentences.jsonl", rows)
    n_new = synthesize_pending(rows, build / "sentences", tts_template)

    alignment = build_timeline(project, build, build / "sentences", pause=pause, max_chars=max_chars)
    alignment_path = build / "alignment.json"
    alignment_path.write_text(json.dumps(alignment, ensure_ascii=False, indent=2), encoding="utf-8")
    retime_project(project, alignment, gap)
    project_path = build / "video-project.json"
    project_path.write_text(json.dumps(project, ensure_ascii=False, indent=2), encoding="utf-8")

    if output.exists():
        shutil.rmtree(output)
    stdout = run([
        sys.executable, str(SCRIPTS / "compose_video.py"), str(project_path), str(alignment_path),
        "--output-dir", str(output), "--name", name, "--gap", str(gap), *(compose_args or []),
    ])
    info = json.loads(stdout.strip().splitlines()[-1])
    if not keep_work:
        shutil.rmtree(output / f".{name}-work", ignore_errors=True)
    return {
        "chapter": name,
        "duration_seconds": info["duration_seconds"],
        "subtitle_cues": info["subtitle_cues"],
        "new_tts_sentences": n_new,
    }


def probe_duration(path: Path) -> float:
    return float(run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(path)]).strip())


def srt_time(seconds: float) -> str:
    millis = round(seconds * 1000)
    hours, millis = divmod(millis, 3_600_000)
    minutes, millis = divmod(millis, 60_000)
    secs, millis = divmod(millis, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def parse_srt_time(stamp: str) -> float:
    hms, millis = stamp.strip().split(",")
    hours, minutes, secs = (int(part) for part in hms.split(":"))
    return hours * 3600 + minutes * 60 + secs + int(millis) / 1000


def read_srt(text: str) -> list[tuple[float, float, list[str]]]:
    cues = []
    for block in text.strip().split("\n\n"):
        lines = block.splitlines()
        if len(lines) < 3:
            continue
        start, end = (parse_srt_time(part) for part in lines[1].split("-->"))
        cues.append((start, end, lines[2:]))
    return cues


def merge_srt(parts: list[tuple[str, float]]) -> tuple[str, int]:
    """Merge ``(srt_text, offset_seconds)`` parts into one SRT; return text and cue count."""

    cues = [(start + offset, end + offset, lines) for text, offset in parts for start, end, lines in read_srt(text)]
    blocks = [
        f"{index}\n{srt_time(start)} --> {srt_time(end)}\n" + "\n".join(lines)
        for index, (start, end, lines) in enumerate(cues, 1)
    ]
    return "\n\n".join(blocks) + "\n", len(cues)


def chapter_title(chapter: Path) -> str:
    """First scene title of a chapter storyboard, else the directory name."""

    storyboard = chapter / "storyboard.json"
    if storyboard.is_file():
        scenes = json.loads(storyboard.read_text(encoding="utf-8")).get("scenes", [])
        if scenes and scenes[0].get("title"):
            return str(scenes[0]["title"])
    return chapter.name


def concat_chapters(chapters: list[Path], out: Path) -> dict[str, object]:
    """Join chapter videos (stream copy), merge subtitles and write chapter start times."""

    videos = [chapter / "output" / f"{chapter.name}.mp4" for chapter in chapters]
    for video in videos:
        if not video.is_file():
            raise FileNotFoundError(video)
    out.parent.mkdir(parents=True, exist_ok=True)
    list_file = out.with_suffix(".concat.txt")
    list_file.write_text("".join(f"file '{video.resolve().as_posix()}'\n" for video in videos), encoding="utf-8")
    run(["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", str(list_file), "-c", "copy",
         "-movflags", "+faststart", str(out)])
    list_file.unlink()

    parts, marks, offset = [], [], 0.0
    for chapter, video in zip(chapters, videos):
        marks.append({"chapter": chapter.name, "title": chapter_title(chapter), "start_seconds": round(offset, 3)})
        parts.append(((chapter / "output" / f"{chapter.name}.srt").read_text(encoding="utf-8"), offset))
        offset += probe_duration(video)
    srt_text, cue_count = merge_srt(parts)
    out.with_suffix(".srt").write_text(srt_text, encoding="utf-8")
    summary = {
        "video": out.name,
        "duration_seconds": round(probe_duration(out), 3),
        "subtitle_cues": cue_count,
        "chapters": marks,
    }
    out.with_suffix(".chapters.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    chapter = sub.add_parser("chapter", help="Build one or more chapter directories.")
    chapter.add_argument("chapters", type=Path, nargs="+")
    chapter.add_argument(
        "--tts-command",
        help="TTS command template with {jsonl} and {outdir}; needed only when sentences are new or changed.",
    )
    chapter.add_argument("--gap", type=float, default=GAP_SECONDS, help="Silence after each scene, seconds.")
    chapter.add_argument("--pause", type=float, default=PAUSE_SECONDS, help="Silence between sentences, seconds.")
    chapter.add_argument("--max-chars", type=float, help="Subtitle line width; default derives from the canvas.")
    chapter.add_argument("--crf", type=int, default=18)
    chapter.add_argument("--preset", default="medium")
    chapter.add_argument("--keep-work", action="store_true", help="Keep compose_video's per-frame clips.")
    concat = sub.add_parser("concat", help="Concatenate built chapters.")
    concat.add_argument("chapters", type=Path, nargs="+")
    concat.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    if args.command == "chapter":
        compose_args = ["--crf", str(args.crf), "--preset", args.preset]
        for path in args.chapters:
            summary = build_chapter(
                path.resolve(), args.tts_command, gap=args.gap, pause=args.pause,
                max_chars=args.max_chars, compose_args=compose_args, keep_work=args.keep_work,
            )
            print(json.dumps(summary, ensure_ascii=False), flush=True)
        return
    summary = concat_chapters([path.resolve() for path in args.chapters], args.out.resolve())
    print(json.dumps(summary, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
