"""Tests for the figure walkthrough renderer, series builder, formula helper and ASR read-back."""

import importlib.util
import json
import math
import shutil
import subprocess
import sys
import tempfile
import unittest
import wave
from array import array
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load_script(name: str):
    path = ROOT / "scripts" / name
    spec = importlib.util.spec_from_file_location(path.stem, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def has_module(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def write_tone(path: Path, seconds: float, rate: int = 24000) -> None:
    samples = array("h", [0] * round(0.05 * rate))
    samples.extend(round(9000 * math.sin(2 * math.pi * 220 * i / rate)) for i in range(round(seconds * rate)))
    samples.extend([0] * round(0.05 * rate))
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes(samples.tobytes())


GRID_STORYBOARD = {
    "figure_regions": {"row_labels": [0, 10, 100, 500], "bars": [600, 10, 700, 520], "panel": [10, 10, 50, 40]},
    "grid": {"x0": 100, "x1": 340, "markers": ["A", "B", "C", "D"], "rows": {"r1": [10, 40], "r2": [41, 70]}},
}


def make_storyboard(base: Path, width: int = 640, height: int = 360) -> Path:
    from PIL import Image

    figure = Image.new("RGB", (400, 300), (40, 90, 160))
    figure.save(base / "figure.png")
    Image.new("RGB", (200, 100), (0, 160, 0)).save(base / "formula.png")
    storyboard = {
        "schema_version": 1,
        "canvas": {"width": width, "height": height, "fps": 24},
        "title": "Test chapter",
        "source": {"report": "report.md", "figure": "figure.png"},
        "figure_regions": {"left": [0, 0, 200, 300]},
        "cards": {
            "a": {"head": "Read", "body": "Rows are markers"},
            "b": {"head": "Formula", "body": "Blue term", "color": "#0072B2"},
        },
        "scenes": [
            {
                "id": "t01a",
                "title": "Scene one claim",
                "narration_display": "第一句。第二句。第三句。",
                "tts_text": "第一句。第二句。第三句。",
                "frames": [
                    {"sentences": 1, "boxes": [{"region": "left"}], "dim": True, "cards": ["a"], "active": "a"},
                    {"sentences": 2, "image": "formula.png", "cards": ["a", "b"], "active": "b"},
                ],
            }
        ],
    }
    path = base / "storyboard.json"
    path.write_text(json.dumps(storyboard, ensure_ascii=False), encoding="utf-8")
    return path


class SubtitleWidthTests(unittest.TestCase):
    def test_default_line_width_fits_canvas(self):
        compose = load_script("compose_video.py")
        units = compose.max_line_units(2560, 1440)
        usable = 2560 - 2 * compose.subtitle_margin(1440)
        # 64 px 字号下一行 CJK 字必须放得下，且旧默认 45 放不下
        self.assertLessEqual(units * 64, usable)
        self.assertGreater(45 * 64, usable)
        self.assertGreaterEqual(units, 30)
        self.assertLess(compose.max_line_units(2560, 1440, font_size=96), units)

    def test_sentence_timeline_wraps_with_canvas_width(self):
        timeline = load_script("sentence_timeline.py")
        sentence = "这" * 40 + "。"
        project = {
            "canvas": {"width": 2560, "height": 1440},
            "scenes": [{"id": "s1", "narration_display": sentence, "tts_text": sentence, "audio_file": "audio/s1.wav"}],
        }
        with tempfile.TemporaryDirectory() as temp_name:
            base = Path(temp_name)
            write_tone(base / "sentences" / "s1_01.wav", 0.5)
            aligned = timeline.build_timeline(project, base, base / "sentences")
            override = timeline.build_timeline(project, base, base / "sentences", max_chars=45)
        self.assertEqual(len(aligned["segments"][0]["cues"][0]["lines"]), 2)
        self.assertEqual(len(override["segments"][0]["cues"][0]["lines"]), 1)


class WalkthroughUnitTests(unittest.TestCase):
    def setUp(self):
        self.walk = load_script("figure_walkthrough.py")

    def test_resolve_box_variants(self):
        resolve = self.walk.resolve_box
        # 每列 60 像素：B 到 C 两列落在 160 至 280
        self.assertEqual(resolve({"cells": {"rows": ["r2"], "cols": ["B", "C"]}}, GRID_STORYBOARD), (160, 41, 280, 71))
        self.assertEqual(resolve({"rows_full": ["r1", "r2"], "with_bars": True}, GRID_STORYBOARD), (0, 10, 700, 71))
        self.assertEqual(resolve({"rows_full": ["r1"]}, GRID_STORYBOARD), (0, 10, 340, 41))
        self.assertEqual(resolve({"region": "panel"}, GRID_STORYBOARD), (10, 10, 50, 40))
        self.assertEqual(resolve({"box": [1, 2, 30, 40]}, {}), (1, 2, 30, 40))
        with self.assertRaises(KeyError):
            resolve({"cells": {"rows": ["r1"], "cols": ["A"]}}, {})

    def test_frame_fractions_follow_cue_starts(self):
        cues = [{"start": 0.1}, {"start": 2.0}, {"start": 5.5}]
        fractions = self.walk.frame_fractions([1, 2], cues, duration=8.0, gap=0.7)
        # 第二帧从第二句开始；最后一帧吃掉场景尾部空白与场景间隔
        self.assertEqual(fractions, [2.0, 6.7])
        with self.assertRaises(ValueError):
            self.walk.frame_fractions([2, 2], cues, duration=8.0, gap=0.7)

    def test_retime_project(self):
        project = {"scenes": [{"id": "s", "frames": [{"sentences": 2, "fraction": 2}, {"sentences": 1, "fraction": 1}]}]}
        alignment = {"segments": [{"scene_id": "s", "duration_seconds": 6.0,
                                   "cues": [{"start": 0.0}, {"start": 1.5}, {"start": 4.25}]}]}
        self.walk.retime_project(project, alignment, gap=0.5)
        self.assertEqual([frame["fraction"] for frame in project["scenes"][0]["frames"]], [4.25, 2.25])

    def test_hex_to_rgb(self):
        self.assertEqual(self.walk.hex_to_rgb("#0072B2"), (0, 114, 178))
        with self.assertRaises(ValueError):
            self.walk.hex_to_rgb("#123")

    def test_explicit_font_argument(self):
        with self.assertRaises(FileNotFoundError):
            self.walk.find_fonts(Path("/nonexistent/font.ttf"))


def colours(image) -> set:
    return {colour for _, colour in image.getcolors(maxcolors=1 << 22)}


def fonts_available(walk) -> bool:
    try:
        walk.find_fonts()
    except FileNotFoundError:
        return False
    return True


class WalkthroughRenderTests(unittest.TestCase):
    def setUp(self):
        self.walk = load_script("figure_walkthrough.py")
        if not fonts_available(self.walk):
            self.skipTest("No CJK font on this machine")

    def test_render_highlight_dim_override_and_cards(self):
        from PIL import Image

        with tempfile.TemporaryDirectory() as temp_name:
            base = Path(temp_name)
            project = self.walk.render_storyboard(make_storyboard(base), base / "build")
            frames = project["scenes"][0]["frames"]
            self.assertEqual([frame["sentences"] for frame in frames], [1, 2])
            self.assertEqual(project["source_bundle"]["producer"], "reviewed-report")
            first = Image.open(base / "build" / frames[0]["image"]).convert("RGB")
            second = Image.open(base / "build" / frames[1]["image"]).convert("RGB")
            layout = self.walk.Layout.for_canvas(640, 360)
            scale, fx, fy = self.walk.place_image(Image.new("RGB", (400, 300)), layout.figure_box)
            # 焦点区域保持原色，非焦点区域被调暗，焦点框边上是强调色
            self.assertEqual(first.getpixel((fx + round(100 * scale), fy + round(150 * scale))), (40, 90, 160))
            dimmed = first.getpixel((fx + round(300 * scale), fy + round(150 * scale)))
            self.assertGreater(dimmed[0], 120)
            edge_x = fx + round(200 * scale) + layout.px(5)
            row = [first.getpixel((x, fy + round(150 * scale))) for x in range(edge_x - 4, edge_x + 5)]
            self.assertIn(self.walk.ACCENT, row)
            # 换图帧显示公式图，且不画高亮框
            self.assertIn((0, 160, 0), colours(second.crop(layout.figure_box)))
            self.assertNotIn(self.walk.ACCENT, colours(second.crop(layout.figure_box)))
            # 带颜色的当前卡片用该颜色画强调条
            self.assertIn((0, 114, 178), colours(second.crop(layout.panel_box)))

    def test_render_rejects_sentence_mismatch(self):
        with tempfile.TemporaryDirectory() as temp_name:
            base = Path(temp_name)
            path = make_storyboard(base)
            data = json.loads(path.read_text(encoding="utf-8"))
            data["scenes"][0]["frames"][1]["sentences"] = 1
            path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
            with self.assertRaises(ValueError):
                self.walk.render_storyboard(path, base / "build")


@unittest.skipUnless(has_module("matplotlib"), "matplotlib is required")
class MathRowTests(unittest.TestCase):
    def test_draw_math_row_shrinks_wide_rows(self):
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        math_row = load_script("math_row.py")
        fig = plt.figure(figsize=(4, 2), dpi=100)
        segments = [(r"$x_{i}$", "#0072B2"), (r"$\ =\ $", "#222222"), (r"$\mu$", "#D55E00")]
        size = math_row.draw_math_row(fig, 0.5, segments, fontsize=20)
        self.assertEqual(size, 20)
        self.assertEqual(len(fig.texts), 3)
        self.assertEqual(fig.texts[0].get_color(), "#0072B2")
        wide = [(r"$\mathrm{a\ very\ long\ formula\ segment}$", "#222222")] * 4
        fig2 = plt.figure(figsize=(4, 2), dpi=100)
        shrunk = math_row.draw_math_row(fig2, 0.5, wide, fontsize=30, max_width=0.9)
        self.assertLess(shrunk, 30)
        widths = math_row.measure_segments(fig2, wide, shrunk)
        self.assertLessEqual(sum(widths) + 0.012 * 3, 0.9 + 0.01)
        plt.close("all")


FAKE_TTS = """
import json, math, sys, wave
from array import array
from pathlib import Path
jsonl, outdir = Path(sys.argv[1]), Path(sys.argv[2])
for line in jsonl.read_text(encoding="utf-8").splitlines():
    row = json.loads(line)
    rate = 24000
    samples = array("h", [round(8000 * math.sin(2 * math.pi * 200 * i / rate)) for i in range(int(rate * 0.4))])
    with wave.open(str(outdir / (row["id"] + ".wav")), "wb") as handle:
        handle.setnchannels(1); handle.setsampwidth(2); handle.setframerate(rate)
        handle.writeframes(samples.tobytes())
    with open(outdir.parent / "tts_calls.log", "a", encoding="utf-8") as log:
        log.write(row["id"] + "\\n")
"""


class SeriesUnitTests(unittest.TestCase):
    def setUp(self):
        self.series = load_script("build_series.py")

    def test_tts_command_template(self):
        command = self.series.tts_command("tts --jsonl {jsonl} --outdir {outdir} --rate 5", Path("/a.jsonl"), Path("/o"))
        self.assertEqual(command, ["tts", "--jsonl", "/a.jsonl", "--outdir", "/o", "--rate", "5"])
        with self.assertRaises(ValueError):
            self.series.tts_command("tts --jsonl {jsonl}", Path("/a"), Path("/o"))

    def test_cache_resynthesizes_only_changed_text(self):
        with tempfile.TemporaryDirectory() as temp_name:
            base = Path(temp_name)
            script = base / "fake_tts.py"
            script.write_text(FAKE_TTS, encoding="utf-8")
            template = f"{sys.executable} {script} {{jsonl}} {{outdir}}"
            rows = [{"id": "s_01", "tts_text": "一。"}, {"id": "s_02", "tts_text": "二。"}]
            sentence_dir = base / "build" / "sentences"
            self.assertEqual(self.series.synthesize_pending(rows, sentence_dir, template), 2)
            self.assertEqual(self.series.synthesize_pending(rows, sentence_dir, None), 0)
            rows[1]["tts_text"] = "二改。"
            with self.assertRaises(RuntimeError):
                self.series.synthesize_pending(rows, sentence_dir, None)
            self.assertEqual(self.series.synthesize_pending(rows, sentence_dir, template), 1)
            calls = (base / "build" / "tts_calls.log").read_text(encoding="utf-8").split()
            self.assertEqual(calls, ["s_01", "s_02", "s_02"])

    def test_merge_srt_offsets(self):
        part = "1\n00:00:00,500 --> 00:00:01,000\n甲\n\n2\n00:00:01,200 --> 00:00:02,000\n乙\n"
        text, count = self.series.merge_srt([(part, 0.0), (part, 10.0)])
        self.assertEqual(count, 4)
        self.assertIn("3\n00:00:10,500 --> 00:00:11,000\n甲", text)


@unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"), "FFmpeg is required")
class SeriesEndToEndTests(unittest.TestCase):
    def test_chapter_and_concat(self):
        walk = load_script("figure_walkthrough.py")
        if not fonts_available(walk):
            self.skipTest("No CJK font on this machine")
        with tempfile.TemporaryDirectory() as temp_name:
            base = Path(temp_name)
            chapter = base / "c01"
            chapter.mkdir()
            make_storyboard(chapter)
            script = base / "fake_tts.py"
            script.write_text(FAKE_TTS, encoding="utf-8")
            command = [
                sys.executable, str(ROOT / "scripts" / "build_series.py"), "chapter", str(chapter),
                "--tts-command", f"{sys.executable} {script} {{jsonl}} {{outdir}}", "--preset", "ultrafast",
            ]
            result = subprocess.run(command, check=True, capture_output=True, text=True)
            summary = json.loads(result.stdout.strip().splitlines()[-1])
            self.assertEqual(summary["subtitle_cues"], 3)
            self.assertEqual(summary["new_tts_sentences"], 3)
            rerun = json.loads(subprocess.run(command, check=True, capture_output=True, text=True).stdout.strip())
            self.assertEqual(rerun["new_tts_sentences"], 0)
            project = json.loads((chapter / "build" / "video-project.json").read_text(encoding="utf-8"))
            fractions = [frame["fraction"] for frame in project["scenes"][0]["frames"]]
            # 第一帧只覆盖第一句：0.4 秒音频加 0.3 秒句间停顿
            self.assertAlmostEqual(fractions[0], 0.7, places=2)
            output = chapter / "output"
            with wave.open(str(output / "c01-narration.wav"), "rb") as handle:
                self.assertEqual(handle.getframerate(), 48000)
            self.assertFalse((output / ".c01-work").exists())

            out = base / "final" / "series.mp4"
            subprocess.run(
                [sys.executable, str(ROOT / "scripts" / "build_series.py"), "concat", str(chapter), str(chapter),
                 "--out", str(out)],
                check=True, capture_output=True, text=True,
            )
            chapters = json.loads(out.with_suffix(".chapters.json").read_text(encoding="utf-8"))
            self.assertEqual(chapters["subtitle_cues"], 6)
            self.assertEqual(chapters["chapters"][0]["title"], "Scene one claim")
            self.assertGreater(chapters["chapters"][1]["start_seconds"], 2.0)


class AsrReadbackTests(unittest.TestCase):
    def setUp(self):
        self.asr = load_script("asr_readback.py")

    def test_diff_reports_real_changes(self):
        issues = self.asr.diff_sentence("读作 CD 十一 b。", "读作cd十b", None)
        self.assertEqual(issues, [{"op": "delete", "expected": "一", "heard": ""}])
        self.assertEqual(self.asr.diff_sentence("第一句，好。", "第一句好", None), [])

    @unittest.skipUnless(has_module("pypinyin"), "pypinyin is required")
    def test_homophones_are_hidden(self):
        to_pinyin = self.asr.load_pinyin()
        self.assertEqual(self.asr.diff_sentence("残差的方差。", "残差的方叉", to_pinyin), [])
        issues = self.asr.diff_sentence("簇的平均值。", "组的平均值", to_pinyin)
        self.assertEqual(issues, [{"op": "replace", "expected": "簇", "heard": "组"}])

    def test_readback_with_transcripts(self):
        rows = [{"id": "a", "tts_text": "一样。"}, {"id": "b", "text": "不同。"}]
        report = self.asr.readback(rows, {"a": "一样", "b": "不对"}, None)
        self.assertEqual([entry["id"] for entry in report], ["b"])


if __name__ == "__main__":
    unittest.main()
