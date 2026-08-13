import importlib.util
import json
import sys
import tempfile
import unittest
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


class SkillContractTests(unittest.TestCase):
    def test_skill_delegates_scientific_content_and_figure_extraction(self):
        text = (ROOT / "SKILL.md").read_text(encoding="utf-8")
        self.assertIn("**REQUIRED SUB-SKILL:** Use science-paper-narrator", text)
        self.assertIn("Do not recreate", text)
        self.assertIn("narration_display", text)
        self.assertIn("tts_text", text)


    def test_skill_requires_sentence_timing_and_separate_visual_wrapping(self):
        text = (ROOT / "SKILL.md").read_text(encoding="utf-8")

        self.assertIn("one timed cue per reviewed sentence", text)
        self.assertIn("normalized atomic-character level", text)
        self.assertIn("explicit reviewed override map", text)
        self.assertIn("translucent subtitle box", text)
    def test_alignment_keeps_reviewed_display_text(self):
        align = load_script("align_subtitles.py")
        display = "VirTues 的 AUROC 为 0.823。"
        spoken = "Virtues 的 A U R O C 为零点八二三。"
        tokens = align.tokenize_for_alignment(spoken)
        timestamps = [[i * 120, (i + 1) * 120] for i in range(len(tokens))]
        cues = align.build_cues(display, spoken, tokens, timestamps, max_chars=24)
        self.assertEqual("".join(cue["text"] for cue in cues), display)
        self.assertTrue(all(a["end"] <= b["start"] for a, b in zip(cues, cues[1:])))


    def test_alignment_keeps_complete_sentences_as_single_timed_cues(self):
        align = load_script("align_subtitles.py")
        display = "\u4ea4\u53c9\u9a8c\u8bc1 AUROC \u4e3a 0.823\uff0c\u6bd4\u57fa\u7ebf\u9ad8 5.14%\u3002\u8fd9\u662f\u7b2c\u4e8c\u53e5\u3002"
        spoken = "\u4ea4\u53c9\u9a8c\u8bc1 A U R O C \u4e3a\u96f6\u70b9\u516b\u4e8c\u4e09\uff0c\u6bd4\u57fa\u7ebf\u9ad8\u767e\u5206\u4e4b\u4e94\u70b9\u4e00\u56db\u3002\u8fd9\u662f\u7b2c\u4e8c\u53e5\u3002"
        tokens = align.tokenize_for_alignment(spoken)
        timestamps = [[index * 100, (index + 1) * 100] for index in range(len(tokens))]

        cues = align.build_cues(display, spoken, tokens, timestamps, max_chars=24)

        first_end = display.index("\u3002") + 1
        self.assertEqual(len(cues), 2)
        self.assertEqual(cues[0]["text"], display[:first_end])
        self.assertEqual(cues[1]["text"], display[first_end:])
        self.assertEqual("".join(cues[0]["lines"]), cues[0]["text"])
        self.assertLessEqual(len(cues[0]["lines"]), 2)

    def test_alignment_rejects_token_content_mismatch(self):
        align = load_script("align_subtitles.py")
        display = spoken = "\u7b2c\u4e00\u53e5\u3002\u7b2c\u4e8c\u53e5\u3002"
        tokens = align.tokenize_for_alignment(spoken)
        tokens[-1] = "x"
        timestamps = [[index * 100, (index + 1) * 100] for index in range(len(tokens))]

        with self.assertRaisesRegex(ValueError, "content differs"):
            align.build_cues(display, spoken, tokens, timestamps)

    def test_sentence_boundaries_follow_real_token_spans_without_cumulative_shift(self):
        align = load_script("align_subtitles.py")
        display = spoken = "\u7b2c\u4e00\u53e5\u3002\u7b2c\u4e8c\u53e5\u3002"
        tokens = align.tokenize_for_alignment(spoken)
        timestamps = [
            [0, 100], [100, 200], [200, 300],
            [900, 1000], [1000, 1100], [1100, 1200],
        ]

        cues = align.build_cues(display, spoken, tokens, timestamps, audio_duration=1.4)

        self.assertEqual(cues[0]["speech_start"], 0.0)
        self.assertEqual(cues[0]["speech_end"], 0.3)
        self.assertEqual(cues[1]["speech_start"], 0.9)
        self.assertEqual(cues[1]["speech_end"], 1.2)
        self.assertLess(cues[0]["end"], cues[1]["start"])

    def test_ass_style_uses_translucent_box_for_consistent_contrast(self):
        compose = load_script("compose_video.py")
        ass = compose.to_ass(
            [{"start": 0.0, "end": 2.0, "text": "caption", "lines": ["caption"]}],
            3840,
            2160,
        )

        style = next(line for line in ass.splitlines() if line.startswith("Style: Default,"))
        fields = style.removeprefix("Style: ").split(",")
        self.assertEqual(fields[15:18], ["3", "6", "0"])

    def test_global_timeline_preserves_visual_line_layout(self):
        compose = load_script("compose_video.py")
        project = {"scenes": [{"id": "s1"}]}
        aligned = {
            "segments": [
                {
                    "scene_id": "s1",
                    "duration_seconds": 2.0,
                    "cues": [
                        {"start": 0.1, "end": 1.8, "text": "firstsecond", "lines": ["first", "second"]}
                    ],
                }
            ]
        }

        cues, _ = compose.build_global_timeline(project, aligned, gap=0.7)

        self.assertEqual(cues[0]["lines"], ["first", "second"])

    def test_two_line_ass_cues_use_compact_font_without_extra_timing_cue(self):
        compose = load_script("compose_video.py")
        cues = [
            {"start": 0.0, "end": 3.0, "text": "firstsecond", "lines": ["first", "second"]}
        ]

        ass = compose.to_ass(cues, 3840, 2160)

        self.assertIn(r"{\fs74}first\Nsecond", ass)
        self.assertEqual(ass.count("Dialogue:"), 1)

    def test_visual_wrapping_prefers_punctuation_over_balanced_word_split(self):
        align = load_script("align_subtitles.py")
        text = "\u7532\u7532\u7532\u7532\u7532\uff0c\u4e59\u4e59\u4e59\u4e59\u4e59\u4e59\u4e59\u4e59\u4e59\u4e59\u4e59\u4e59\u4e59\u4e59\u4e59\u4e59\u3002"

        lines = align.split_display_lines(text, max_chars=18)

        self.assertEqual(len(lines), 2)
        self.assertTrue(lines[0].endswith("\uff0c"))
        self.assertEqual("".join(lines), text)

    def test_alignment_accepts_merged_english_tokens_when_atomic_content_matches(self):
        align = load_script("align_subtitles.py")
        display = spoken = "Cellpose\u3001InstanSeg \u53ef\u8fc1\u79fb\u3002\u4e0b\u4e00\u53e5\u3002"
        tokens = align.tokenize_for_alignment(spoken)
        tokens = [tokens[0] + tokens[1], *tokens[2:]]
        timestamps = [[index * 100, (index + 1) * 100] for index in range(len(tokens))]

        cues = align.build_cues(display, spoken, tokens, timestamps)

        self.assertEqual(len(cues), 2)
        self.assertEqual("".join(cue["text"] for cue in cues), display)
        self.assertEqual(cues[0]["token_end"], 4)

    def test_narrator_adapter_applies_reviewed_display_overrides_only(self):
        adapter = load_script("adapt_narrator_storyboard.py")
        storyboard = {
            "canvas": {"width": 3840, "height": 2160, "fps": 30},
            "audio_segments": [
                {"id": "s1", "slide_number": 1, "visible": "summary", "narration": "metric text"}
            ],
            "frames": [{"audio": 1, "slide_image": "slides/frame-01.png", "fraction": 1}],
        }
        timings = [{"slide_number": 1, "audio_file": "slide-01.wav", "text": "spoken metric"}]
        overrides = {"s1": "metric 0.823"}

        project = adapter.build_project(storyboard, timings, overrides, audio_subdir="audio")

        scene = project["scenes"][0]
        self.assertEqual(scene["narration_display"], "metric 0.823")
        self.assertEqual(scene["tts_text"], "spoken metric")
        self.assertEqual(scene["audio_file"], "audio/slide-01.wav")
        self.assertEqual(scene["frames"][0]["image"], "slides/frame-01.png")
        self.assertEqual(project["source_bundle"]["producer"], "science-paper-narrator")

    def test_validator_rejects_cues_that_split_one_sentence(self):
        validate = load_script("validate_package.py")
        narration = "\u8fd9\u662f\u5b8c\u6574\u65c1\u767d\u3002"
        project = {
            "scenes": [
                {
                    "id": "s1",
                    "narration_display": narration,
                    "tts_text": narration,
                    "audio_file": "s1.wav",
                    "frames": [{"image": "s1.png", "fraction": 1}],
                }
            ]
        }
        aligned = {
            "segments": [
                {
                    "scene_id": "s1",
                    "display_text": narration,
                    "duration_seconds": 3.0,
                    "cues": [
                        {"start": 0.0, "end": 1.0, "text": narration[:2]},
                        {"start": 1.0, "end": 2.0, "text": narration[2:]},
                    ],
                }
            ]
        }

        with self.assertRaisesRegex(ValueError, "sentence boundaries"):
            validate.validate_alignment(project, aligned)

    def test_validator_rejects_summary_text_as_subtitles(self):
        validate = load_script("validate_package.py")
        package = {
            "scenes": [
                {
                    "id": "s1",
                    "narration_display": "这是完整旁白。这里还有第二句。",
                    "tts_text": "这是完整旁白。这里还有第二句。",
                    "audio_file": "s1.wav",
                    "frames": [{"image": "s1.png", "fraction": 1}],
                }
            ]
        }
        aligned = {
            "segments": [
                {
                    "scene_id": "s1",
                    "display_text": "这是摘要。",
                    "duration_seconds": 3.0,
                    "cues": [{"start": 0.0, "end": 2.0, "text": "这是摘要。"}],
                }
            ]
        }
        with self.assertRaisesRegex(ValueError, "reviewed narration"):
            validate.validate_alignment(package, aligned)

    def test_minimal_contract_fixture_is_valid(self):
        validate = load_script("validate_package.py")
        fixture = ROOT / "tests" / "fixtures" / "minimal" / "video-project.json"
        package = json.loads(fixture.read_text(encoding="utf-8"))
        validate.validate_project_contract(package, fixture.parent)

    def test_subtitle_export_uses_one_shared_timeline(self):
        compose = load_script("compose_video.py")
        project = {"scenes": [{"id": "s1"}, {"id": "s2"}]}
        aligned = {
            "segments": [
                {"scene_id": "s1", "duration_seconds": 2.0, "cues": [{"start": 0.1, "end": 1.0, "text": "第一句。"}]},
                {"scene_id": "s2", "duration_seconds": 3.0, "cues": [{"start": 0.2, "end": 1.2, "text": "第二句。"}]},
            ]
        }
        cues, duration = compose.build_global_timeline(project, aligned, gap=0.7)
        self.assertEqual(cues[0]["start"], 0.1)
        self.assertAlmostEqual(cues[1]["start"], 2.9)
        self.assertAlmostEqual(duration, 6.4)
        self.assertIn("00:00:02,900", compose.to_srt(cues))


if __name__ == "__main__":
    unittest.main()
