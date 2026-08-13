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

    def test_alignment_keeps_reviewed_display_text(self):
        align = load_script("align_subtitles.py")
        display = "VirTues 的 AUROC 为 0.823。"
        spoken = "Virtues 的 A U R O C 为零点八二三。"
        tokens = align.tokenize_for_alignment(spoken)
        timestamps = [[i * 120, (i + 1) * 120] for i in range(len(tokens))]
        cues = align.build_cues(display, spoken, tokens, timestamps, max_chars=24)
        self.assertEqual("".join(cue["text"] for cue in cues), display)
        self.assertTrue(all(a["end"] <= b["start"] for a, b in zip(cues, cues[1:])))

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
