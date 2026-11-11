import importlib.util
import math
import sys
import tempfile
import unittest
import wave
from array import array
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RATE = 24000


def load_script(name: str):
    path = ROOT / "scripts" / name
    spec = importlib.util.spec_from_file_location(path.stem, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def write_clip(path: Path, lead: float, voice: float, tail: float, rate: int = RATE) -> int:
    """Write silence + 300 Hz tone + silence as 16-bit mono PCM; return frame count."""

    samples = array("h", [0] * round(lead * rate))
    samples.extend(
        round(12000 * math.sin(2 * math.pi * 300 * i / rate)) for i in range(round(voice * rate))
    )
    samples.extend([0] * round(tail * rate))
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes(samples.tobytes())
    return len(samples)


def two_scene_project() -> dict:
    return {
        "schema_version": 1,
        "source_bundle": {"producer": "reviewed-report", "figures_verified": True},
        "canvas": {"width": 320, "height": 180, "fps": 24},
        "scenes": [
            {
                "id": "s1",
                "narration_display": "AUROC 为 0.823。第二句没有数字！",
                "tts_text": "A U R O C 为零点八二三。第二句没有数字！",
                "audio_file": "audio/s1.wav",
                "frames": [{"image": "s1.png", "fraction": 1}],
            },
            {
                "id": "s2",
                "narration_display": "只有一句。",
                "tts_text": "只有一句。",
                "audio_file": "audio/s2.wav",
                "frames": [{"image": "s2.png", "fraction": 1}],
            },
        ],
    }


class SentenceTimelineTests(unittest.TestCase):
    def test_plan_pairs_display_and_tts_sentences(self):
        timeline = load_script("sentence_timeline.py")

        rows = timeline.plan_sentences(two_scene_project())

        self.assertEqual([row["id"] for row in rows], ["s1_01", "s1_02", "s2_01"])
        self.assertEqual(rows[0]["display_text"], "AUROC 为 0.823。")
        self.assertEqual(rows[0]["tts_text"], "A U R O C 为零点八二三。")

    def test_plan_rejects_mismatched_sentence_counts(self):
        timeline = load_script("sentence_timeline.py")
        project = two_scene_project()
        project["scenes"][1]["tts_text"] = "只有，一句"
        project["scenes"][1]["narration_display"] = "只有。一句。"

        with self.assertRaisesRegex(ValueError, "sentences"):
            timeline.plan_sentences(project)

    def test_build_times_each_cue_from_its_own_clip(self):
        timeline = load_script("sentence_timeline.py")
        validate = load_script("validate_package.py")
        project = two_scene_project()
        with tempfile.TemporaryDirectory() as temp_name:
            base = Path(temp_name)
            clips = base / "sentences"
            first = write_clip(clips / "s1_01.wav", lead=0.10, voice=1.00, tail=0.20)
            second = write_clip(clips / "s1_02.wav", lead=0.05, voice=0.70, tail=0.15)
            write_clip(clips / "s2_01.wav", lead=0.0, voice=0.50, tail=0.0)

            aligned = timeline.build_timeline(project, base, clips, pause=0.3, max_chars=24)

            cues = aligned["segments"][0]["cues"]
            self.assertEqual(aligned["timing_policy"], "per-sentence-tts-audio")
            self.assertEqual(cues[0]["start"], 0.0)
            self.assertEqual(cues[0]["end"], round(first / RATE, 3))
            self.assertEqual(cues[1]["start"], round(first / RATE + 0.3, 3))
            self.assertEqual(cues[1]["end"], round((first + second) / RATE + 0.3, 3))
            self.assertAlmostEqual(cues[0]["speech_start"], 0.10, delta=0.03)
            self.assertAlmostEqual(cues[0]["speech_end"], 1.10, delta=0.03)
            self.assertAlmostEqual(cues[1]["speech_end"], cues[1]["start"] + 0.75, delta=0.03)
            self.assertEqual(cues[0]["text"], "AUROC 为 0.823。")

            with wave.open(str(base / "audio" / "s1.wav"), "rb") as handle:
                frames = handle.getnframes()
            self.assertEqual(frames, first + second + round(0.3 * RATE))
            self.assertEqual(aligned["segments"][0]["duration_seconds"], round(frames / RATE, 3))

            report = validate.validate_alignment(project, aligned)
            self.assertEqual(report["subtitle_cues"], 3)

    def test_build_rejects_missing_or_mixed_rate_clips(self):
        timeline = load_script("sentence_timeline.py")
        project = two_scene_project()
        with tempfile.TemporaryDirectory() as temp_name:
            base = Path(temp_name)
            clips = base / "sentences"
            write_clip(clips / "s1_01.wav", 0.0, 0.5, 0.0)
            with self.assertRaises(FileNotFoundError):
                timeline.build_timeline(project, base, clips)
            write_clip(clips / "s1_02.wav", 0.0, 0.5, 0.0, rate=16000)
            with self.assertRaisesRegex(ValueError, "sample rates"):
                timeline.build_timeline(project, base, clips)


class AlignmentDriftTests(unittest.TestCase):
    def test_speech_bounds_finds_voiced_span(self):
        align = load_script("align_subtitles.py")
        with tempfile.TemporaryDirectory() as temp_name:
            clip = Path(temp_name) / "clip.wav"
            write_clip(clip, lead=0.4, voice=1.2, tail=0.6)
            samples, rate = align.read_mono_samples(clip)

        start, end = align.speech_bounds(samples, rate)

        self.assertAlmostEqual(start, 0.4, delta=0.03)
        self.assertAlmostEqual(end, 1.6, delta=0.03)

    def test_drift_check_rejects_measured_fa_zh_compression(self):
        align = load_script("align_subtitles.py")

        # 2026-11-11 实测：49.06 s 音频，语音到约 49 s，fa-zh 末 token 为 44.65 s
        with self.assertRaisesRegex(ValueError, "sentence_timeline.py"):
            align.check_alignment_drift(44.65, 49.0, 49.06)

    def test_drift_check_uses_both_absolute_and_relative_limits(self):
        align = load_script("align_subtitles.py")

        self.assertAlmostEqual(align.check_alignment_drift(19.8, 20.0, 20.5), 0.2)
        with self.assertRaises(ValueError):
            align.check_alignment_drift(39.3, 40.0, 40.5)  # 0.7 s > 0.5 s
        with self.assertRaises(ValueError):
            align.check_alignment_drift(4.5, 4.8, 5.0)  # 0.3 s > 3% of 5 s


class ProducerPolicyTests(unittest.TestCase):
    def test_validator_is_strict_by_default_and_allows_explicit_producer(self):
        validate = load_script("validate_package.py")
        project = two_scene_project()

        with self.assertRaisesRegex(ValueError, "--allow-producer"):
            validate.validate_project_contract(project, ROOT)
        validate.validate_project_contract(
            project, ROOT, allowed_producers=("science-paper-narrator", "reviewed-report")
        )


if __name__ == "__main__":
    unittest.main()
