import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


@unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"), "FFmpeg is required")
class ComposeEndToEndTests(unittest.TestCase):
    def test_compose_produces_video_audio_and_exact_subtitles(self):
        with tempfile.TemporaryDirectory() as temp_name:
            temp = Path(temp_name)
            frame = temp / "frame.png"
            audio = temp / "narration.wav"
            project_path = temp / "video-project.json"
            alignment_path = temp / "alignment.json"
            output = temp / "output"

            subprocess.run(
                [
                    "ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi",
                    "-i", "color=c=0x17324d:s=320x180:d=0.1", "-frames:v", "1", str(frame),
                ],
                check=True,
            )
            subprocess.run(
                [
                    "ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi",
                    "-i", "sine=frequency=440:duration=1.2", "-ar", "48000", "-ac", "1", str(audio),
                ],
                check=True,
            )

            display_text = "Exact reviewed narration."
            project = {
                "schema_version": 1,
                "source_bundle": {"producer": "science-paper-narrator", "figures_verified": True},
                "canvas": {"width": 320, "height": 180, "fps": 24},
                "scenes": [
                    {
                        "id": "s1",
                        "slide_summary": "One-line slide summary",
                        "narration_display": display_text,
                        "tts_text": display_text,
                        "audio_file": audio.name,
                        "frames": [{"image": frame.name, "fraction": 1}],
                    }
                ],
            }
            alignment = {
                "segments": [
                    {
                        "scene_id": "s1",
                        "display_text": display_text,
                        "duration_seconds": 1.2,
                        "cues": [{"start": 0.05, "end": 1.1, "text": display_text}],
                    }
                ]
            }
            project_path.write_text(json.dumps(project), encoding="utf-8")
            alignment_path.write_text(json.dumps(alignment), encoding="utf-8")

            subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts" / "compose_video.py"),
                    str(project_path),
                    str(alignment_path),
                    "--output-dir", str(output),
                    "--name", "e2e",
                    "--gap", "0.2",
                    "--preset", "ultrafast",
                ],
                check=True,
                capture_output=True,
                text=True,
            )

            video = output / "e2e.mp4"
            self.assertTrue(video.is_file())
            self.assertEqual((output / "e2e.srt").read_text(encoding="utf-8").count(display_text), 1)
            probe = json.loads(
                subprocess.run(
                    [
                        "ffprobe", "-v", "error", "-show_entries",
                        "stream=codec_type,width,height,sample_rate,channels", "-of", "json", str(video),
                    ],
                    check=True,
                    capture_output=True,
                    text=True,
                ).stdout
            )
            streams = probe["streams"]
            video_stream = next(item for item in streams if item["codec_type"] == "video")
            audio_stream = next(item for item in streams if item["codec_type"] == "audio")
            self.assertEqual((video_stream["width"], video_stream["height"]), (320, 180))
            self.assertEqual(audio_stream["sample_rate"], "48000")
            self.assertEqual(audio_stream["channels"], 1)


if __name__ == "__main__":
    unittest.main()
