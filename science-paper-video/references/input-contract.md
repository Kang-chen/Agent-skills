# Input contract

`video-project.json` is the boundary between scientific narration and video production.

```json
{
  "schema_version": 1,
  "source_bundle": {
    "producer": "science-paper-narrator",
    "article": "article.md",
    "figures_verified": true
  },
  "canvas": {"width": 3840, "height": 2160, "fps": 30},
  "scenes": [
    {
      "id": "s01",
      "slide_summary": "One concise on-screen claim",
      "narration_display": "Reviewed text shown in subtitles.",
      "tts_text": "Pronunciation-adjusted text sent to TTS.",
      "audio_file": "audio/s01.wav",
      "frames": [{"image": "frames/s01.png", "fraction": 1}]
    }
  ]
}
```

Paths resolve from the JSON file. `figures_verified` must be true. Each scene needs non-empty display and TTS text, one audio file, and at least one frame with a positive fraction. Figure provenance stays in the upstream narrator bundle and is referenced, not copied into a second ledger.
