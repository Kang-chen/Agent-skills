# QA criteria

## Scientific integrity

- Concatenated subtitle cues equal `narration_display` exactly for every scene.
- Slide summaries do not replace verbatim subtitles.
- Claims, numbers, qualifiers, Figure/panel identifiers, and sources remain traceable to the narrator bundle.

## Visuals

- Render at least 1920×1080; prefer 3840×2160 for dense scientific figures.
- Inspect representative exported-video frames at original resolution.
- Keep axes, legends, units, panel letters, statistics, scale bars, and comparison groups visible.
- Reject crops that upscale a low-resolution source without restoring readable information.

## Audio and subtitles

- Use the final rendered audio for alignment.
- Keep cue times monotonic, non-overlapping, positive, and within scene duration.
- Target no more than two lines and roughly 24 Chinese characters per cue.
- Record local TTS model and voice provenance. Do not silently use an online service.

## Media package

- Verify video dimensions, frame rate, codec, duration, audio sample rate, and channel count with `ffprobe`.
- Compare video duration with the sum of scene audio and declared gaps.
- Preserve alignment JSON and editable subtitles beside the burned-in MP4.
