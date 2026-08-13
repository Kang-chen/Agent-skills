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
## Stable subtitle acceptance

- Each cue equals one reviewed sentence; no cue may end at a comma, colon, or arbitrary character limit.
- A visual line break is stored in `lines` and does not create another timed cue.
- Displayed technical notation is checked against the reviewed display layer, including digits, decimal points, percent signs, capitalization, and aliases.
- Forced-alignment atoms must equal normalized `tts_text`; token merges are allowed, content differences are errors.
- Record raw speech spans and padded cue spans. Lead and tail padding must be local to each sentence and clipped at adjacent speech midpoints.
- Reject one-line overflow, more than two visual lines, a split inside an ASCII technical token, or a sentence boundary inside one alignment token.
- Inspect a dense two-line cue over a visually busy slide and a cue containing numeric/technical notation in the exported video.
