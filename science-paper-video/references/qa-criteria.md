# QA criteria

## Scientific integrity

- Concatenated subtitle cues equal `narration_display` exactly for every scene.
- Slide summaries do not replace verbatim subtitles.
- Claims, numbers, qualifiers, Figure/panel identifiers, and sources remain traceable to the narrator bundle.

## Visuals

- Render at least 1920×1080; prefer 3840×2160 for dense scientific figures.
- Inspect representative exported-video frames at original resolution.
- Keep axes, legends, units, panel letters, statistics, scale bars, and comparison groups visible.
- The opening page shows the usage sentence from the SKILL.md resource use policy, verbatim, readable, and clear of subtitles.
- Every figure slide is an unmodified original with a credit, or an own chart labeled as such.
- Reject crops that upscale a low-resolution source without restoring readable information.

## Audio and subtitles

- Use the final rendered audio for alignment.
- Default to per-sentence TTS timing (`sentence_timeline.py`); `timing_policy` in the alignment JSON records which path was used.
- With forced alignment, the last cue's `speech_end` must reach the end of audible speech. fa-zh failed this on 2026-11-11 (Doubao TTS 2.0 about 9% short; Kokoro-82M similar); treat a drift error as a blocker.
- Spot-check the last cue of the longest scene by ear in the rendered video.
- Keep cue times monotonic, non-overlapping, positive, and within scene duration.
- Keep each cue within two visual lines; line width is `--max-chars` in `sentence_timeline.py build` / `align_subtitles.py`, by default derived from canvas width, subtitle size and margins (36 width units at 1440p; CJK = 1, ASCII = 0.56). If a reviewed sentence cannot fit, revise the narration instead of splitting the timed cue.
- Run `scripts/asr_readback.py` on the sentence clips and listen to every flagged sentence; homophones are already hidden when `pypinyin` is installed.
- Narration WAV and MP4 audio are 48 kHz.
- Record the TTS engine or service, model version, and voice. Name any online service in the QA report; never use one silently.

## Media package

- Verify video dimensions, frame rate, codec, duration, audio sample rate, and channel count with `ffprobe`.
- Compare video duration with the sum of scene audio and declared gaps.
- Preserve alignment JSON and editable subtitles beside the burned-in MP4.

## Stable subtitle acceptance

- Each cue equals one reviewed sentence; no cue may end at a comma, colon, or arbitrary character limit.
- A visual line break is stored in `lines` and does not create another timed cue.
- Displayed technical notation is checked against the reviewed display layer, including digits, decimal points, percent signs, capitalization, and aliases.
- Forced-alignment path only: atoms must equal normalized `tts_text`; token merges are allowed, content differences are errors.
- Record raw speech spans and padded cue spans. Lead and tail padding must be local to each sentence and clipped at adjacent speech midpoints.
- Reject one-line overflow, more than two visual lines, a split inside an ASCII technical token, or a sentence boundary inside one alignment token.
- Inspect a dense two-line cue over a visually busy slide and a cue containing numeric/technical notation in the exported video.
- Subtitles have no background box by default; check legibility of the outlined text over the brightest and busiest slide.

## Figure explanation

- Each figure follows: question, how to read, one worked example, overall pattern, meaning and evidence limits, transition.
- Self-built statistics have an on-screen formula with color-coded variables and a stated normalization; common statistics are only named.
- The highlighted region matches the sentence being spoken (`figure_walkthrough.py retime` after timing).
- Self-drawn formulas and schematics contain English and math only.

## In-house reviewed reports

- Each chapter has a `facts.md` ledger citing `file:line` for every number and claim in the narration.
- An independent narration review checked overstatement and stated denominators; each accepted rebuttal was verified against the source.
