---
name: science-paper-video
description: Use when turning a verified scientific-paper explanation, narrator content bundle, article, or paper PDF into a narrated PPT-style video with readable paper figures, local open-source TTS, synchronized Chinese subtitles, and editable deliverables.
---

# Science Paper Video

## Core boundary

Build the audiovisual layer around reviewed scientific content.

**REQUIRED SUB-SKILL:** Use science-paper-narrator when the input lacks a reviewed article, evidence ledger, or verified high-resolution figure crops.

Treat `science-paper-narrator` as the only source for paper interpretation, scientific claims, figure extraction, crop manifests, and figure-integrity review. Do not recreate or silently fork those workflows here. This skill owns storyboarding for video, slide composition, local TTS, audio-text alignment, rendering, and media QA.

## Resource use policy

Everything in these videos is for personal study and research, with no commercial use.

- Show this sentence, verbatim, at the bottom of the video's opening page: `所有figure和资源仅用于个人学习和研究，不进行任何商业行为`
- Show the paper's original figures directly and leave them unmodified: no repainting, masking, or relabeling. Zooming, panning, moving the figure aside, and a highlight box are display choices. Explanatory animation sits beside the figure, not on top of it. For a no-derivatives licence, prefer the whole figure plus zoom; if a panel crop is needed for legibility, keep its label, axes, and legend, name the panel in the credit, and tell the user.
- Use an own chart only when the numbers exist solely in a table or text, and credit it as `Own chart from <source>`. Credit originals as `<citation> · <licence> · original figure, unmodified · research and study only`.
- The sentence states intent; it does not change licence terms. Record each source's licence in the provenance manifest, and flag no-derivatives and non-CC sources to the user before any public release.

## Text contract

Keep three fields separate:

| Field | Purpose |
|---|---|
| `slide_summary` | Short text visible in the slide layout |
| `narration_display` | Reviewed narration and final subtitle source of truth |
| `tts_text` | Pronunciation form sent to the local TTS engine |

Allow aliases such as `VirTues` → `Virtues`, `H&E` → `H E`, and `0.823` → `零点八二三` only in `tts_text`. Never use free ASR output as final subtitle text.

## Workflow

1. Validate the narrator bundle. If it is incomplete, invoke `science-paper-narrator`; do not implement a fallback paper reader or screenshotter.
2. Create `video-project.json` following [references/input-contract.md](references/input-contract.md). Preserve every scientific qualifier and figure provenance record.
3. Compose 16:9 slides from verified crops. Keep plot labels, legends, axes, units, panel letters, statistical marks, and required controls readable at the target resolution.
4. Default timing path, sentence by sentence: run `scripts/sentence_timeline.py plan` to get one TTS request per reviewed sentence, synthesize each to `<sentence-dir>/<scene>_<NN>.wav`, then run `scripts/sentence_timeline.py build`. It concatenates the clips per scene (0.3 s pause) and writes `alignment.json` with each cue taken from its own clip's real start and end. Record the TTS model, voice, speed, sentence text, duration, and file hash.
5. Use `scripts/align_subtitles.py` (FunASR fa-zh forced alignment) only for audio that cannot be synthesized per sentence. It stops when its last token and the audio energy disagree by more than 0.5 s or 3%; then switch to step 4, never rescale.
6. Render PPTX, WAV, SRT/ASS, and MP4 from one timeline with `scripts/compose_video.py`. Subtitles default to a transparent background: bold white `Noto Sans CJK SC`, black outline, light shadow (64 px one line, 60 px two lines at 1440p, scaled with height); `--subtitle-style box` restores the translucent box. Never estimate subtitle timing from character counts.
7. Run `scripts/validate_package.py` and the checks in [references/qa-criteria.md](references/qa-criteria.md). Inspect representative frames at original resolution before delivery. For an already reviewed in-house report, set `source_bundle.producer` to `reviewed-report` and pass `--allow-producer reviewed-report`; the default stays strict.

## Known defect: fa-zh compresses the time axis

Measured 2026-11-11: on Doubao TTS 2.0 audio (24 kHz, speech_rate 5) fa-zh put the last sentence's end at 0.90-0.93 of the audio length in all 10 scenes (a 49.06 s clip ended at 44.65 s, voice ran to about 49 s). Resampling to 16 kHz gave identical output. The same day, Kokoro-82M audio and single-sentence clips from both engines also ended 7-28% early, so treat fa-zh timestamps on TTS audio as unreliable. Subtitles drift ahead of the voice, worst at the end. Hence sentence-level timing from real clips is the default.

## Explaining a figure

For each figure: the question it answers, how to read it (variables and encodings), one cell or row as a worked example, the overall pattern, biological meaning and evidence limits, then the bridge to the next figure.

- Name common statistics (such as a z value) without deriving them.
- Give every self-built statistic a short formula, color each variable, and state how it is normalized.
- Keep schematics minimal; usually show only the counterexample that gets flagged.
- Highlight on screen whatever the narration is talking about, timed from the real sentence audio.

## Failure policy

- Stop when figure crops are unverified or too small; return the missing upstream requirement.
- Stop when aligned subtitle text differs from reviewed narration.
- Preserve intermediate slides, audio, alignment JSON, and logs if TTS, alignment, or rendering fails.
- Fall back from word-level to sentence-level alignment only when sentence boundaries come from actual audio segments, such as per-sentence TTS clips. Never fall back to proportional character timing.
- Stop when `align_subtitles.py` reports drift; do not lower the limits to get past it.

## Deliverables

Return the editable deck, final video, narration audio, SRT/ASS, `video-project.json`, alignment JSON, provenance manifest, and QA report. Report the exact video resolution, duration, audio format, subtitle cue count, and unresolved subjective checks such as voice preference.
## Stable subtitle contract

Treat timing, semantic cue boundaries, and visual line wrapping as separate layers:

1. Create one timed cue per reviewed sentence. A two-line subtitle is one cue with a `lines` field, not two timed cues.
2. Keep technical notation in `narration_display` exactly as reviewed: for example `AUROC`, `0.823`, `5.14%`, `CD4`, and `VirTues`.
3. Keep pronunciation aliases and Chinese number readings only in `tts_text`. Apply display corrections through an explicit reviewed override map; never globally convert Chinese number words.
4. Verify FunASR output against `tts_text` at normalized atomic-character level. Accept token merges only when content is identical; reject missing, added, or changed content.
5. Map reviewed sentence boundaries to actual aligned-token boundaries. If a boundary falls inside one alignment token, stop for review.
6. Add lead/tail padding from each sentence's own speech span. Resolve overlap at the midpoint between adjacent speech spans; never shift later cues cumulatively.
7. Wrap long sentences into at most two visual lines, preferring punctuation or whitespace and preserving technical tokens. Use a compact two-line size and a transparent background with outline and shadow; an outline above about 2.5 px at 1440p blurs the strokes.

Run `scripts/adapt_narrator_storyboard.py` only for legacy narrator bundles. It converts the existing storyboard and TTS manifest into the video input contract; it does not reinterpret the paper or extract figures.
