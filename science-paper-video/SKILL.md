---
name: science-paper-video
description: Use when turning a verified scientific-paper explanation, narrator content bundle, article, or paper PDF into a narrated PPT-style video with readable paper figures, local open-source TTS, synchronized Chinese subtitles, and editable deliverables.
---

# Science Paper Video

## Core boundary

Build the audiovisual layer around reviewed scientific content.

**REQUIRED SUB-SKILL:** Use science-paper-narrator when the input lacks a reviewed article, evidence ledger, or verified high-resolution figure crops.

Treat `science-paper-narrator` as the only source for paper interpretation, scientific claims, figure extraction, crop manifests, and figure-integrity review. Do not recreate or silently fork those workflows here. This skill owns storyboarding for video, slide composition, local TTS, audio-text alignment, rendering, and media QA.

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
4. Generate one local open-source TTS file per scene. Record the model, voice reference, speed, `tts_text`, duration, and file hash.
5. Run `scripts/align_subtitles.py` with the final audio. Use reviewed `narration_display` for visible text and `tts_text` only for timestamp prediction.
6. Render PPTX, WAV, SRT/ASS, and MP4 from one timeline. Never estimate subtitle timing from character counts.
7. Run `scripts/validate_package.py` and the checks in [references/qa-criteria.md](references/qa-criteria.md). Inspect representative frames at original resolution before delivery.

## Failure policy

- Stop when figure crops are unverified or too small; return the missing upstream requirement.
- Stop when aligned subtitle text differs from reviewed narration.
- Preserve intermediate slides, audio, alignment JSON, and logs if TTS, alignment, or rendering fails.
- Fall back from word-level to sentence-level alignment only when sentence boundaries come from actual audio segments. Never fall back to proportional character timing.

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
7. Wrap long sentences into at most two visual lines, preferring punctuation or whitespace and preserving technical tokens. Use a compact two-line style and a translucent subtitle box for stable contrast.

Run `scripts/adapt_narrator_storyboard.py` only for legacy narrator bundles. It converts the existing storyboard and TTS manifest into the video input contract; it does not reinterpret the paper or extract figures.
