# Storyboard format (figure walkthroughs)

One `storyboard.json` per chapter folder. `scripts/figure_walkthrough.py render` turns it into frame PNGs and `video-project.json`; `scripts/build_series.py chapter <folder>` runs the whole chapter.

```json
{
  "schema_version": 1,
  "canvas": {"width": 2560, "height": 1440, "fps": 30},
  "source": {
    "producer": "reviewed-report",
    "report": "../../report.md",
    "figure": "../../figures/heatmap.png"
  },
  "figure_regions": {"panel_a": [120, 25, 1345, 655]},
  "cards": {
    "read": {"head": "How to read", "body": "Rows are markers, columns are clusters"},
    "mean": {"head": "Batch mean", "body": "Mean residual of all cells", "color": "#009E73"}
  },
  "scenes": [
    {
      "id": "c01a",
      "title": "Conclusion sentence, not a figure name",
      "narration_display": "First sentence. Second sentence.",
      "tts_text": "First sentence. Second sentence.",
      "frames": [
        {"sentences": 1, "boxes": [{"region": "panel_a"}], "dim": true, "cards": ["read"], "active": "read"},
        {"sentences": 1, "image": "schematic/formula.png", "cards": ["read", "mean"], "active": "mean"}
      ]
    }
  ]
}
```

Paths resolve from the storyboard file.

## Fields

- `source.figure`: the figure that highlight boxes refer to. Optional `source.next_figure` is shown by frames with `"next_figure": true` (a preview of the next chapter, no boxes). `source.producer` defaults to `reviewed-report`; `source.report` becomes `source_bundle.article`.
- `scenes[].narration_display` / `tts_text`: subtitle text and TTS reading (see the text contract in SKILL.md). Both must split into the same number of sentences.
- `frames[].sentences`: how many sentences this frame covers. The sum per scene must equal the sentence count. After timing, `retime` makes each frame start when its first sentence starts.
- `frames[].boxes`: highlights on `source.figure`, in original figure pixels:
  - `{"region": "<name>"}` from `figure_regions` (`[x0, y0, x1, y1]`);
  - `{"box": [x0, y0, x1, y1]}` for a one-off box;
  - `{"cells": {"rows": [...], "cols": [...]}}` and `{"rows_full": [...], "with_bars": true}` need a `grid`:
    `{"x0": 100, "x1": 900, "markers": ["CD4", "CD8"], "rows": {"L0": [y0, y1]}}`. Columns split `x0..x1` evenly by `markers`. `rows_full` starts at `figure_regions.row_labels` if defined (else `x0`) and ends at `x1`, or at `figure_regions.bars` with `with_bars`.
- `frames[].dim`: fade everything except the boxes (the whole image when there are no boxes).
- `frames[].image`: show another image instead (formula, schematic, crop); boxes are not drawn on it.
- `frames[].cards` / `active`: cards shown in the right panel and the one being talked about (tinted, accent bar). A card `color` sets its head colour and accent bar; use the same colour as the matching formula term. Too many cards for the panel is an error: show fewer per frame.

## Building

```bash
python scripts/build_series.py chapter chapters/01_x chapters/02_y \
  --tts-command "python3 ~/skills-src/zgcai-tts/scripts/zgcai_tts.py batch --jsonl {jsonl} --outdir {outdir} --format wav --speech-rate 5 --workers 4"
python scripts/build_series.py concat chapters/01_x chapters/02_y --out final/video.mp4
```

Each chapter gets `build/` (frames, `video-project.json`, `sentences.jsonl`, `sentences/<id>.wav` cache with `texts.json`, `alignment.json`) and `output/` (mp4, srt, ass, 48 kHz narration). The TTS program reads `{"id", "text"}` rows and writes 16-bit mono `<outdir>/<id>.wav`; it reads its own credentials. Without `--tts-command` the build fails only if some sentence is new or changed.

Fonts: `render` uses `--font-regular/--font-bold/--font-index`, else `SPV_FONT_REGULAR`/`SPV_FONT_BOLD`/`SPV_FONT_INDEX`, else common Noto Sans CJK / PingFang / Microsoft YaHei paths, else `fc-match "Noto Sans CJK SC"`.

Formulas: `scripts/math_row.py` `draw_math_row(fig, y, [(text, colour), ...])` draws one row of colour-coded mathtext, shrinking the font to fit `max_width` with fixed gaps. Import it with `sys.path.insert(0, "<skill>/scripts")`.
