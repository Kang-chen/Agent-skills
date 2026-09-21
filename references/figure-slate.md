# Figure slate

A report explains a process. A process is easier to see than to read. Plan the
Figures before writing prose, and plan them as a set that covers the whole
pipeline, not as decoration for the conclusion.

## The four families

Name every Figure file by its family prefix so the coverage is checkable:
`concept_*`, `data_*`, `result_*`, `diag_*`.

### `concept_` — schematics (minimum 2)

Drawn, not measured. They exist so a reader with zero background can follow the
rest of the report.

- **Setting overview:** where the raw material comes from, what is measured,
  what the model consumes, what it predicts, and where the evaluation split
  sits. One box per real object, arrows for the actual data flow.
- **Variants side by side:** one panel per compared arm, aligned so the reader
  sees exactly what differs — which inputs enter, where they join, what is
  frozen, what is fitted.

Draw them with vector code (`matplotlib`, SVG) when the content is boxes,
arrows, and panels, which is most of the time. Use an image-generation tool only
for pictorial illustrations, and mark those as illustrations in the caption.
Never let a schematic imply a measurement it does not contain.

### `data_` — what goes in (minimum 3)

- **Real material.** Show the actual images, slides, crops, or raw records the
  pipeline consumes: one thumbnail per dataset or slide, plus at least one
  full-resolution detail crop. A reader who has never seen this modality needs
  to see it once. Every image panel carries a scale bar, and its plot data
  records the source file, the pyramid level or resolution, and the crop origin,
  so the panel can be regenerated. Draw the model's actual input unit (tile,
  window, patch) on the detail crop so the reader sees the scale the encoder
  works at.
- **Cohort composition.** Counts per group, per class, per batch, as bars. Show
  the class imbalance and the unit of analysis.
- **Value distributions before and after processing.** Histograms of the same
  quantity as it enters and after normalisation, clipping, or resampling. This
  is where silent preprocessing damage becomes visible.

### `result_` — what came out (minimum 2)

- **Per-arm metric with uncertainty.** Points or bars with intervals, arms
  ordered by value, and the chance level or comparator drawn as a reference
  line. Never a bare bar chart of point estimates.
- **Paired differences.** The comparison the study was designed to make, as
  differences with intervals, on the paired units. A difference plot and a
  levels plot are not the same Figure; show both when the design is paired.
- Add per-unit detail (per-patient, per-fold, per-slide scores) whenever the
  sample is small enough that the mean hides the spread.

### `diag_` — why it came out that way (minimum 3)

Required whether the result is positive or negative; a positive result also
needs to be explained.

- **Intermediate representation structure.** What the embedding actually
  encodes: variance spectrum, low-dimensional projection coloured by label and
  by the strongest covariate, or the collapse diagnostic if there is one.
- **Relation to known covariates.** Correlation of components or scores with the
  quantities already known to drive the endpoint. This is how a reader learns
  whether the new feature added information or re-encoded an old one.
- **Per-unit errors.** Which units are predicted wrongly, by how much, and
  whether the same units fail in every arm.

## Rules that apply to every Figure

- **One question per Figure**, and the caption states it. If you cannot write
  the question as a sentence, the Figure does not belong in the report.
- **No sentences inside the image.** No explanatory text blocks in the figure
  area or under it inside the image. The legend is graphical: line, marker, and
  colour handles with short labels. All explanation lives in the Markdown
  caption and body.
- **Figure text is English**, regardless of report language: titles, axis
  labels, tick labels, legend entries, panel letters, colourbar labels.
- **Encode uncertainty** wherever an estimate is shown, and say in the caption
  what the interval is.
- **Paired data gets paired encoding**: connected points or differences, not two
  independent boxes.
- **Every measured Figure ships its plot data** at `plot_data/<same stem>.csv`
  or `.tsv`, and its generation entry is a real path or command.
- **Every code on an axis, legend, or panel label is in the glossary.** A Figure
  whose axis is a list of internal codes is unreadable to everyone but its
  author, however good the encoding is.
- **Numbers printed on a Figure go through the same verification as prose
  numbers**, and appear in `review/number_checks.tsv`.
- **Look at the rendered file** before citing it. Check axes, direction, legend,
  overlap, type size, and that the caption matches what is drawn.

## Placement

Put the Figure where its question is asked, not in an appendix gallery. Each
slot section carries at least one Figure or Table. The `tldr` slot carries the
single Figure a reader would keep if they kept only one.
