---
name: evidence-backed-scientific-report
description: Use when the primary deliverable is a new or substantively revised scientific or technical Markdown report built from real data, analyses, figures, tables, code, or verified literature, and the work requires source traceability, claim boundaries, Figure/Table story planning, or reproducibility handoff. Not for a short brief or weekly update of reports that already exist, report-only audit, publication conversion, single-Figure plotting, one-line polishing, citation lookup, or pure format conversion.
---

# Evidence-Backed Scientific Report

Build one canonical Markdown report that a colleague who has never heard of this
project can read from top to bottom, understand what was attempted and what came
out, see it in pictures, and trace every number back to an inspectable source.

Three things make a report fail its reader, and this skill exists to prevent all
three: too few figures, prose only the author can parse, and a structure
reinvented from scratch on every run. The section skeleton, the figure slate,
and the plain-language contract below are part of the deliverable, not a style
preference.

## What the report IS

A report produced by this skill has four properties. Check them before you call
the work done.

1. **It is a picture book with text between the pictures.** Every main section
   opens with or contains a Figure. The Figure carries the finding; the prose
   says what to look at and what it means.
2. **It tells the whole attempt, including what failed.** A reader learns what
   was tried first, why it was changed, what the change bought, and what is
   still open. A report that shows only the final configuration hides the work.
3. **It assumes zero background.** Every object the reader meets — dataset, arm,
   model, metric, code name — gets a one-line plain definition the first time it
   appears, and again in the glossary table.
4. **It is traceable.** Numbers, Figures, and claims close back to sources,
   plot data, and commands.

When the result is negative or flat, all four still apply. A null result gets
the same picture count, the same narrative, and an explicit diagnosis section.
The report's job is to explain the process, not to advertise an outcome.

## Route the request

Use this skill for a new report or a revision that changes the report's
scientific story, evidence, claims, Figures, Tables, or reproducibility handoff.
For an existing report, modify the canonical file in place and let Git carry
versions. Do not create `_v2`, `_new`, or parallel translated copies.

Route a short brief, progress update, or weekly report built from reports that
already exist to `evidence-backed-report-brief`; it reads this skill's package
as its evidence layer. Route a training-plus-report workflow to its domain
workflow first. Route a single Figure request to `scientific-visualization`,
a citation lookup to a
research workflow, a prose-only polish to `stop-slop`, and publication-format
conversion to the relevant document workflow.

## Fix the shape before writing

Copy `assets/report_skeleton.md` to `report.md` and keep its slot markers. Copy
`assets/report_manifest.yaml` and `assets/evidence.tsv` into the package. New
packages and substantive revisions use manifest `schema_version: 2`. The three
files have distinct roles: `report.md` presents the complete reader narrative,
`report_manifest.yaml` maps each source to a stable `report_reference`, and
`evidence.tsv` remains the persistent review sidecar. Its `record_id` values
stay in the sidecar and are not report citations. Write report citations as
`[N](#ref-stable-anchor)` and define each target once in References with
`<a id="ref-stable-anchor"></a>`.

Resolve one scalar `report_language`: user-explicit language, then project rule,
then the user's current language. Technical tokens, paths, and identifiers stay
in their native form. **Figures are always labelled in English**, whatever the
report language, because figure text is not translated downstream.

Package layout:

```text
<output_root>/
├── report.md
├── figures/          # concept_*.png  data_*.png  result_*.png  diag_*.png
├── plot_data/        # one file per measured Figure, same stem as the Figure
├── tables/
├── manifests/report_manifest.yaml
├── manifests/evidence.tsv
└── review/number_checks.tsv   # every number re-derived from its source
```

When a brief built with `evidence-backed-report-brief` takes `report.md`, rename
this report to `report_full.md`, write its manifest locators against that name,
and pass `--report report_full.md` to both checks.

## The section slots

`report.md` carries these slots in this order. Each slot is a real section with
a concrete question as its heading; the marker comment stays in the file so the
quality checker can find it. Add extra sections anywhere; do not drop or reorder
these.

| slot | the question this section answers | what closes it |
|---|---|---|
| `tldr` | What did we ask, do, find, and what happens next? | 5 short bullets plus the single most important Figure |
| `glossary` | What do the names in this report mean? | a table: every abbreviation, arm code, dataset name, custom metric |
| `background` | What is this task, and where does the data come from? | a concept Figure of the end-to-end setting, written for zero background |
| `inputs` | What exactly goes into the model, and how was it processed? | real-image examples, cohort statistics, before/after preprocessing distributions |
| `model` | How is the model built, and what differs between the variants? | a concept Figure of the architecture and of each compared arm |
| `results` | What came out? | the main metric Figure with uncertainty, plus paired differences |
| `process` | What did we try, in what order, and what did each change buy? | the round-by-round narrative, failed attempts included |
| `diagnosis` | Why did it come out this way? | diagnostic Figures of intermediate representations and errors |
| `limits` | What does this result not support? | claim boundary, detection limit, unverified assumptions |
| `next` | What should be done next, and what would it cost? | ranked actions, each with expected gain and price |
| `repro` | How does a reader re-check this? | exact paths and runnable commands, with one line of purpose each |

`appendix` is optional and goes before References. References is the final
formal section.

When the work genuinely produced no adjustment rounds — a single configuration
run once — `process` still records that fact in one paragraph and lists the
decisions frozen before the run. Do not delete the slot.

## The figure slate

Read [figure-slate.md](references/figure-slate.md) before planning Figures. The
minimum slate, by filename prefix:

| prefix | family | minimum | must show |
|---|---|---|---|
| `concept_` | schematic drawings | 2 | the end-to-end setting; the compared variants side by side |
| `data_` | input description | 3 | real imagery or raw records; cohort composition; value distributions before and after processing |
| `result_` | outcome | 2 | per-arm metric with uncertainty; paired difference between arms |
| `diag_` | diagnosis | 3 | intermediate representation structure; relation to known covariates; per-unit errors |

Ten main Figures is the floor, not the target. Every slot section except
`glossary`, `repro`, and References contains at least one Figure or Table.

Each Figure answers a question you can write as a sentence. If you cannot write
the question, cut the Figure. Each measured Figure has a file in `plot_data/`
with the same stem. Schematics (`concept_`) are drawn — by vector code or by an
image-generation tool — and are labelled as illustrations, never as measurements.

## Plain language

Read [plain-language.md](references/plain-language.md) before writing prose. The
contract, in short:

- Open each section with one sentence naming the question, and close it with one
  sentence giving the answer.
- Define every abbreviation, arm code, and custom metric at first use, in the
  body, in the form `全称（英文/代号）` plus one clause of what it is for.
- Give every number its unit, comparison, uncertainty, and sample size.
- Prefer one fact per sentence. Split any sentence that needs a second reading.
- Say the mechanism in words a reader can picture before naming the technique.
- Every code that appears as a Figure axis label, legend entry, or panel label
  is defined in the glossary table. A reader who cannot decode the axis cannot
  use the Figure.

## The process narrative

Read [process-narrative.md](references/process-narrative.md). Each round is four
beats: why we changed something, what exactly changed, what the numbers did, and
what we learned. Rounds that failed are kept with their numbers. Anything tried
but not quantified is labelled as such.

## Run the stages

1. **Preflight:** identify the canonical report, project rules, available and
   missing sources, language, output root, and revision boundary. Record gaps;
   never conceal them to meet a deadline.
2. **Inventory:** run `scripts/inventory_sources.py` in dry-run mode, inspect
   the result, then use `--write` to update only manifest source/preflight
   blocks. Inventory sources in the manifest, then register report-critical
   evidence records in `evidence.tsv`. For each evidence row, set `source_ids`
   to the IDs of its supporting manifest sources and preserve each manifest
   source's stable `report_reference`. Register claims, not sentences.
3. **Plan the story:** write the Figure list before plotting. For each planned
   Figure, name its slot, its family prefix, the question it answers, its data
   source, and its encoding. Check the list against the minimum slate and
   against the one-visual-per-section rule. Plan Tables where the evidence is
   genuinely tabular; a Table does not discharge the figure-slate minimum.
4. **Build the Figures:** use `scientific-visualization` for measured-data
   Figures and save `plot_data/` alongside each one. Then **open every rendered
   Figure and look at it** — axes, direction, legend, overlap, type size, and
   whether it still answers its question. A Figure nobody looked at is not
   evidence.
5. **Write:** fill the slots in order, in the resolved report language. Put the
   traceability references inline as `[N](#ref-stable-anchor)`. Build one
   numbered References list from the manifest source references; display numbers
   follow list order while stable anchors do not change. References is the final
   formal report section, and any Appendix goes before it. After facts and claim
   boundaries are frozen, apply `stop-slop`, preserving numbers, directions,
   hedging, limitations, and terminology.
6. **Check:** re-derive every number the prose states, from the sources, and
   write the result to `review/number_checks.tsv` — one row per number with the
   claim, the stated value, the source path, the recomputed value, and a status.
   Counts, splits, and ranges are numbers too, and so is every number printed on
   a Figure or a schematic. A number that came from a planning document rather
   than a run artifact is labelled with that provenance in the row and in the
   prose. Never write that a verification passed unless that ledger is in the
   package and every row reads `ok`. Then run
   `python3 scripts/check_report_quality.py <output_root>` and
   `scripts/validate_report.py check`, and fix what they report. Both must pass
   before terminal status; use `terminal` only after human review and artifact
   readback are complete.
7. **Review:** do the claim-coverage, fresh-reader, and visual checks described
   in [claims-and-review.md](references/claims-and-review.md). Show completed
   steps, excluded hypotheses, limitations, unresolved items, and next actions.

Read [contracts-and-status.md](references/contracts-and-status.md) before editing
the manifest or evidence ledger, and [story-and-figures.md](references/story-and-figures.md)
for the reader-path and Figure-card rules.

When the budget is tight, a complete reader narrative with its Figures outranks
an exhaustive evidence ledger: register the report-critical claims and say in
the manifest which rows were left unregistered.

## Finish honestly

Use `working`, `review`, `final`, `complete_with_limitations`, or `blocked`.
Missing report-critical sources or an unavailable validator block terminal
status. A deterministic pass checks structure and traceability only; it does not
establish scientific correctness, paper readiness, or visual persuasiveness.

A substantive revision of a v1 package must migrate the manifest to
`schema_version: 2`, add source `report_reference` values, and replace legacy
record citations with reader-facing numeric references before terminal
validation.

Markdown is canonical. DOCX, PDF, HTML, and LaTeX are derived-format handoffs.
This skill records their requested format and inherited language but does not
build or verify them.
