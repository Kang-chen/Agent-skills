---
name: evidence-backed-scientific-report
description: Use when the primary deliverable is a new or substantively revised scientific or technical Markdown report built from real data, analyses, figures, tables, code, or verified literature, and the work requires source traceability, claim boundaries, Figure/Table story planning, or reproducibility handoff. Not for report-only audit, publication conversion, single-Figure plotting, one-line polishing, citation lookup, or pure format conversion.
---

# Evidence-Backed Scientific Report

Create or substantively revise one canonical scientific Markdown report whose
claims, Figures, Tables, limits, and next actions can be traced to inspectable
sources. Make the report understandable without requiring the reader to open
code or logs, while preserving exact paths and commands for drill-down.

## Route the request

Use this skill for a new report or a revision that changes the report's
scientific story, evidence, claims, Figures, Tables, or reproducibility handoff.
For an existing report, modify the canonical file in place and let Git carry
versions. Do not create `_v2`, `_new`, or parallel translated copies.

Route a training-plus-report workflow to its domain workflow first. Route a
single Figure request to `scientific-visualization`, a citation lookup to a
research workflow, a prose-only polish to `stop-slop`, and publication-format
conversion to the relevant document workflow.

## Establish the contract

Copy the three assets into the output package. Resolve one scalar
`report_language`: user-explicit language, then project rule, then the user's
current language. Technical tokens, paths, and identifiers may remain in their
native form. A second language is a separate translation run, never a bilingual
mode inside this report run.

Read [contracts-and-status.md](references/contracts-and-status.md) before
editing the manifest or evidence ledger. Read
[story-and-figures.md](references/story-and-figures.md) before planning the
reader path. Read [claims-and-review.md](references/claims-and-review.md) when
claims are causal, mechanistic, clinical, generalizing, disputed, negative,
null, or inconclusive.

## Run the five stages

1. **Preflight:** identify the canonical report, project rules, available and
   missing sources, language, output root, and revision boundary. Record gaps;
   never conceal them to meet a deadline.
2. **Inventory:** run `scripts/inventory_sources.py` in dry-run mode, inspect
   the result, then use `--write` to update only manifest source/preflight
   blocks. Register every report-critical source in `evidence.tsv`.
3. **Plan the story:** default to Figure-first; use a Table-led story when the
   evidence is genuinely tabular. Every main Figure or Table answers a concrete
   question and links its source IDs, generation entry, plot data or
   intermediate, implementation path, boundary, and report locator.
4. **Execute and write:** at the Figure execution stage, use
   `scientific-visualization` to create or review measured-data Figures and
   inspect the rendered artifact. Write concrete question headings. Put an
   optional glossary first and include only task-specific or custom metrics;
   do not expand common AUC, GO, or KEGG terms by default. After facts and
   claim boundaries are frozen, apply `stop-slop` in the late writing stage.
   Preserve scientific hedging, numbers, directions, limitations, and terms.
5. **Review and validate:** perform separate claim-coverage, fresh-reader, and
   visual checks when applicable. Show completed steps, excluded hypotheses,
   limitations, unresolved items, and next actions. Run
   `scripts/validate_report.py check`; use `terminal` only after human reviews
   and artifact readback are complete.

## Finish honestly

Use `working`, `review`, `final`, `complete_with_limitations`, or `blocked`.
Missing report-critical sources or an unavailable validator block terminal
status. A deterministic pass checks structure and traceability only; it does
not establish scientific correctness, paper readiness, or visual
persuasiveness.

Markdown is canonical. DOCX, PDF, HTML, and LaTeX are derived-format handoffs.
This skill records their requested format and inherited language but does not build
or verify them.
