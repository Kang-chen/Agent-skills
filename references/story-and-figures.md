# Story and Figures

Design the report around the questions a reader needs answered, not around the
order scripts ran. The reader should find the problem, main answer, strongest
evidence, inspection entrypoint, limitations, status, and next action without
opening code or logs.

## Reader shell

Lead with a title framed as object plus core question. Put a glossary first only
when task-specific concepts create real cognitive load. Define custom scores,
cohort labels, bespoke stages, or project-specific abbreviations. Do not expand
common AUC, GO, or KEGG terms by default.

After a short summary, show the main Figure or Table. Then provide a compact
inspection card with the exact source, plot-data/table path, generation command,
and implementation entrypoint. Results headings should be concrete questions,
for example, “Does treatment change the measured outcome?” or “Which failure
mode explains the performance gap?” Do not use “Question 1,” “Analysis,” or a
fixed IMRAD outline as substitutes for the report's actual questions.

Each question block closes four semantic slots:

1. the answer or current disposition;
2. the verified Figure/Table evidence;
3. the interpretation, alternative, and boundary;
4. the decision/status or unresolved reason and next evidence.

The slots need not be visible subheadings. Preserve negative, null, disputed,
and excluded hypotheses in the reader path; do not remove them because they
make the story less attractive.

Every report-critical question block includes the corresponding source
reference for each source joined through its evidence rows. Write it as
`[N](#ref-stable-anchor)`, using the stable target from the manifest source's
`report_reference`; let `N` follow the source's display order in References.
This keeps the reader path self-contained while the evidence sidecar retains
the claim-level audit relationships.

Place optional appendices before References. Keep one terminal References
chapter, with entries numbered 1 through N in visible order; do not place
ordinary report headings or reference entries after it.

## Figure-first default

Each main Figure has a card in `evidence.tsv` containing its question,
comparison, analysis unit, sample count, uncertainty, encoding, final size,
save target, source IDs, generation entry, artifact path, report locator, claim
boundary, and QA status. Link the Figure in Markdown and link or name its plot
data and implementation path in nearby prose. In that same report section,
include the numeric source references required by the Figure card's source
IDs. A Figure is not complete until someone inspects the rendered artifact for
axes, direction, legend, color, overlap, type size, and caption consistency.

Use `scientific-visualization` during Figure execution for measured-data
Figures. Keep its publication-quality and accessibility checks, while following
the current project's language and style rules. The families, minimum counts,
naming, and in-image rules are in [figure-slate.md](figure-slate.md).

## Table-led evidence

Use a Table when tables, schemas, deterministic logs, or algorithmic gates are
the direct evidence, and give each main Table the same question, source,
generation, boundary, locator, and QA links. A Table replaces a Figure only for
that specific piece of evidence; it never discharges the minimum figure slate in
[figure-slate.md](figure-slate.md), because the schematic, input-description,
and diagnostic families explain the process rather than the outcome. Do not
manufacture a decorative Figure either: if a planned Figure has no question, cut
it and replace it with one that does.

## Fresh-reader check

Give an independent reader only the raw report text. Confirm that the raw
report shows no internal `record_id`, every numeric reference resolves to a
unique, non-empty, reader-understandable source description, and the sources
and conclusions are understandable without opening the manifest or
`evidence.tsv`. Ask the reader to locate six items: core question, answer, main
evidence, inspection/reproduction entrypoint, limitations, and current
done/not-done/next status. Record missing or ambiguous locators. A checklist of
present headings is not evidence that the story is understandable. Claim
coverage and the validator remain responsible for the machine joins.
