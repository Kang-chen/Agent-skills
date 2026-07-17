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

## Figure-first default

Each main Figure has a card in `evidence.tsv` containing its question,
comparison, analysis unit, sample count, uncertainty, encoding, final size,
save target, source IDs, generation entry, artifact path, report locator, claim
boundary, and QA status. Link the Figure in Markdown and link or name its plot
data and implementation path in nearby prose. A Figure is not complete until
someone inspects the rendered artifact for axes, direction, legend, color,
overlap, type size, and caption consistency.

Use `scientific-visualization` during Figure execution for measured-data
Figures. Keep its publication-quality and accessibility checks, while following
the current project's language and style rules.

## Table-led override

Use a Table-led story when tables, schemas, deterministic logs, or algorithmic
gates are the direct evidence. Record the reason in the manifest override. Do
not manufacture a decorative Figure to satisfy the default. Give each main
Table the same question, source, generation, boundary, locator, and QA links.

## Fresh-reader check

Ask an independent reader to locate six items: core question, answer, main
evidence, inspection/reproduction entrypoint, limitations, and current
done/not-done/next status. Record missing or ambiguous locators. A checklist of
present headings is not evidence that the story is understandable.
