# Contracts and status

Use `report.md`, `manifests/report_manifest.yaml`, and
`manifests/evidence.tsv` as the three canonical report facts. The Markdown is
reader-facing prose. The manifest owns run-level scope, sources,
reproducibility, reviews, language, and status. The TSV owns record-level
evidence relationships. Do not duplicate machine state in prose and assume it
is reconciled.

## Required package

Prefer a project-defined canonical location. Otherwise use:

```text
results/{experiment_id}/
├── report.md
├── figures/
├── plot_data/
├── tables/
├── intermediate/
├── manifests/report_manifest.yaml
├── manifests/evidence.tsv
└── review/
```

Every measured-data Figure needs paired `plot_data/` or a reconstructable
intermediate. Every executable claim needs an implementation or command path.
Paths in the report should be relative to the report package when possible.

## Language

Store exactly one scalar `core.report_language`. Resolve it by user-explicit
choice, then project rule, then the user's current language. For revision,
record any existing-language conflict and its disposition before editing.
Technical tokens and paths may remain unchanged. Figure language follows the
project rule separately. Do not create a bilingual or translated twin in the
same run.

A requested derived-format handoff inherits the same report language. Mark a
different downstream language `rejected` and route it to a separate translation
task. This skill does not build DOCX/PDF/HTML/LaTeX artifacts.

## Sources and evidence joins

Give each source a stable ID, locator, availability, identity, and reverse list
of evidence record IDs. Every TSV `source_ids` value must resolve to one
manifest source. The reverse source pairs must equal the forward TSV pairs.
Use `available`, `restricted`, or `missing`; never relabel missing as
restricted. A restricted source may support a bounded claim when its concrete
identity is recorded and the access limit is disclosed.

For computed artifacts, `generation_entry` names an executable path or command.
For source-native or literature artifacts, use `not_applicable` and retain the
source join. For manual assembly, use `manual:<instruction path>`; use
`manual:unavailable` only with an explicit limitation, never as verified
quantitative support.

## Status and terminal behavior

- `working`: evidence or prose is still changing.
- `review`: deterministic repair or human review remains.
- `final`: all required sources, joins, artifacts, and reviews pass.
- `complete_with_limitations`: the same integrity gate passes and disclosed,
  bounded limitations remain.
- `blocked`: a report-critical source, required tool, or validator is missing.

Run the validator prospectively. A failure must write `review` or `blocked`,
not reuse an old pass. Any later change to report, manifest, evidence, or a
linked artifact invalidates terminal status and requires another validation.
