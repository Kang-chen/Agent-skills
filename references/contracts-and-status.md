# Contracts and status

Use `report.md`, `manifests/report_manifest.yaml`, and
`manifests/evidence.tsv` as three canonical files with distinct responsibilities:

- `report.md` is the self-contained, reader-facing narrative. It carries the
  conclusions, numbers, interpretations, limits, and numbered References a
  reader needs without opening either sidecar.
- `manifests/report_manifest.yaml` uses `schema_version: 2` and owns run-level
  scope, source identity and availability, source `report_reference` values,
  reproducibility, reviews, language, and status.
- `manifests/evidence.tsv` is the persistent review ledger. It owns record-level
  `record_id`, `supported_by_ids`, `source_ids`, support state, and
  `report_locator` relationships. A `record_id` stays in this sidecar and is
  not a reader-facing report citation.

Do not duplicate machine state in prose and assume it is reconciled. Connect
the reader document to both sidecars through the v2 reference joins below.

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
of evidence record IDs. Each report-critical manifest source also has one
unique `report_reference` in `report.md#ref-stable-anchor` form. Every TSV
`source_ids` value must resolve to one manifest source. The reverse source pairs
must equal the forward TSV pairs.
Use `available`, `restricted`, or `missing`; never relabel missing as
restricted. A restricted source may support a bounded claim when its concrete
identity is recorded and the access limit is disclosed.

Close each report-critical join in this order:

1. `evidence.tsv.report_locator` identifies the report section containing the
   claim.
2. `evidence.tsv.source_ids` identifies the supporting manifest sources.
3. Each manifest source's `report_reference` identifies its stable References
   anchor.
4. The located report section cites that anchor as
   `[N](#ref-stable-anchor)`, and References defines it once as
   `<a id="ref-stable-anchor"></a>` next to numbered entry `N`.

Reference numbers may change when References are reordered; stable anchors and
manifest `report_reference` values do not.

References is the final formal report section. Put any Appendix before
References. Reference entries appear in numbered order from 1 through N, and
only entries inside that terminal References section satisfy a source join.
Reader-facing citations may appear in any earlier report section. Citations in
HTML comments or fenced code, including blockquote/list container fences, do
not satisfy the join.

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

Do not terminal-validate a v1 package under the v2 contract. Before a
substantive revision reaches terminal validation, migrate it to
`schema_version: 2`, add or inventory source `report_reference` values, create
the reader-facing numeric References, and rerun validation. This is an explicit
migration, not an automatic rewrite of an old report.
