# Evidence-backed report

Use one canonical report language. Replace the hypothetical Results questions
with concrete questions from the current task. Remove all instructional prose
before terminal validation.
Keep one explicit lowercase stable anchor immediately before every section used
by `evidence.tsv.report_locator`, and make each locator resolve exactly, for
example `report.md#primary-answer` to `<a id="primary-answer"></a>`.

## Glossary (only when task-specific terms need definition)

Include only task-specific or custom terms and metrics.
Do not expand AUC, GO, or KEGG by default.

<a id="primary-answer"></a>
## What is the primary answer and strongest evidence?

State the main answer, the Figure/Table evidence ID, and the boundary in plain
language. Place the main Figure or Table here and link its source data,
generation entry, implementation path, and evidence record.

<a id="reader-inspection"></a>
## How can a reader inspect or reproduce the evidence?

Give exact relative paths and runnable commands. Explain their purpose so the
report remains understandable without opening code or logs.

<a id="primary-comparison"></a>
## Does the primary comparison change the measured outcome?

Close the answer, verified observation, interpretation, alternative, boundary,
decision/status, and next evidence. Use this as a hypothetical pattern; write a
case-specific question in the real report.

<a id="alternative-explanation"></a>
## Which alternative explanation remains plausible?

Show competing explanations and the evidence that retains or excludes each
hypothesis. Do not hide negative, null, disputed, or inconclusive findings.

<a id="status-limitations-next"></a>
## What is done, excluded, limited, and next?

List completed steps, excluded hypotheses and their criteria, limitations,
unresolved work, and next actions.

<a id="methods-reproducibility"></a>
## Methods and reproducibility handoff

Record inputs, analysis units, parameters, environment or version evidence,
generation commands, output paths, and any derived-format handoff.

## References or appendix (only when needed)

Include verified literature and supporting detail that would otherwise obscure
the main reader path.
