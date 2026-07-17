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

State the main answer, the strongest Figure/Table evidence, and the boundary in
plain language. Support each report-critical statement with its reader-facing
source reference, for example [1](#ref-source-example). Place the main Figure or
Table here and link its source data, generation entry, and implementation path.
Keep the claim-level evidence record in the review sidecar.

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

## Appendix (only when needed)

Put supporting detail here only when it would otherwise obscure the main reader
path. Keep this section before References so References remains the final
formal report section.

<a id="references"></a>
## References

Use the display number assigned by References and the stable anchor from the
manifest source's `report_reference`. Replace this example with a complete,
reader-understandable source description.

<a id="ref-source-example"></a>
1. Source author or organization. *Source title*. Stable locator or publication
   details.
