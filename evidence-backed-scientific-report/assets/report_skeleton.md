# <object> — <the core question, as a question>

Use one canonical report language. Replace every heading below with the concrete
question this report answers. Keep the `<!-- slot:... -->` markers; the quality
checker locates sections by them. Keep one explicit lowercase stable anchor
before every section used by `evidence.tsv.report_locator`, for example
`report.md#primary-answer` resolving to `<a id="primary-answer"></a>`.
Remove all instructional prose before terminal validation.

<!-- slot:tldr -->
<a id="primary-answer"></a>
## 一页纸结论

Five bullets: what we asked, what we did, what came out, why it came out that
way, what happens next. Then the single Figure a reader would keep if they kept
only one, with a caption that states its question and the interval it shows.
Support each report-critical statement with its reader-facing source reference,
for example [1](#ref-source-example).

<!-- slot:glossary -->
## 名词与代号

A table with every abbreviation, arm code, dataset name, pipeline stage, and
custom metric used anywhere in the report: name, full name, one line of what it
is and why it appears here. Common metrics do not need expansion; project
internal codes always do.

<!-- slot:background -->
## 这个任务是什么

For a reader with zero background: the clinical or scientific question, the
material, the endpoint, the unit of analysis, and why this experiment was worth
running. Carries the `concept_` setting overview Figure.

<!-- slot:inputs -->
## 输入是什么，怎么处理的

The raw material shown as real imagery or records, the cohort composition, the
unit definitions, and the value distributions before and after processing.
Carries the `data_` Figures. State every processing step that can change the
numbers.

<!-- slot:model -->
## 模型是怎么搭的

The architecture in words a reader can picture, then the compared variants side
by side: what enters each arm, where modalities join, what is frozen, what is
fitted, what the readout is. Carries the `concept_` variants Figure. Give the
shapes and the parameter counts that matter.

<!-- slot:results -->
## 结果是什么

The primary outcome with uncertainty and the paired differences, each as a
Figure. State the comparison, the unit, the sample size, and the protocol that
produced the number before interpreting it.

<!-- slot:process -->
## 我们试了什么，为什么这么改

The round-by-round narrative: a compact table of rounds, then four beats per
round — why we changed something, what exactly changed, what the numbers did,
what we learned. Failed rounds stay, with their numbers.

<!-- slot:diagnosis -->
## 为什么是这个结果

The diagnostic Figures and what they rule in or out: what the intermediate
representation encodes, how it relates to known covariates, which units fail.
For a null result, separate "information absent", "not reachable by this
readout", "redundant with an existing feature", and "destroyed by processing".

<!-- slot:limits -->
## 这个结果不支持什么

Claim boundary, detection limit for this design, unverified assumptions,
restricted or missing sources, and the alternative explanations that remain
open.

<!-- slot:next -->
## 下一步

Ranked actions. For each: what it tests, the expected gain, the cost, and what
evidence would make it unnecessary.

<!-- slot:repro -->
## 怎么复核

Exact relative paths and runnable commands with one line of purpose each: source
data, plot data, figure generation, analysis entrypoints, environment evidence.

## Appendix (optional)

Supporting detail that would otherwise obscure the main path. Keep it before
References.

<a id="references"></a>
## References

Numbered 1..N in visible order, each with the stable anchor from the manifest
source's `report_reference`.

<a id="ref-source-example"></a>
1. Source author or organization. *Source title*. Stable locator or publication
   details.
