# Claims and review

Register report-critical takeaways, main Figure/Table observations,
decision-critical numbers, negative/null/inconclusive results, excluded
hypotheses, decisions, and disputed or restricted-source claims in
`evidence.tsv`. Reuse one record ID when prose repeats the same claim. Do not
create a new row for every sentence.

## Observation, interpretation, decision

Separate what was measured from what it means and what action follows.

- An observation states a verified artifact, value, direction, or test result.
- An interpretation names a bounded meaning, alternative, and claim boundary.
- A decision records its criterion and the locator where the decision is made.
- A hypothesis records supported, not-supported, open, ruled-out, or
  inconclusive disposition without turning absence of evidence into evidence
  of absence.

Use `supported_by_ids` from the current row to upstream evidence. Every bounded
interpretation or terminal decision needs a path to a source-grounded verified
observation. Preserve exploratory and unresolved rows outside terminal-critical
scope and pair them with an active open item. Do not mark disputed or
insufficient evidence as verified to close the report.

## Claim ceilings

State the strongest claim the design supports. Descriptive analyses support
descriptive statements. Associations support bounded association claims.
Causal, mechanistic, clinical, generalization, or recommendation claims require
the corresponding design and explicit science review. Always record the tested
population, analysis unit, endpoint or metric, comparison, uncertainty, and
important alternatives.

For negative and null results, state the tested design and detection boundary.
“No evidence under this analysis” is not “no biological or technical effect
exists.” For excluded hypotheses, record the criterion, evidence, and boundary
that justified exclusion.

## Separate review channels

1. **Claim coverage:** compare Summary, main Results, conclusion, and
   decision-critical captions against evidence IDs. Record missing locators.
2. **Science review:** assess design, analysis unit, leakage, statistics,
   alternatives, claim level, and null interpretation. Require it for
   report-critical causal/mechanistic/clinical/generalization/recommendation or
   disputed claims.
3. **Fresh-reader review:** check that a reader can find the question, answer,
   evidence, inspection path, limits, and status.
4. **Visual review:** inspect actual main Figures and record the viewed files.
5. **Style review:** after evidence and claims are frozen, apply `stop-slop`.
   Preserve numbers, direction, hedging, boundaries, limitations, terminology,
   and evidence locators; record any semantic change and repair it.

The deterministic validator checks structure, joins, paths, and declared review
state. It cannot decide whether evidence is scientifically sufficient, the
statistics are appropriate, the Figure is persuasive, or the report is ready
for publication.
