---
name: evidence-backed-report-brief
description: Use when one or more verified scientific or technical reports must become a short presentation brief, progress update, weekly report, 汇报版, 简版 or 周报 that a collaborator can absorb in a few minutes, or when a reader says an existing report is too long to read. Not for building the underlying evidence package, single-figure plotting, citation lookup, or prose-only polishing.
---

# Evidence-Backed Report Brief

## Overview

A brief is a reading path through verified work, not a shorter copy of it. It
answers the decisions a reader has to make inside a fixed attention budget,
puts every result, limit and open question that could change those decisions
in front of the reader, and links into the full report for everything else.

The brief creates no new results. Every number, Figure and claim is taken from
a source that has already been checked. If no such source exists, build it
first with `evidence-backed-scientific-report`.

Readers usually skim the Figures and their captions. A drill-down link only helps a reader who
already knows what to look for. A limit, counter-result or unconfirmed status
that could change a decision therefore belongs in the main layer. It may not
appear only in an appendix or behind a link.

## Preflight

Record these before writing. Ask the user when one of them is unknown.

| Item | What to record |
|---|---|
| Sources | each verified report or package, its status, and its number ledger |
| Reader | who reads it and what they already know |
| Decisions | 3–6 judgments the reader must make, e.g. which version to adopt, which candidate to confirm, where the next resources go |
| Budget | main-layer reading time, default 5 minutes |
| Clock | observation time of every platform or job state you will mention |
| Canonical file | which file becomes `report.md`; where the full report lives |

Before writing the brief, write 6–10 fixed reader questions with an answer key
taken from the sources. Include each decision, at least one easily confused
distinction (recommended vs. strongest candidate vs. independently confirmed;
two metrics or splits), one counter-example claim the reader should reject, and
one "where is the evidence for X" question. See
[reader-test.md](references/reader-test.md).

## The output is three layers

Copy `assets/brief_skeleton.md` and keep its markers. The checker finds layers
and blocks by them.

**Main layer** `<!-- layer:main -->`. This is the part read in the budget. Its
contents, in this order:

1. Title: the projects, the period, and the core question.
2. A one-paragraph scope note: which sources it summarizes, their status, the
   observation time of states, where the full version is. It also defines every
   rule word the decision table relies on (confirmed, eligible, fallback) and
   names the reference each main Figure compares against.
3. One-sentence answer per project.
4. `<!-- brief:decisions -->` table with columns: judgment, current answer,
   evidence level, what result would change the answer. One row per decision.
   Phrase each judgment as the question the evidence actually tested. When the
   evidence does not settle what to use now, the answer cell says so.
5. At most two main Figures, each with a Figure card.
6. `<!-- brief:limits -->` up to five bullets: the boundaries, counter-results
   and unconfirmed states that could change a decision.
7. `<!-- brief:next -->` up to five ranked actions, each with why it comes
   first, its cost, and what result closes it.

**Question layer** `<!-- layer:questions -->`. One section per key question,
heading written as the question. Each section opens with its answer and gives one
Figure with a card, or one Table. It then lists 3–5 evidence bullets that carry
the premise needed to read each number, and ends with a deep link into the full
report section that holds the rest.

**Lookup layer** `<!-- layer:lookup -->`. `<!-- brief:glossary -->` table for
every code and abbreviation used in the brief, a table of directions tried that
failed or remain unconfirmed, and a source index with deep links.

## Evidence levels

Give every decision row and every key claim one evidence level, using the
project's own words. Define once, in the main layer, what "confirmed" means in
this project. Keep these apart:

- rule-recommended (chosen by a rule fixed in advance)
- strongest candidate (highest point estimate, possibly chosen after seeing results)
- confirmed on the same patients with new seeds or splits
- confirmed on independent data
- exploratory, registered after seeing development results
- started, platform-succeeded but not yet checked, plan only, negative

## Figure cards

Every Figure in the brief has a card directly below it:

```markdown
**图 N｜<the question this Figure answers>？** 看哪里：<the mark, row or panel to look at>。结论：<what it shows, with interval>。限制：<the boundary that matters for the decision>。下一项证据：[<what to read next>](report_full.md#anchor)。
```

English briefs use `Look at:`, `Takeaway:`, `Limit:`, `Next evidence:`.

Choose main Figures that already separate what the decision needs separated.
Do not use a Figure that:

- joins steps with a rising line or ladder when the steps were not tested in
  sequence on the same batch, which makes the gains look additive
- puts different batches, metrics or splits on one axis
- shows a post-hoc or exploratory result without that status in the card
- hides the baseline that beats or matches the headline result

Reuse verified Figures byte-identically and record source path and SHA-256.
Draw a new Figure only from existing plot data or tables. Open every Figure
you use and look at it.

## Writing rules

- Each number comes with its comparison, interval and sample size, or with a
  link to where they are given.
- Give the one-clause premise a reader needs to interpret a number (what the
  metric is, what the split is). A link alone is not an explanation.
- Every platform or job state carries its observation time.
- When an interval excludes 0, say in which direction.
- A code that means two things in the sources is qualified at every
  main-layer use, e.g. "R1 (the readout experiment)".
- When a sentence is stronger than its source, weaken the sentence.
- Remove instructional prose from the skeleton.
- After facts are frozen, apply `stop-slop`. Keep numbers, hedges and limits.

## Check

1. `python3 scripts/check_brief.py <package_root>`. Run `--help` for options
   (report file, ledger, budget, Figure caps). It checks layers and blocks,
   main-layer reading time, Figure count and cards, plot data, that every number
   is in the ledger with status `ok`, that relative links and anchors resolve,
   and that main-layer codes are in the glossary.
2. Numbers the sources do not already hold, such as a new difference, get a new
   ledger row re-derived from the source.
3. Run the fixed reader questions with a fresh reader who sees only the brief,
   and record the result in the package. See
   [reader-test.md](references/reader-test.md).
4. The checker passing says the structure is sound. It does not say the brief
   is correct or readable.

## Revisions

When a reader asks a question, first sort it into one of three kinds. If the
answer exists but is hard to find, add an anchor or an in-place link. If a
premise is missing, add one clause where the number appears. If the evidence
itself is missing, register a task under next. Do not lengthen the main layer
to answer every question.

When new results arrive, update only the rows, claims and Figures they affect,
with the new observation time, and re-run the check.

## Common mistakes

| Mistake | Fix |
|---|---|
| Brief keeps the full report's section order, only shorter | Start from the decisions; drop sections no decision needs |
| Limits gathered in an appendix | Decision-changing limits go in `brief:limits`; the rest go in the question layer |
| "Confirmed" used for same-patient reruns | Say what was rerun, and name the independent validation still missing |
| Highest point estimate presented as the recommendation | Show the rule-recommended version and the strongest candidate as two rows |
| Main layer reduced to bare numbers and file links | Add the premise clause; keep links for detail only |
| Old platform state written as current | Add the observation time, or drop the state |
| New Figure drawn to fill a slot | Reuse a verified Figure, or use a Table |
