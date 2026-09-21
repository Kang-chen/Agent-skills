# Process narrative

Research is a sequence of attempts. A report that shows only the last one hides
the reasoning and forces the next person to repeat the dead ends. The `process`
slot carries the sequence.

## Four beats per round

Each round is written as four beats, in this order:

1. **Why we changed something.** The problem left by the previous round, stated
   as an observation with its number.
2. **What exactly changed.** The executable difference: which component, which
   parameter, which input, which code path. Someone should be able to
   re-implement the change from this sentence.
3. **What the numbers did.** The metric, its uncertainty, and the comparison
   against the previous round on the same units.
4. **What we learned.** What the round rules in, rules out, or leaves open, and
   what it implies for the next change.

A round that failed is written the same way and kept. A change that was tried
but never quantified is labelled "未量化" with the reason.

## Shape

Give the reader the shape of the sequence before the details: a compact table of
rounds (round, motivation, change, headline number, verdict), then the beats. A
timeline or ladder Figure works well here when there are more than three rounds.

## What belongs in the narrative

- Decisions frozen before running, and when they were frozen. Post-hoc analyses
  are labelled post-hoc.
- Engineering failures that changed the science: a preprocessing bug, a
  mismatched key, a leaked split. Say what was wrong, how it was found, and what
  was re-run.
- Comparators and baselines, with the reason each was chosen.
- Abandoned branches and the criterion that abandoned them.

Engineering recovery is not a modelling result. Record it as engineering, and do
not let a fixed bug become a claimed improvement.

## When the outcome is null

A flat or negative result makes the narrative more important, not less. The
report then owes the reader three specific things:

1. **The detection limit.** What difference this design could have found, given
   the sample size and the evaluation protocol.
2. **The diagnosis.** Which of these the evidence supports: the added
   information is absent, present but not reachable by this readout, present but
   redundant with an existing feature, or destroyed by processing. Cite the
   diagnostic Figure that separates them.
3. **The next test.** The cheapest experiment that would distinguish the
   remaining explanations, and what it would cost.

"没有证据" is not "证明没有". State which one the evidence supports.
