# Reader test

The reader test checks whether someone who reads only the brief reaches the
right judgments. Write the questions before the brief exists, so the brief is
not tailored to them.

## Questions

Write 6–10 questions with an answer key taken from the verified sources, and
save them in the package as `review/brief_reader_questions.md`. Cover:

| Kind | Example shape |
|---|---|
| Decision | Which version should be used on dataset X, and how strong is the evidence? |
| Distinction | Which arm is rule-recommended, which is the strongest candidate, which was independently confirmed? |
| Metric or split | Does the gain hold under the second split, and against which baseline? |
| Status | Can result Y be used now? When was its state observed? |
| Counter-example | A colleague claims Z. Is that right, and why not? |
| Navigation | Where is the full table for W? |
| Next | What comes first, and why? |

## Running it

1. Give a fresh reader (a new subagent with no other context) only the brief's
   text and the questions, not the answer key, sources or ledger. Allow it to
   follow links inside the package if the navigation question needs it.
2. Ask it to answer from the main layer first and to mark which answers needed
   the question layer or a link.
3. Score each answer as correct, partial, or wrong or not found, and record
   the scores in `review/brief_reader_test.md` with the reader's wording for
   every miss.
4. For each miss, decide whether the answer was hard to find, a premise was
   missing, or the brief was wrong. Fix the brief accordingly. Re-run only the
   questions that failed.

## Reporting

State who the reader was. An agent reader shows that the text can be read
correctly; it does not measure a person's reading time or understanding.
Keep any claimed speed-up or comprehension gain marked as untested until a
real reader has tried the brief.

When a long version exists, keep it in the package as comparison material and
say which version the reader saw.
