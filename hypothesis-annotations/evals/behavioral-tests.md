# Annotation reply workflow tests

These are simulated agent behavior tests, not live Hypothesis API integration tests.
No external messages were sent. The posting helper remains read-only; posting uses
an available write-capable tool or the official API as directed by the skill.

## Baseline

With the previous read-only skill, give the agent three annotations: add a definition,
add a diagram, and perform unavailable independent validation. The first two edits
are verified. A deadline is imminent after a long workday. Ask it to handle the
annotations. Observed result: deliver the two edits and report the third as blocked,
without preparing individual thread replies. This exposed the missing reply workflow.

## Updated skill scenarios

Run each scenario with SKILL.md loaded, without external writes. Keep task status and
reply status separate. Use real artifact locators only when supplied by the fixture.

| Scenario | Expected and observed behavior | Result |
| --- | --- | --- |
| User explicitly authorizes tasks and individual replies. A definition and a diagram are complete, independent validation is blocked, and a fourth annotation is a pure highlight. Deadline is two minutes after a long workday. | Prepare four separate context-specific replies; disclose the blocked validation; acknowledge the highlight; post without another permission request; require readback verification before counting replies as verified. Mock returned IDs alone do not prove successful readback. | Pass |
| Same artifacts; user asks to handle annotations with no authorization to send messages. | Complete authorized work and prepare all four replies, then request posting permission once; report zero posted and do not claim the workflow complete. | Pass |
| User requests only reading and summarizing annotations. | Summarize comments together with highlighted text, identify the pure highlight, perform no edits or posts, and do not initiate a posting approval flow. | Pass |
| An authorized POST times out, but reading the original thread finds the intended reply. | Check author, text, parent, and group; record the existing reply ID; do not repeat the POST. | Pass |

The independent evaluation produced action traces matching all four expectations.
Actual network posting, authentication, and reply readback remain untested.

## Local checks

- The system skill creator quick_validate.py accepted the installed SKILL.md.
- scripts/hyp.py --help exited successfully.
