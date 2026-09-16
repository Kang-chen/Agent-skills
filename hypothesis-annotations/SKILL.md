---
name: hypothesis-annotations
description: >-
  Read Hypothes.is annotations (web comments/highlights) via the Hypothesis API,
  carry out requested annotation tasks, and reply to each annotation.
  Use this whenever the user pastes a Hypothesis link (hypothes.is/a/ID, hyp.is/ID,
  or a hypothes.is/groups/PUBID group link) OR pastes any document URL and asks to
  see its comments / annotations / 批注 / 评论 / notes / highlights, OR asks to pull
  every annotation from a Hypothesis group. Handles three cases automatically: a single
  annotation link reads that one; a document URL lists all comments on that page
  (public + the user's private groups); a group link dumps the whole group. Reads
  private/group annotations that the public API returns 404 for, because it authenticates
  with the user's token. Trigger even when the user does not say the word "Hypothesis" but
  clearly refers to comments left on a hosted document (e.g. a localhost markdown page).
---

# Hypothesis Annotations

Read Hypothes.is annotations — the highlights and margin comments people leave on web
pages and hosted documents — through the official Hypothesis API (`https://api.hypothes.is/api`).

Why a skill instead of a plain `curl`: annotations in a **private group** or marked
**private** return `404` from the public unauthenticated API. They are only readable by a
**token that belongs to a member of that group**. The public `WebFetch` tool also cannot
send an `Authorization` header, so authenticated reads must go through `curl`/this script.
This skill wraps the auth, the URI encoding, the group scoping, and the extraction of the
two things the user actually wants: the **comment body** and the **highlighted source text**.

## Prerequisites: the token

The script reads a Hypothesis **developer token** from the environment, preferring
`H_TOKEN`, then `HYPOTHESIS_API_TOKEN`. An optional default group pubid can live in
`HYPOTHESIS_GROUP`. If those env vars are unset, the script falls back to reading
`~/.config/hypothesis/token.env` (the user's own chmod-600 file) directly — this matters
because agents run non-interactive shells where `~/.bashrc`'s interactive guard skips
exported vars, so relying on the shell to export the token is unreliable. Because of this
fallback you normally do **not** need to source or export anything before calling the script.

If the user has no token anywhere, tell them to generate one at
`https://hypothes.is/account/developer` (one personal token per account) and either export
`H_TOKEN='6879-...'` or write that same `export H_TOKEN='...'` line into
`~/.config/hypothesis/token.env`. Never print the token value back to the user or write it
into any file or report — refer to it only as `$H_TOKEN`.

## The one tool: `scripts/hyp.py`

Everything runs through one dependency-free script (Python stdlib only). It auto-detects
what the user pasted, so in the common case you just pass the pasted string straight through.

```bash
python scripts/hyp.py "<pasted link or URL>"
```

### Case 1 — user pastes a single annotation link (read that one)

Recognizes `https://hypothes.is/a/<id>`, `https://hyp.is/<id>/<url>`, or a bare annotation id.

```bash
python scripts/hyp.py "https://hypothes.is/a/b8aWFn6SEfGZNbs0QySbCQ"
# or force it:
python scripts/hyp.py --mode annotation b8aWFn6SEfGZNbs0QySbCQ
```

### Case 2 — user pastes a document URL and asks "看这个文档的 comment" (the default case)

This is the most common request. Pass the document URL; the script searches that exact URI
and lists every annotation the user can see (public **and** their private groups in one call).

```bash
python scripts/hyp.py "http://localhost:8642/.claude/worktrees/CellPA-crossdomain/claudedocs/x1-1/rerun_provenance_and_plan_20260710.md"
# scope to one group only:
python scripts/hyp.py --group A3inoP1m "http://localhost:8642/.../doc.md"
```

Match the URL **exactly** as the annotator's browser saw it — scheme (`http` vs `https`),
host, path, and trailing slash all matter for URI matching. Paste the URL the user gives you
verbatim; do not "clean it up".

### Case 3 — user wants the whole group

Recognizes a `https://hypothes.is/groups/<pubid>/<slug>` link, or force it with `--mode group`.

```bash
python scripts/hyp.py "https://hypothes.is/groups/A3inoP1m/cellpa"
python scripts/hyp.py --mode group --group A3inoP1m
```

### Helpers

```bash
python scripts/hyp.py --whoami    # verify which account the token belongs to
python scripts/hyp.py --groups    # list the pubid/name/type of the user's groups
```

Use `--groups` when you need a group's `pubid` (the short code like `A3inoP1m`) and the user
only gave you a group name.

## Output

By default the script prints, per annotation: `id`, `group`, `author`, `created`, the
`comment` body, each `highlight` (the exact source text the comment is anchored to), and an
`incontext` link back to the annotation. Add `--json` to get raw annotation objects when you
need structured data or fields beyond the defaults. Add `--limit N` to cap how many are
fetched (default 1000; group dumps paginate automatically via a `search_after` cursor).

## Reading the result for the user

The two fields that carry meaning are `comment` (what the person wrote) and `highlight`
(what passage they were reacting to). When you relay annotations, pair them: quote the
highlighted passage, then the comment on it, so the user sees the context without reopening
the document. A highlight with an empty `comment` is a pure highlight (no note) — say so
rather than implying there was text.

## Completing annotation tasks and replying

When the user asks to act on provided annotations, completion includes both carrying out
the requested work and replying individually to **every annotation in scope**. A final
chat summary or one combined reply does not replace replies in the annotation threads.
For a read-only request to list or summarize annotations, keep the operation read-only.

1. Inventory the annotations by ID and link before making changes. Track task status and
   reply status separately for each ID; do not silently omit items because retrieval was
   capped or paginated. Keep the scope to the annotations the user supplied or requested.
2. Complete and verify each requested change. Each reply should directly answer its
   annotation, state what changed, and point to the relevant document section, file, or
   evidence. For blocked or partially completed work, explain the limitation and remaining
   work honestly. Pure highlights and annotations needing no change still receive a brief,
   context-specific acknowledgment; do not invent a task or claim an unmade change.
3. Post a separate reply in each original Hypothesis thread when sending replies is
   authorized. The user's instruction to complete annotations and reply to every one is
   authorization for those replies; do not ask again. If the session has no authorization
   to send messages, finish the work and prepare every reply before requesting permission.
4. `scripts/hyp.py` currently reads annotations only. Use an available write-capable
   Hypothesis tool or the official API for posting, checking its current reply schema first.
   Preserve the parent thread and group visibility. Verify each posted reply and record
   its returned ID/link. After an ambiguous failure, check the thread before retrying to
   avoid duplicate replies. Never expose the token.
5. Before closing, reconcile every annotation ID with its task outcome and verified reply.
   Report how many were handled and how many replies were posted. If posting is blocked,
   provide the unsent replies by annotation ID/link and explain the blocker; drafts are
   not sent replies, and the workflow remains incomplete until replies are posted.

## Notes and gotchas

- **404 from the public API means "private or not yours", not "deleted".** If a single-id
  read 404s, the token likely is not a member of that annotation's group; try `--whoami` to
  confirm the account, and `--groups` to confirm membership.
- **URI must match byte-for-byte.** A document served at `http://localhost:8642/...` is a
  different URI from an `https://` or trailing-slash variant; a search on the wrong form
  returns nothing. When a doc search comes back empty but you expect comments, re-check the
  exact URL the annotator used.
- **Group `pubid` vs group name.** The API uses the short `pubid` (e.g. `A3inoP1m`), not the
  human name ("CellPA"). Resolve names to pubids with `--groups`.
