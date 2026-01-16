---
name: jupytext-notebook-edits
description: >
  Use when reading/adding/reordering Jupyter notebook cells. Treat the Jupytext percent-format
  .py as source of truth for structural cell edits: ensure the .py is first synced/generated
  from the latest .ipynb, edit cells in .py, then sync back to .ipynb (prefer --update to
  preserve outputs). Only edit .ipynb directly for in-place changes inside a single existing
  cell (no new/deleted/reordered cells), and sync back to .py afterward.
---

# Jupytext Notebook Edits

Use this skill whenever the task involves inspecting, adding, or reorganizing notebook cells.
Operate through the percent-format `.py` via Jupytext. Avoid hand-editing `.ipynb` unless doing
a small in-cell text change without adding/removing/reordering cells.

## Preflight (always run before reading/editing `notebook.py`)
Before opening/reading the percent-format `.py`, first sync/refresh it from the latest `.ipynb`
so the agent sees the most up-to-date notebook state.

Preferred (if the notebook is already paired via Jupytext formats metadata):
- `python -m jupytext --sync notebook.ipynb`

Fallback (if pairing is not set up / `--sync` doesn’t work):
- `python -m jupytext --to py:percent notebook.ipynb -o notebook.py`

> Note: If you want reliable bidirectional sync over time, pair the notebook once:
> - `python -m jupytext --set-formats ipynb,py:percent notebook.ipynb`

---

## Workflow A: add/insert/reorder cells (preferred)
1) Preflight: ensure `.py` is synced from the latest `.ipynb`:
   - Prefer: `python -m jupytext --sync notebook.ipynb`
   - Or fallback: `python -m jupytext --to py:percent notebook.ipynb -o notebook.py`

2) Edit cells in `notebook.py` using `# %%` (code) and `# %% [markdown]` blocks.

3) Sync back to the notebook (preserve existing outputs when needed):
   - `python -m jupytext --to ipynb --update notebook.py -o notebook.ipynb`
   - Skip `--update` only if you explicitly want outputs cleared / regenerated later.

4) If outputs must be kept, never hand-edit the `.ipynb`; always re-run with `--update`.

5) Windows activation note for venvs: use `.\\.venv\\Scripts\\activate` instead of
   `source .venv/bin/activate`.

---

## Workflow B: micro-edit inside a single existing cell (no add/delete/reorder)
Use only when you are NOT adding/removing/reordering cells.

1) Load the notebook with nbformat and adjust the target cell’s `source`, keeping
   metadata/outputs intact:
   ```python
   import nbformat
   path = "notebook.ipynb"
   nb = nbformat.read(path, as_version=4)

   target = nb["cells"][CELL_INDEX]  # set CELL_INDEX
   target["source"] = target["source"].replace("old", "new", 1)

   nbformat.write(nb, path)
````

2. Do not touch `cell["outputs"]` or `cell["metadata"]` unless explicitly asked.

3. After this direct `.ipynb` micro-edit, sync the updated `.ipynb` back to `.py`
   so future structural edits still operate on the latest source:

   * Prefer (paired): `python -m jupytext --sync notebook.ipynb`
   * Or fallback: `python -m jupytext --to py:percent notebook.ipynb -o notebook.py`

4. If edits become structural (add/insert/delete/reorder), switch back to Workflow A.

---

## Do / Don’t

* Do: run Preflight sync **before reading `notebook.py`** so you operate on the latest notebook.
* Do: treat `.py` as source of truth for any cell additions, moves, or tag changes.
* Do: use `--update` when syncing `.py` → `.ipynb` to keep existing outputs.
* Don’t: hand-edit the `.ipynb` JSON except for tiny in-cell text edits without structural changes.
* Don’t: make structural cell changes via nbformat patch; use Jupytext percent-format edits instead.
