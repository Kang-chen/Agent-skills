---
name: jupyter-notebook-ops
description: >
  Comprehensive Jupyter notebook operations: editing cells (add/delete/reorder/modify), reading
  notebooks, executing notebooks, format conversion. Use when working with .ipynb files. Supports
  Cursor EditNotebook tool, Jupytext workflows, nbformat programmatic edits, papermill execution,
  and nbconvert export. Trigger on notebook edits, cell operations, notebook execution, or format
  conversion tasks.
---

# Jupyter Notebook Operations

Comprehensive guide for Jupyter notebook manipulation across multiple workflows and tools.

## Quick Decision Tree

```
Need to edit a notebook?
├─ In Cursor IDE with EditNotebook tool available?
│   ├─ Small edit in single existing cell → Use EditNotebook directly
│   └─ Add/delete/reorder cells → Use EditNotebook (set is_new_cell appropriately)
│
├─ Structural changes (add/delete/reorder cells)?
│   └─ Use Jupytext workflow (Workflow A)
│
├─ Small edit in single existing cell?
│   └─ Use nbformat micro-edit (Workflow B)
│
├─ Batch processing multiple notebooks?
│   └─ Use nbformat programmatic approach (Workflow C)
│
└─ Execute notebook and capture outputs?
    └─ Use papermill (Workflow D)
```

---

## Workflow A: Cursor EditNotebook Tool (Cursor IDE Only)

Use the `EditNotebook` tool when available in Cursor IDE. This is the **preferred method** for
Cursor users as it handles notebook JSON complexity automatically.

### Key Parameters

| Parameter | Required | Description |
|-----------|----------|-------------|
| `target_notebook` | Yes | Path to .ipynb file |
| `cell_idx` | Yes | 0-based cell index |
| `is_new_cell` | Yes | `true` = create new cell, `false` = edit existing |
| `cell_language` | Yes | `python`, `markdown`, `r`, `sql`, `shell`, `raw`, `other` |
| `old_string` | Yes | Text to replace (empty for new cells) |
| `new_string` | Yes | Replacement/new content |

### Creating New Cells

```
EditNotebook(
  target_notebook="analysis.ipynb",
  cell_idx=5,              # Insert at position 5
  is_new_cell=true,
  cell_language="python",
  old_string="",           # Empty for new cells
  new_string="import pandas as pd\ndf = pd.read_csv('data.csv')"
)
```

### Editing Existing Cells

```
EditNotebook(
  target_notebook="analysis.ipynb",
  cell_idx=3,
  is_new_cell=false,
  cell_language="python",
  old_string="# Original comment\nx = 1",    # Include 3-5 lines context
  new_string="# Updated comment\nx = 10"
)
```

### Critical Rules for EditNotebook

1. **Always set `is_new_cell` correctly** - wrong value causes failures
2. **Include sufficient context in `old_string`** - at least 3-5 lines before/after the change
3. **`old_string` must be unique** within the cell
4. **Cell indices are 0-based**
5. **Cannot delete cells** - only clear content by setting `new_string=""``
6. **Markdown cells may render as "raw"** - this is expected behavior

---

## Workflow B: Jupytext (Structural Edits)

Use Jupytext for adding, deleting, or reordering cells. Operate through percent-format `.py` files.

### Setup (One-Time)

Pair notebook for bidirectional sync:

```bash
python -m jupytext --set-formats ipynb,py:percent notebook.ipynb
```

### Preflight: Sync Before Reading

**Always sync before reading/editing `.py`:**

```bash
# Preferred (if paired)
python -m jupytext --sync notebook.ipynb

# Fallback (manual export)
python -m jupytext --to py:percent notebook.ipynb -o notebook.py
```

### Edit Cells in `.py`

Percent-format syntax:

```python
# %% [markdown]
# # Section Title
# This is a markdown cell.

# %%
import pandas as pd
df = pd.read_csv("data.csv")

# %% tags=["parameters"]
# Cell with tags
param1 = "default"

# %% [raw]
Raw cell content here
```

### Sync Back to Notebook

```bash
# Preserve existing outputs
python -m jupytext --to ipynb --update notebook.py -o notebook.ipynb

# Clear outputs (regenerate later)
python -m jupytext --to ipynb notebook.py -o notebook.ipynb
```

### Do / Don't

- **Do**: Sync `.ipynb` → `.py` before reading `.py`
- **Do**: Use `--update` to preserve outputs
- **Don't**: Hand-edit `.ipynb` JSON for structural changes

---

## Workflow C: nbformat (Micro-Edits & Batch Processing)

Use nbformat for small in-cell edits or programmatic batch operations.

### Single Cell Micro-Edit

```python
import nbformat

path = "notebook.ipynb"
nb = nbformat.read(path, as_version=4)

# Edit specific cell (0-indexed)
cell_idx = 3
nb["cells"][cell_idx]["source"] = nb["cells"][cell_idx]["source"].replace(
    "old_text", "new_text", 1
)

nbformat.write(nb, path)
```

### Add New Cell

```python
import nbformat

nb = nbformat.read("notebook.ipynb", as_version=4)

# Create new code cell
new_cell = nbformat.v4.new_code_cell(source="print('Hello World')")

# Insert at position 5
nb["cells"].insert(5, new_cell)

nbformat.write(nb, "notebook.ipynb")
```

### Delete Cell

```python
import nbformat

nb = nbformat.read("notebook.ipynb", as_version=4)
del nb["cells"][3]  # Delete cell at index 3
nbformat.write(nb, "notebook.ipynb")
```

### Create New Markdown Cell

```python
import nbformat

md_cell = nbformat.v4.new_markdown_cell(source="# Section Header\n\nDescription text.")
nb["cells"].append(md_cell)
```

### Batch Process Multiple Notebooks

```python
import nbformat
from pathlib import Path

for nb_path in Path(".").glob("**/*.ipynb"):
    if ".ipynb_checkpoints" in str(nb_path):
        continue
    
    nb = nbformat.read(nb_path, as_version=4)
    
    # Apply transformations
    for cell in nb["cells"]:
        if cell["cell_type"] == "code":
            cell["source"] = cell["source"].replace("old_import", "new_import")
    
    nbformat.write(nb, nb_path)
```

### After nbformat Edits: Sync to Jupytext

If using Jupytext workflow, sync changes back:

```bash
python -m jupytext --sync notebook.ipynb
# or
python -m jupytext --to py:percent notebook.ipynb -o notebook.py
```

---

## Workflow D: Execute Notebooks (papermill)

Use papermill to execute notebooks with parameters and capture outputs.

### Basic Execution

```bash
papermill input.ipynb output.ipynb
```

### With Parameters

```bash
papermill input.ipynb output.ipynb -p data_path "/path/to/data" -p n_samples 1000
```

### Python API

```python
import papermill as pm

pm.execute_notebook(
    "input.ipynb",
    "output.ipynb",
    parameters={"data_path": "/path/to/data", "n_samples": 1000},
    kernel_name="python3"
)
```

### Parameterize Cells

Tag cells with `parameters` in notebook metadata or Jupytext:

```python
# %% tags=["parameters"]
data_path = "/default/path"
n_samples = 100
```

---

## Workflow E: Format Conversion (nbconvert)

### Convert to HTML

```bash
jupyter nbconvert --to html notebook.ipynb
```

### Convert to PDF

```bash
jupyter nbconvert --to pdf notebook.ipynb
```

### Convert to Python Script

```bash
jupyter nbconvert --to script notebook.ipynb
```

### Execute and Convert

```bash
jupyter nbconvert --execute --to html notebook.ipynb
```

### Available Formats

| Format | Flag | Output |
|--------|------|--------|
| HTML | `--to html` | Standalone HTML |
| PDF | `--to pdf` | PDF (requires LaTeX) |
| LaTeX | `--to latex` | .tex file |
| Markdown | `--to markdown` | .md file |
| Python | `--to script` | .py file |
| RST | `--to rst` | reStructuredText |
| Slides | `--to slides` | reveal.js slides |

---

## Reading Notebooks

### Quick Cell Overview (Python)

```python
import nbformat

nb = nbformat.read("notebook.ipynb", as_version=4)

for i, cell in enumerate(nb["cells"]):
    cell_type = cell["cell_type"]
    preview = cell["source"][:80].replace("\n", " ")
    print(f"[{i}] {cell_type:10} | {preview}...")
```

### List Cells with Line Counts

```python
for i, cell in enumerate(nb["cells"]):
    lines = cell["source"].count("\n") + 1
    print(f"Cell {i}: {cell['cell_type']} ({lines} lines)")
```

### Extract All Code Cells

```python
code_cells = [c["source"] for c in nb["cells"] if c["cell_type"] == "code"]
```

---

## Cell Metadata Operations

### Add/Modify Cell Tags

```python
cell = nb["cells"][idx]
cell["metadata"]["tags"] = cell["metadata"].get("tags", [])
cell["metadata"]["tags"].append("skip-execution")
```

### Mark Cell as Hidden (nbextensions)

```python
cell["metadata"]["hide_input"] = True
```

### Set Cell ID (nbformat v4.5+)

```python
import uuid
cell["id"] = str(uuid.uuid4())[:8]
```

---

## Common Issues & Solutions

### Issue: Outputs Lost After Jupytext Sync

**Solution**: Always use `--update` flag:

```bash
python -m jupytext --to ipynb --update notebook.py -o notebook.ipynb
```

### Issue: Cell Execution Order Confused

**Solution**: Clear outputs and re-execute in order:

```bash
jupyter nbconvert --clear-output --inplace notebook.ipynb
```

### Issue: Kernel Not Found

**Solution**: List available kernels and specify:

```bash
jupyter kernelspec list
papermill input.ipynb output.ipynb -k python3
```

### Issue: Large Notebook File Size

**Solution**: Clear outputs before committing:

```bash
jupyter nbconvert --ClearOutputPreprocessor.enabled=True --inplace notebook.ipynb
```

### Issue: Encoding Errors

**Solution**: Ensure UTF-8:

```python
nb = nbformat.read("notebook.ipynb", as_version=4)
nbformat.write(nb, "notebook.ipynb")  # Re-writes with proper encoding
```

### Issue: Windows Path Issues

- Use raw strings: `r"C:\path\to\notebook.ipynb"`
- Or forward slashes: `"C:/path/to/notebook.ipynb"`
- Venv activation: `.\.venv\Scripts\activate` instead of `source`

---

## Best Practices

1. **Version Control**: Clear outputs before committing to git
2. **Reproducibility**: Use papermill for parameterized execution
3. **Large Notebooks**: Split into multiple focused notebooks
4. **Collaboration**: Use Jupytext for text-based diffs
5. **CI/CD**: Execute with papermill/nbconvert in pipelines
6. **Cell Order**: Keep imports at top, run cells sequentially

---

## Quick Reference

| Task | Tool | Command/Method |
|------|------|----------------|
| Edit cell (Cursor) | EditNotebook | Tool call with parameters |
| Add/delete/reorder cells | Jupytext | `jupytext --sync` + edit `.py` |
| Small in-cell edit | nbformat | `nb["cells"][i]["source"] = ...` |
| Execute notebook | papermill | `papermill in.ipynb out.ipynb` |
| Convert format | nbconvert | `jupyter nbconvert --to html` |
| Clear outputs | nbconvert | `--ClearOutputPreprocessor.enabled=True` |
| Batch process | nbformat | Loop over `.glob("**/*.ipynb")` |
