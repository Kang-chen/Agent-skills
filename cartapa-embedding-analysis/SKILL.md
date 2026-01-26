---
name: cartapa-embedding-analysis
description: >-
  Complete CartaPA embedding analysis workflow. Use when user wants to:
  (1) extract embeddings from CartaPA models,
  (2) build AnnData with embeddings and metadata,
  (3) perform UMAP/Leiden clustering analysis,
  (4) analyze cell composition and response patterns,
  (5) visualize spatial distributions and interaction embeddings,
  (6) auto-annotate niches/clusters.
  Supports CODEX-HCC, CODEX-TNBC, IMC-TNBC, SAFE-HNSCC datasets.
---

# CartaPA Embedding Analysis Skill

Extract CartaPA embeddings and perform comprehensive spatial proteomics analysis including clustering, composition analysis, and visualization.

## Quick Start

### Step 1: Extract Embeddings

```bash
~/miniconda3/envs/purturb/bin/python \
  ~/project/CartaPA_analysis/.cursor/skills/cartapa-embedding-analysis/scripts/extract_embeddings.py \
  --model_name MODEL_NAME \
  --model_seed SEED_HASH \
  --dataset DATASET_ID \
  --output_dir OUTPUT_DIR \
  --output_name embeddings.pkl
```

### Step 2: Build AnnData with Embeddings

```bash
~/miniconda3/envs/cartaPA/bin/python \
  ~/project/CartaPA_analysis/.cursor/skills/cartapa-embedding-analysis/scripts/build_anndata.py \
  --dataset DATASET_ID \
  --embedding_path embeddings.pkl \
  --output_path output.h5ad \
  --add_interaction_embeddings  # Optional: add 1-hop interaction embeddings
```

### Step 3: Run Analysis (Auto-annotation)

```bash
~/miniconda3/envs/cartaPA/bin/python \
  ~/project/CartaPA_analysis/.cursor/skills/cartapa-embedding-analysis/scripts/run_analysis.py \
  --h5ad_path output.h5ad \
  --output_dir analysis_results/ \
  --auto_annotate \
  --generate_notebook
```

## Best Models

| Dataset | Dataset ID | Model Name | Seed |
|---------|-----------|------------|------|
| CODEX-HCC | `codex_hcc_pre` | `CellPA_INDOMAIN_L2D128DR1_on_datasets_all_merged_soft_task0_codex_hcc_pre` | `70b68adba70b0e95_seed49` |
| CODEX-TNBC | `codex_tnbc_pre` | `CellPA_COMP_L2D128DR1_on_datasets_all_merged_soft_task6_codex_tnbc_pre` | `0b4017f08804574a_seed42` |
| IMC-TNBC | `imc_tnbc_pre` | `CellPA_COMP_L2D128DR1_on_datasets_all_merged_soft_task6_imc_tnbc_pre` | `9c6b9df1cba45add_seed42` |
| SAFE-HNSCC | `safe_hnscc_pre` | `CellPA_COMP_L2D128DR1_on_datasets_all_merged_soft_task6_safe_hnscc_pre` | `5719493d46a5d788_seed44` |

## Output Structure

### AnnData Structure

```python
adata.X                              # Protein expression (N x markers)
adata.obsm['X_cartapa']              # Dense embeddings (N x 128)
adata.obsm['X_cartapa_sparse']       # Sparse embeddings (N x 128)
adata.obsm['X_interaction']          # 1-hop interaction embeddings (N x n_celltypes)
adata.obsm['spatial']                # Spatial coordinates
adata.obs['node_response_prob']      # Per-cell response probability
adata.obs['slice_label']             # Slice-level response label
adata.obs['cell_type']               # Cell type annotation
adata.obs['leiden']                  # Leiden cluster
adata.obs['niche_annotation']        # Auto-annotated niche (if enabled)
```

## Analysis Workflow

1. **Data Loading & Exploration**
   - Load h5ad, check embedding structure
   - Verify response probability validity

2. **Response Metric Selection** (IMPORTANT)
   - Check if `node_response_prob` is valid (not all zeros)
   - If invalid, use `response_ratio` from `slice_label` instead
   - See [references/response-alternatives.md](references/response-alternatives.md)

3. **Dimensionality Reduction & Clustering**
   - UMAP on CartaPA embeddings
   - Leiden clustering at multiple resolutions

4. **Composition Analysis**
   - Cell type composition heatmap by cluster
   - Response score/ratio by cluster

5. **Spatial Visualization**
   - Spatial distribution of clusters
   - Spatial distribution of response scores

6. **Interaction Embedding Analysis**
   - Add 1-hop neighbor cell type counts
   - Differential interaction patterns between groups

7. **Auto-Annotation** (Optional)
   - Annotate niches based on dominant cell types
   - Identify key spatial patterns

## Key Visualization Functions

See [references/plot-functions.md](references/plot-functions.md) for detailed function documentation:

- `plot_composition_response_v2()` - Composition + response ratio combined plot
- `plot_spatial_distribution()` - Spatial scatter plots by group
- `plot_interaction_difference()` - Interaction embedding heatmaps
- `plot_stack_bar()` - Stacked bar charts
- `plot_cluster_proportion_boxplot()` - Cluster proportion by response

## Environment Notes

| Task | Environment | Notes |
|------|-------------|-------|
| Embedding extraction | `purturb` | Fast scipy Delaunay |
| AnnData building | `cartaPA` | Full scanpy support |
| Analysis & plotting | `cartaPA` | matplotlib/seaborn |

## References

- [Analysis Workflow](references/analysis-workflow.md)
- [Plot Functions](references/plot-functions.md)
- [Response Alternatives](references/response-alternatives.md)
