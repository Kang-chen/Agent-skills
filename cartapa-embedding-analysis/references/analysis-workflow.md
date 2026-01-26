# CartaPA Embedding Analysis Workflow

Complete workflow for analyzing CartaPA embeddings from spatial proteomics data.

## Overview

Based on the HCC analysis pipeline (Q4_kang_HCC_application.pdf):

```
1. Embedding Extraction → 2. AnnData Building → 3. UMAP/Leiden Clustering
       ↓
4. Niche Annotation (Cell Type Composition + Response Score)
       ↓
5. Per-Niche Analysis (Markers, Interactions, Spatial Distribution)
       ↓
6. Differential Analysis (High vs Low Response Groups)
```

## Step 1: Extract Embeddings

```bash
# Use purturb environment for faster extraction
~/miniconda3/envs/purturb/bin/python \
  scripts/extract_embeddings.py \
  --model_name CellPA_COMP_L2D128DR1_on_datasets_all_merged_soft_task6_imc_tnbc_pre \
  --model_seed 9c6b9df1cba45add_seed42 \
  --dataset imc_tnbc_pre \
  --output_dir data/imc_tnbc/ \
  --output_name imc_tnbc_embeddings.pkl
```

## Step 2: Build AnnData

```bash
# Use cartaPA environment
~/miniconda3/envs/cartaPA/bin/python \
  scripts/build_anndata.py \
  --dataset imc_tnbc_pre \
  --embedding_path data/imc_tnbc/imc_tnbc_embeddings.pkl \
  --output_path data/imc_tnbc/imc_tnbc.h5ad \
  --add_interaction_embeddings
```

## Step 3: UMAP and Leiden Clustering

```python
import scanpy as sc
import numpy as np

# Load data
adata = sc.read_h5ad('data/imc_tnbc/imc_tnbc.h5ad')

# Subsample for faster computation (optional)
if adata.n_obs > 50000:
    np.random.seed(42)
    sample_idx = np.random.choice(adata.n_obs, 50000, replace=False)
    adata = adata[sample_idx].copy()

# Compute neighbors using CartaPA embeddings
sc.pp.neighbors(adata, use_rep='X_cartapa', n_neighbors=15)

# Run UMAP
sc.tl.umap(adata, min_dist=0.3)

# Run Leiden at multiple resolutions
for res in [0.3, 0.5, 1.0]:
    sc.tl.leiden(adata, resolution=res, key_added=f'leiden_{res}')
    print(f"Resolution {res}: {adata.obs[f'leiden_{res}'].nunique()} clusters")

# Use optimal resolution
adata.obs['leiden'] = adata.obs['leiden_0.5']
```

## Step 4: Response Metric Selection

**IMPORTANT**: Check if `node_response_prob` is valid before analysis.

```python
def check_response_validity(adata):
    """Check and select appropriate response metric."""
    
    if 'node_response_prob' not in adata.obs.columns:
        print("WARNING: node_response_prob not found")
        use_ratio = True
    else:
        probs = adata.obs['node_response_prob']
        
        if probs.std() < 1e-6 or probs.sum() == 0:
            print(f"WARNING: node_response_prob invalid (std={probs.std():.6f})")
            use_ratio = True
        else:
            print(f"node_response_prob valid: range=[{probs.min():.3f}, {probs.max():.3f}]")
            use_ratio = False
    
    if use_ratio:
        if 'slice_label' in adata.obs.columns:
            print("Using response_ratio from slice_label")
            adata.obs['response_label'] = adata.obs['slice_label'].astype(int)
            return 'response_label', True
        else:
            raise ValueError("No valid response metric available")
    
    return 'node_response_prob', False

response_col, use_ratio = check_response_validity(adata)
```

## Step 5: Niche Annotation

### 5.1 Cell Type Composition Heatmap

```python
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Calculate composition matrix
clusters = sorted(adata.obs['leiden'].unique(), key=lambda x: int(x))
celltypes = sorted(adata.obs['cell_type'].unique())

comp_matrix = pd.DataFrame(index=clusters, columns=celltypes, dtype=float)
for cluster in clusters:
    mask = adata.obs['leiden'] == cluster
    counts = adata.obs.loc[mask, 'cell_type'].value_counts()
    total = counts.sum()
    for ct in celltypes:
        comp_matrix.loc[cluster, ct] = counts.get(ct, 0) / total

# Plot heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(comp_matrix.astype(float), annot=True, fmt='.2f', cmap='YlOrRd')
plt.xlabel('Cell Type')
plt.ylabel('Leiden Cluster')
plt.title('Cell Type Composition by Cluster')
plt.tight_layout()
plt.savefig('composition_heatmap.png', dpi=150)
```

### 5.2 Response Score by Cluster

```python
# Calculate mean response per cluster
cluster_response = adata.obs.groupby('leiden')[response_col].agg(['mean', 'std', 'count'])
cluster_response = cluster_response.sort_values('mean', ascending=False)

print("Cluster Response Scores:")
print(cluster_response)

# Bar plot
fig, ax = plt.subplots(figsize=(10, 6))
ax.bar(cluster_response.index.astype(str), cluster_response['mean'],
       yerr=cluster_response['std'], capsize=3, color='steelblue')
ax.axhline(y=0.5, color='red', linestyle='--', alpha=0.5)
ax.set_xlabel('Leiden Cluster')
ax.set_ylabel('Mean Response Score')
ax.set_title('Response Score by Cluster')
plt.tight_layout()
plt.savefig('cluster_response.png', dpi=150)
```

## Step 6: Spatial Visualization

```python
def plot_spatial_by_cluster(adata, slice_ids=None, n_slices=6):
    """Plot spatial distribution colored by leiden cluster."""
    if slice_ids is None:
        slice_ids = adata.obs['slice_id'].unique()[:n_slices]
    
    ncols = 3
    nrows = int(np.ceil(len(slice_ids) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5*ncols, 5*nrows))
    axes = np.atleast_2d(axes).flatten()
    
    for idx, slice_id in enumerate(slice_ids):
        ax = axes[idx]
        mask = adata.obs['slice_id'] == slice_id
        slice_data = adata[mask]
        
        coords = slice_data.obsm['spatial']
        ax.scatter(coords[:, 0], coords[:, 1], 
                   c=slice_data.obs['leiden'].astype(int), 
                   s=1, alpha=0.7, cmap='tab20')
        ax.set_title(slice_id)
        ax.set_aspect('equal')
        ax.axis('off')
    
    for idx in range(len(slice_ids), len(axes)):
        axes[idx].axis('off')
    
    plt.suptitle('Spatial Distribution by Cluster')
    plt.tight_layout()
    plt.savefig('spatial_clusters.png', dpi=150)

plot_spatial_by_cluster(adata)
```

## Step 7: Interaction Embedding Analysis

```python
def plot_interaction_by_response(adata, response_col):
    """Compare interaction patterns between response groups."""
    
    if 'X_interaction' not in adata.obsm:
        print("No interaction embeddings found")
        return
    
    celltypes = adata.uns['interaction_celltypes']
    
    # Split by response
    high_mask = adata.obs[response_col] >= 0.5
    low_mask = adata.obs[response_col] < 0.5
    
    # Mean interaction for each group
    high_interaction = adata.obsm['X_interaction'][high_mask.values].mean(axis=0)
    low_interaction = adata.obsm['X_interaction'][low_mask.values].mean(axis=0)
    
    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    x = np.arange(len(celltypes))
    width = 0.35
    
    axes[0].bar(x, high_interaction, width, label='High Response')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(celltypes, rotation=45, ha='right')
    axes[0].set_title('High Response Group')
    axes[0].set_ylabel('Mean Neighbor Proportion')
    
    axes[1].bar(x, low_interaction, width, label='Low Response')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(celltypes, rotation=45, ha='right')
    axes[1].set_title('Low Response Group')
    
    diff = high_interaction - low_interaction
    colors = ['red' if d > 0 else 'blue' for d in diff]
    axes[2].bar(x, diff, width, color=colors)
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(celltypes, rotation=45, ha='right')
    axes[2].set_title('Difference (High - Low)')
    axes[2].axhline(0, color='black', linestyle='-', linewidth=0.5)
    
    plt.tight_layout()
    plt.savefig('interaction_comparison.png', dpi=150)

plot_interaction_by_response(adata, response_col)
```

## Step 8: Auto-Annotation

```python
def auto_annotate_niches(adata, leiden_col='leiden', celltype_col='cell_type',
                         response_col='response_label', threshold=0.3):
    """Automatically annotate niches based on dominant cell types."""
    
    clusters = sorted(adata.obs[leiden_col].unique(), 
                      key=lambda x: int(x) if str(x).isdigit() else x)
    
    annotations = {}
    
    for cluster in clusters:
        mask = adata.obs[leiden_col] == cluster
        subset = adata.obs[mask]
        
        # Get dominant cell types
        celltype_props = subset[celltype_col].value_counts(normalize=True)
        dominant = celltype_props[celltype_props >= threshold].index.tolist()
        
        if not dominant:
            dominant = [celltype_props.index[0]]
        
        # Get response level
        if response_col in subset.columns:
            ratio = subset[response_col].mean()
            if ratio > 0.6:
                response_tag = "High"
            elif ratio < 0.4:
                response_tag = "Low"
            else:
                response_tag = "Mid"
        else:
            response_tag = ""
        
        # Build annotation
        type_str = "_".join(dominant[:2])
        annotation = f"N{cluster}_{type_str}"
        if response_tag:
            annotation += f"_{response_tag}Resp"
        
        annotations[cluster] = annotation
    
    adata.obs['niche_annotation'] = adata.obs[leiden_col].map(annotations)
    
    return adata, annotations

adata, annotations = auto_annotate_niches(adata, response_col=response_col)
print("Niche Annotations:")
for k, v in annotations.items():
    print(f"  {k}: {v}")
```

## Command Line Execution

For automated analysis:

```bash
# Full pipeline
~/miniconda3/envs/cartaPA/bin/python scripts/run_analysis.py \
  --h5ad_path data/imc_tnbc/imc_tnbc.h5ad \
  --output_dir results/imc_tnbc/ \
  --auto_annotate \
  --add_interaction \
  --save_h5ad
```

## Output Files

After running the analysis, you will have:

```
results/imc_tnbc/
├── umap_overview.png           # UMAP colored by cell type, cluster, response
├── composition_response_leiden.png  # Composition heatmap + response bars
├── spatial_leiden.png          # Spatial distribution by cluster
├── spatial_response_label.png  # Spatial distribution by response
├── interaction_difference.png  # Interaction pattern comparison
├── analysis_report.md          # Summary report
└── analyzed.h5ad               # Processed AnnData (if --save_h5ad)
```
