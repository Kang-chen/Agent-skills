# CartaPA Analysis Plot Functions

This document contains all key plotting functions for CartaPA embedding analysis.

## Table of Contents

1. [Composition + Response Combined Plot](#1-composition--response-combined-plot)
2. [Spatial Distribution Plot](#2-spatial-distribution-plot)
3. [Interaction Embedding Analysis](#3-interaction-embedding-analysis)
4. [Stack Bar Plot](#4-stack-bar-plot)
5. [Cluster Proportion Boxplot](#5-cluster-proportion-boxplot)
6. [UMAP Overview](#6-umap-overview)

---

## 1. Composition + Response Combined Plot

Plot cell type composition heatmap with response ratio/probability bars.

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.spatial.distance import pdist

def plot_composition_response_v2(df, group_col, celltype_col='cell_type',
                                  response_col='response_label',
                                  title_suffix='', cluster_rows=True,
                                  save_path=None):
    """
    Plot cell composition and response ratio combined figure.
    
    Suitable for binary response labels (when node_response_prob is invalid).
    
    Parameters
    ----------
    df : DataFrame
        Must contain group_col, celltype_col, and response_col
    group_col : str
        Grouping column (e.g., 'leiden')
    celltype_col : str
        Cell type column
    response_col : str
        Binary response column (0/1), or continuous probability
    title_suffix : str
        Title suffix for the plot
    cluster_rows : bool
        Whether to cluster rows by composition similarity
    save_path : str, optional
        Path to save figure
    
    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    plt.rcParams['axes.grid'] = False
    
    def smart_sort(items):
        try:
            return sorted(items, key=lambda x: int(x))
        except (ValueError, TypeError):
            return sorted(items)
    
    # 1. Get unique groups and cell types
    target_groups = smart_sort(df[group_col].dropna().unique())
    all_celltypes = smart_sort(df[celltype_col].dropna().unique())
    
    # 2. Calculate composition matrix
    avg_comp_vecs = []
    for grp in target_groups:
        subset = df[df[group_col] == grp]
        celltype_counts = subset[celltype_col].value_counts()
        total = len(subset)
        comp_vec = np.array([celltype_counts.get(ct, 0) / total for ct in all_celltypes])
        avg_comp_vecs.append(comp_vec)
    avg_comp_vecs = np.array(avg_comp_vecs)
    
    # 3. Calculate response ratios per group
    response_ratios_grp = {}
    for grp in target_groups:
        subset = df[df[group_col] == grp]
        if response_col in subset.columns:
            response_ratios_grp[grp] = subset[response_col].mean()
        else:
            response_ratios_grp[grp] = 0.5
    
    # 3.5. Calculate response ratios per cell type
    response_ratios_ct = {}
    for ct in all_celltypes:
        subset = df[df[celltype_col] == ct]
        if response_col in subset.columns:
            response_ratios_ct[ct] = subset[response_col].mean()
        else:
            response_ratios_ct[ct] = 0.5
    
    # 4. Get group names
    grp_name_list = [str(grp) for grp in target_groups]
    
    # 5. Row clustering order
    if cluster_rows and len(target_groups) > 2:
        try:
            distances = pdist(avg_comp_vecs, metric='euclidean')
            link = linkage(distances, method='ward')
            dendro = dendrogram(link, no_plot=True)
            row_order = dendro['leaves']
        except:
            row_order = list(range(len(target_groups)))
    else:
        row_order = list(range(len(target_groups)))
    
    ordered_groups = [target_groups[i] for i in row_order]
    ordered_comp_vecs = avg_comp_vecs[row_order]
    ordered_grp_names = [grp_name_list[i] for i in row_order]
    
    # 6. Create figure with GridSpec
    fig = plt.figure(figsize=(14, 10))
    gs = GridSpec(2, 3, height_ratios=[1, 4], width_ratios=[4, 0.3, 1.5],
                  hspace=0.05, wspace=0.1)
    
    # Top-middle: Cell type response ratio bar
    ax_bar_top = fig.add_subplot(gs[0, 0])
    response_vals = [response_ratios_ct.get(ct, 0.5) for ct in all_celltypes]
    bars = ax_bar_top.bar(np.arange(len(all_celltypes)), response_vals, 
                          color='steelblue', alpha=0.7)
    ax_bar_top.axhline(0.5, color='gray', lw=1, ls='--')
    ax_bar_top.set_xticks([])
    ax_bar_top.set_xlim(-0.5, len(all_celltypes) - 0.5)
    ax_bar_top.set_ylim(0, 1)
    ax_bar_top.set_ylabel('Response\nRatio', fontsize=10)
    ax_bar_top.set_title(f'Cell Type Response - {title_suffix}', fontsize=12)
    
    # Main heatmap: Composition
    ax_heatmap = fig.add_subplot(gs[1, 0])
    im = ax_heatmap.imshow(ordered_comp_vecs, aspect='auto', cmap='YlOrRd')
    ax_heatmap.set_xticks(np.arange(len(all_celltypes)))
    ax_heatmap.set_xticklabels(all_celltypes, rotation=45, ha='right', fontsize=9)
    ax_heatmap.set_yticks(np.arange(len(ordered_groups)))
    ax_heatmap.set_yticklabels(ordered_grp_names, fontsize=9)
    ax_heatmap.set_xlabel('Cell Type', fontsize=11)
    ax_heatmap.set_ylabel(f'{title_suffix}', fontsize=11)
    
    # Colorbar
    ax_cbar = fig.add_subplot(gs[1, 1])
    cbar = plt.colorbar(im, cax=ax_cbar)
    cbar.set_label('Proportion', fontsize=10)
    
    # Right: Group response ratio bar (horizontal)
    ax_bar_right = fig.add_subplot(gs[1, 2])
    response_vals_grp = [response_ratios_grp.get(grp, 0.5) for grp in ordered_groups]
    bars_h = ax_bar_right.barh(np.arange(len(ordered_groups)), response_vals_grp,
                               color='coral', alpha=0.7)
    ax_bar_right.axvline(0.5, color='gray', lw=1, ls='--')
    ax_bar_right.set_yticks(np.arange(len(ordered_groups)))
    ax_bar_right.set_yticklabels([])
    ax_bar_right.set_xlim(0, 1)
    ax_bar_right.set_ylim(-0.5, len(ordered_groups) - 0.5)
    ax_bar_right.set_xlabel('Response Ratio', fontsize=10)
    ax_bar_right.set_title('Group Response', fontsize=12)
    ax_bar_right.invert_yaxis()
    
    plt.suptitle(f'Composition and Response Analysis - {title_suffix}', fontsize=14, y=0.98)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved to {save_path}")
    
    return fig
```

---

## 2. Spatial Distribution Plot

Plot spatial distribution of cells colored by group or continuous variable.

```python
def plot_spatial_distribution(adata, color_by, slice_ids=None, 
                              ncols=3, point_size=1, 
                              figsize_per_subplot=(5, 5),
                              cmap='tab20', continuous_cmap='viridis',
                              save_path=None):
    """
    Plot spatial distribution of cells for selected slices.
    
    Parameters
    ----------
    adata : AnnData
        Must have obsm['spatial'] and obs columns
    color_by : str
        Column name in obs to color by
    slice_ids : list, optional
        List of slice_ids to plot. If None, plots first 6 slices
    ncols : int
        Number of columns in subplot grid
    point_size : float
        Size of scatter points
    figsize_per_subplot : tuple
        Figure size per subplot
    cmap : str
        Colormap for categorical data
    continuous_cmap : str
        Colormap for continuous data
    save_path : str, optional
        Path to save figure
    
    Returns
    -------
    fig, axes
    """
    import matplotlib.pyplot as plt
    import numpy as np
    
    # Get slice IDs
    if slice_ids is None:
        slice_ids = adata.obs['slice_id'].unique()[:6]
    
    nrows = int(np.ceil(len(slice_ids) / ncols))
    fig, axes = plt.subplots(nrows, ncols, 
                             figsize=(figsize_per_subplot[0] * ncols, 
                                      figsize_per_subplot[1] * nrows))
    axes = np.atleast_2d(axes)
    
    # Check if color_by is categorical or continuous
    is_categorical = adata.obs[color_by].dtype == 'category' or \
                     adata.obs[color_by].dtype == 'object'
    
    # Get unique categories for consistent coloring
    if is_categorical:
        categories = adata.obs[color_by].unique()
        cmap_obj = plt.cm.get_cmap(cmap, len(categories))
        cat_to_color = {cat: cmap_obj(i) for i, cat in enumerate(categories)}
    
    for idx, slice_id in enumerate(slice_ids):
        row = idx // ncols
        col = idx % ncols
        ax = axes[row, col]
        
        # Get slice data
        mask = adata.obs['slice_id'] == slice_id
        slice_adata = adata[mask]
        
        coords = slice_adata.obsm['spatial']
        x, y = coords[:, 0], coords[:, 1]
        
        if is_categorical:
            colors = [cat_to_color[v] for v in slice_adata.obs[color_by]]
            scatter = ax.scatter(x, y, c=colors, s=point_size, alpha=0.7)
        else:
            values = slice_adata.obs[color_by].values
            scatter = ax.scatter(x, y, c=values, cmap=continuous_cmap, 
                                s=point_size, alpha=0.7, vmin=0, vmax=1)
            plt.colorbar(scatter, ax=ax, shrink=0.8)
        
        ax.set_title(f'{slice_id}', fontsize=10)
        ax.set_aspect('equal')
        ax.axis('off')
    
    # Hide empty subplots
    for idx in range(len(slice_ids), nrows * ncols):
        row = idx // ncols
        col = idx % ncols
        axes[row, col].axis('off')
    
    plt.suptitle(f'Spatial Distribution: {color_by}', fontsize=14)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig, axes
```

---

## 3. Interaction Embedding Analysis

### 3.1 Add Interaction Embeddings

```python
def add_interaction_embeddings(adata, n_neighbors=15):
    """
    Add 1-hop neighbor cell type count embeddings.
    
    Creates obsm['X_interaction'] with shape (n_cells, n_celltypes)
    where each row contains normalized counts of neighbor cell types.
    
    Parameters
    ----------
    adata : AnnData
        Must have obsm['spatial'] and obs['cell_type']
    n_neighbors : int
        Number of neighbors to consider
    
    Returns
    -------
    adata : AnnData
        With added obsm['X_interaction']
    """
    from sklearn.neighbors import NearestNeighbors
    import numpy as np
    
    celltypes = adata.obs['cell_type'].unique()
    celltype_to_idx = {ct: i for i, ct in enumerate(celltypes)}
    
    # Process each slice separately
    interaction_matrix = np.zeros((adata.n_obs, len(celltypes)))
    
    for slice_id in adata.obs['slice_id'].unique():
        mask = adata.obs['slice_id'] == slice_id
        slice_indices = np.where(mask)[0]
        
        coords = adata.obsm['spatial'][slice_indices]
        cell_types = adata.obs['cell_type'].iloc[slice_indices].values
        
        # Build KNN
        nn = NearestNeighbors(n_neighbors=min(n_neighbors + 1, len(coords)))
        nn.fit(coords)
        _, indices = nn.kneighbors(coords)
        
        # Count neighbor cell types (excluding self)
        for local_idx, global_idx in enumerate(slice_indices):
            neighbor_local_indices = indices[local_idx, 1:]  # Exclude self
            neighbor_types = cell_types[neighbor_local_indices]
            
            for nt in neighbor_types:
                ct_idx = celltype_to_idx[nt]
                interaction_matrix[global_idx, ct_idx] += 1
        
        # Normalize
        row_sums = interaction_matrix[slice_indices].sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1  # Avoid division by zero
        interaction_matrix[slice_indices] /= row_sums
    
    adata.obsm['X_interaction'] = interaction_matrix
    adata.uns['interaction_celltypes'] = list(celltypes)
    
    print(f"Added obsm['X_interaction'] with shape {interaction_matrix.shape}")
    return adata
```

### 3.2 Plot Interaction Differences

```python
def plot_interaction_difference(adata, group_col, group_values,
                                celltype_col='cell_type',
                                figsize=(12, 8), save_path=None):
    """
    Plot interaction embedding differences between two groups.
    
    Shows heatmap of mean interaction patterns for each group and their difference.
    
    Parameters
    ----------
    adata : AnnData
        Must have obsm['X_interaction'] and obs[group_col]
    group_col : str
        Column to group by
    group_values : tuple of 2
        Two group values to compare (value1, value2)
    celltype_col : str
        Cell type column for row/column labels
    save_path : str, optional
        Path to save figure
    
    Returns
    -------
    fig
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np
    import pandas as pd
    
    if 'X_interaction' not in adata.obsm:
        raise ValueError("Run add_interaction_embeddings() first")
    
    celltypes = adata.uns.get('interaction_celltypes', 
                              adata.obs[celltype_col].unique().tolist())
    
    val1, val2 = group_values
    
    # Get interaction matrices for each group
    mask1 = adata.obs[group_col] == val1
    mask2 = adata.obs[group_col] == val2
    
    # Mean interaction per cell type for each group
    def get_mean_interaction_by_celltype(adata, mask, celltypes, celltype_col):
        """Get mean interaction profile for each center cell type."""
        result = {}
        for ct in celltypes:
            ct_mask = mask & (adata.obs[celltype_col] == ct)
            if ct_mask.sum() > 0:
                result[ct] = adata.obsm['X_interaction'][ct_mask].mean(axis=0)
            else:
                result[ct] = np.zeros(len(celltypes))
        return pd.DataFrame(result, index=celltypes).T
    
    df1 = get_mean_interaction_by_celltype(adata, mask1, celltypes, celltype_col)
    df2 = get_mean_interaction_by_celltype(adata, mask2, celltypes, celltype_col)
    df_diff = df1 - df2
    
    # Plot
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    
    vmax = max(df1.values.max(), df2.values.max())
    
    sns.heatmap(df1, ax=axes[0], cmap='YlOrRd', vmin=0, vmax=vmax,
                annot=True, fmt='.2f', cbar_kws={'shrink': 0.8})
    axes[0].set_title(f'Group: {val1}', fontsize=12)
    axes[0].set_xlabel('Neighbor Cell Type')
    axes[0].set_ylabel('Center Cell Type')
    
    sns.heatmap(df2, ax=axes[1], cmap='YlOrRd', vmin=0, vmax=vmax,
                annot=True, fmt='.2f', cbar_kws={'shrink': 0.8})
    axes[1].set_title(f'Group: {val2}', fontsize=12)
    axes[1].set_xlabel('Neighbor Cell Type')
    axes[1].set_ylabel('')
    
    # Difference plot
    vmax_diff = max(abs(df_diff.values.min()), abs(df_diff.values.max()))
    sns.heatmap(df_diff, ax=axes[2], cmap='RdBu_r', center=0,
                vmin=-vmax_diff, vmax=vmax_diff,
                annot=True, fmt='.2f', cbar_kws={'shrink': 0.8})
    axes[2].set_title(f'Difference ({val1} - {val2})', fontsize=12)
    axes[2].set_xlabel('Neighbor Cell Type')
    axes[2].set_ylabel('')
    
    plt.suptitle(f'Interaction Embedding Comparison: {group_col}', fontsize=14)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig
```

---

## 4. Stack Bar Plot

```python
def plot_stack_bar(adata, cluster_key, group_key, title, 
                   palette='tab20', response_col='response'):
    """
    Plot stacked bar chart of cluster composition by group.
    
    Parameters
    ----------
    adata : AnnData
        Input data
    cluster_key : str
        Column for cluster/celltype (e.g., 'celltype')
    group_key : str
        Column for grouping (e.g., 'leiden')
    title : str
        Plot title
    palette : str
        Color palette name
    response_col : str
        Response column for sorting
    
    Returns
    -------
    fig, ax
    """
    import matplotlib.pyplot as plt
    import pandas as pd
    import numpy as np
    
    # Calculate proportions
    cross = pd.crosstab(adata.obs[group_key], adata.obs[cluster_key], normalize='index')
    
    # Sort groups by response if available
    if response_col in adata.obs.columns:
        response_order = adata.obs.groupby(group_key)[response_col].mean().sort_values()
        cross = cross.loc[response_order.index]
    
    # Plot
    fig, ax = plt.subplots(figsize=(12, 6))
    
    cmap = plt.cm.get_cmap(palette, cross.shape[1])
    colors = [cmap(i) for i in range(cross.shape[1])]
    
    cross.plot(kind='bar', stacked=True, ax=ax, color=colors, width=0.8)
    
    ax.set_xlabel(group_key, fontsize=12)
    ax.set_ylabel('Proportion', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(title=cluster_key, bbox_to_anchor=(1.02, 1), loc='upper left')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
    
    plt.tight_layout()
    return fig, ax
```

---

## 5. Cluster Proportion Boxplot

```python
def plot_cluster_proportion_boxplot(adata, group_key, patient_level_key='slice_id', 
                                    response_col='slice_label', ncols=3, figsize=None):
    """
    Plot cluster proportions at patient/slice level, grouped by response.
    
    Parameters
    ----------
    adata : AnnData
        Input data
    group_key : str
        Cluster column (e.g., 'leiden')
    patient_level_key : str
        Patient/slice identifier column
    response_col : str
        Response column (0/1)
    ncols : int
        Number of columns in subplot grid
    figsize : tuple, optional
        Figure size
    
    Returns
    -------
    fig
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    import pandas as pd
    import numpy as np
    
    # Calculate proportions per patient/slice
    df = adata.obs.copy()
    
    # Get proportions
    prop_df = df.groupby([patient_level_key, group_key]).size().unstack(fill_value=0)
    prop_df = prop_df.div(prop_df.sum(axis=1), axis=0)
    
    # Add response info
    response_per_patient = df.groupby(patient_level_key)[response_col].first()
    prop_df['response'] = response_per_patient
    
    # Melt for plotting
    prop_melt = prop_df.reset_index().melt(
        id_vars=[patient_level_key, 'response'],
        var_name='cluster',
        value_name='proportion'
    )
    
    clusters = sorted(prop_df.columns[:-1].tolist(), 
                      key=lambda x: int(x) if str(x).isdigit() else x)
    nrows = int(np.ceil(len(clusters) / ncols))
    
    if figsize is None:
        figsize = (4 * ncols, 3 * nrows)
    
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    axes = np.atleast_2d(axes).flatten()
    
    for idx, cluster in enumerate(clusters):
        ax = axes[idx]
        cluster_data = prop_melt[prop_melt['cluster'] == cluster]
        
        sns.boxplot(data=cluster_data, x='response', y='proportion', ax=ax,
                    palette=['#4ECDC4', '#FF6B6B'])
        sns.stripplot(data=cluster_data, x='response', y='proportion', ax=ax,
                      color='black', size=4, alpha=0.5)
        
        ax.set_title(f'Cluster {cluster}', fontsize=10)
        ax.set_xlabel('Response')
        ax.set_ylabel('Proportion')
    
    # Hide empty axes
    for idx in range(len(clusters), len(axes)):
        axes[idx].axis('off')
    
    plt.suptitle(f'Cluster Proportions by Response ({group_key})', fontsize=14)
    plt.tight_layout()
    
    return fig
```

---

## 6. UMAP Overview

```python
def plot_umap_overview(adata, color_cols=['cell_type', 'leiden', 'slice_id'],
                       response_col='node_response_prob',
                       figsize=(14, 10), save_path=None):
    """
    Plot UMAP overview with multiple colorings.
    
    Parameters
    ----------
    adata : AnnData
        Must have obsm['X_umap']
    color_cols : list
        Columns to color UMAP by
    response_col : str
        Response column for continuous coloring
    figsize : tuple
        Figure size
    save_path : str, optional
        Path to save figure
    
    Returns
    -------
    fig
    """
    import scanpy as sc
    import matplotlib.pyplot as plt
    
    n_plots = len(color_cols) + 1  # +1 for response
    ncols = 2
    nrows = int(np.ceil(n_plots / ncols))
    
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    axes = axes.flatten()
    
    for idx, col in enumerate(color_cols):
        if col in adata.obs.columns:
            sc.pl.umap(adata, color=col, ax=axes[idx], show=False,
                       title=col, frameon=False)
    
    # Response plot
    if response_col in adata.obs.columns:
        sc.pl.umap(adata, color=response_col, ax=axes[len(color_cols)], show=False,
                   title='Response', cmap='RdYlBu_r', vmin=0, vmax=1, frameon=False)
    
    # Hide empty axes
    for idx in range(n_plots, len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig
```
