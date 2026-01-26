#!/usr/bin/env python3
"""
CartaPA Embedding Analysis Script

Run complete analysis pipeline on CartaPA embeddings including:
- UMAP dimensionality reduction
- Leiden clustering
- Automatic response metric selection
- Interaction embedding computation
- Auto-annotation of niches
- Figure generation

Usage:
    ~/miniconda3/envs/cartaPA/bin/python run_analysis.py \
        --h5ad_path input.h5ad \
        --output_dir analysis_results/ \
        --auto_annotate \
        --add_interaction

Author: CartaPA Project
"""

import argparse
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import pdist
from sklearn.neighbors import NearestNeighbors

warnings.filterwarnings('ignore')

# Configure scanpy
sc.settings.set_figure_params(dpi=100, frameon=False, figsize=(5, 5))
plt.rcParams['axes.grid'] = False


def parse_args():
    parser = argparse.ArgumentParser(description='CartaPA Embedding Analysis')
    parser.add_argument('--h5ad_path', type=str, required=True,
                        help='Path to input h5ad file with CartaPA embeddings')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Output directory for results and figures')
    parser.add_argument('--sample_size', type=int, default=50000,
                        help='Number of cells to sample for analysis (default: 50000)')
    parser.add_argument('--leiden_resolution', type=float, default=0.5,
                        help='Leiden clustering resolution (default: 0.5)')
    parser.add_argument('--n_neighbors', type=int, default=15,
                        help='Number of neighbors for UMAP/interaction (default: 15)')
    parser.add_argument('--auto_annotate', action='store_true',
                        help='Enable automatic niche annotation')
    parser.add_argument('--add_interaction', action='store_true',
                        help='Add interaction embeddings')
    parser.add_argument('--celltype_col', type=str, default='cell_type',
                        help='Cell type column name (default: cell_type)')
    parser.add_argument('--slice_col', type=str, default='slice_id',
                        help='Slice ID column name (default: slice_id)')
    parser.add_argument('--save_h5ad', action='store_true',
                        help='Save processed h5ad file')
    return parser.parse_args()


def check_response_validity(adata) -> Tuple[str, bool]:
    """
    Check if node_response_prob is valid and select appropriate metric.
    
    Returns
    -------
    tuple : (metric_name, use_ratio)
    """
    if 'node_response_prob' not in adata.obs.columns:
        print("WARNING: node_response_prob not found")
        use_ratio = True
    else:
        response_probs = adata.obs['node_response_prob']
        
        # Check for constant values or all zeros
        if response_probs.std() < 1e-6:
            print(f"WARNING: node_response_prob is constant (std={response_probs.std():.6f})")
            use_ratio = True
        elif response_probs.sum() == 0:
            print("WARNING: node_response_prob is all zeros")
            use_ratio = True
        else:
            use_ratio = False
            print(f"node_response_prob is valid: range=[{response_probs.min():.3f}, {response_probs.max():.3f}]")
    
    if use_ratio:
        if 'slice_label' in adata.obs.columns:
            print("Using response_ratio from slice_label instead")
            adata.obs['response_label'] = adata.obs['slice_label'].astype(int)
            return 'response_label', True
        else:
            raise ValueError("Neither valid node_response_prob nor slice_label available")
    
    return 'node_response_prob', False


def add_interaction_embeddings(adata, n_neighbors=15, celltype_col='cell_type', 
                               slice_col='slice_id'):
    """
    Add 1-hop neighbor cell type count embeddings.
    """
    print(f"Computing interaction embeddings with {n_neighbors} neighbors...")
    
    celltypes = sorted(adata.obs[celltype_col].unique())
    celltype_to_idx = {ct: i for i, ct in enumerate(celltypes)}
    
    interaction_matrix = np.zeros((adata.n_obs, len(celltypes)))
    
    for slice_id in adata.obs[slice_col].unique():
        mask = adata.obs[slice_col] == slice_id
        slice_indices = np.where(mask)[0]
        
        if len(slice_indices) < n_neighbors + 1:
            continue
            
        coords = adata.obsm['spatial'][slice_indices]
        cell_types = adata.obs[celltype_col].iloc[slice_indices].values
        
        nn = NearestNeighbors(n_neighbors=min(n_neighbors + 1, len(coords)))
        nn.fit(coords)
        _, indices = nn.kneighbors(coords)
        
        for local_idx, global_idx in enumerate(slice_indices):
            neighbor_local_indices = indices[local_idx, 1:]
            neighbor_types = cell_types[neighbor_local_indices]
            
            for nt in neighbor_types:
                if nt in celltype_to_idx:
                    ct_idx = celltype_to_idx[nt]
                    interaction_matrix[global_idx, ct_idx] += 1
        
        # Normalize per slice
        row_sums = interaction_matrix[slice_indices].sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        interaction_matrix[slice_indices] /= row_sums
    
    adata.obsm['X_interaction'] = interaction_matrix
    adata.uns['interaction_celltypes'] = celltypes
    
    print(f"Added obsm['X_interaction'] with shape {interaction_matrix.shape}")
    return adata


def auto_annotate_niches(adata, leiden_col='leiden', celltype_col='cell_type',
                         response_col='response_label', threshold=0.3):
    """
    Automatically annotate niches based on dominant cell types and response.
    """
    print("Auto-annotating niches...")
    
    clusters = sorted(adata.obs[leiden_col].unique(), 
                      key=lambda x: int(x) if str(x).isdigit() else x)
    
    annotations = {}
    
    for cluster in clusters:
        mask = adata.obs[leiden_col] == cluster
        subset = adata.obs[mask]
        
        # Get dominant cell types
        celltype_props = subset[celltype_col].value_counts(normalize=True)
        dominant_types = celltype_props[celltype_props >= threshold].index.tolist()
        
        # Get response level
        if response_col in subset.columns:
            response_ratio = subset[response_col].mean()
            if response_ratio > 0.6:
                response_tag = "High_Response"
            elif response_ratio < 0.4:
                response_tag = "Low_Response"
            else:
                response_tag = "Mixed_Response"
        else:
            response_tag = ""
        
        # Build annotation
        if len(dominant_types) == 0:
            dominant_types = [celltype_props.index[0]]
        
        type_str = "_".join(dominant_types[:2])
        
        if response_tag:
            annotation = f"N{cluster}_{type_str}_{response_tag}"
        else:
            annotation = f"N{cluster}_{type_str}"
        
        annotations[cluster] = annotation
    
    # Add to adata
    adata.obs['niche_annotation'] = adata.obs[leiden_col].map(annotations)
    
    print(f"Added {len(annotations)} niche annotations:")
    for cluster, annotation in annotations.items():
        print(f"  Cluster {cluster}: {annotation}")
    
    return adata, annotations


def plot_composition_response(adata, group_col, celltype_col, response_col,
                              output_dir, title_suffix=''):
    """Plot composition + response combined figure."""
    print(f"Plotting composition and response for {group_col}...")
    
    df = adata.obs.copy()
    
    def smart_sort(items):
        try:
            return sorted(items, key=lambda x: int(x))
        except (ValueError, TypeError):
            return sorted(items, key=str)
    
    target_groups = smart_sort(df[group_col].dropna().unique())
    all_celltypes = smart_sort(df[celltype_col].dropna().unique())
    
    # Calculate composition matrix
    avg_comp_vecs = []
    for grp in target_groups:
        subset = df[df[group_col] == grp]
        celltype_counts = subset[celltype_col].value_counts()
        total = len(subset)
        comp_vec = np.array([celltype_counts.get(ct, 0) / total for ct in all_celltypes])
        avg_comp_vecs.append(comp_vec)
    avg_comp_vecs = np.array(avg_comp_vecs)
    
    # Calculate response ratios
    response_ratios_grp = {grp: df[df[group_col] == grp][response_col].mean() 
                          for grp in target_groups}
    response_ratios_ct = {ct: df[df[celltype_col] == ct][response_col].mean() 
                         for ct in all_celltypes}
    
    # Row clustering
    if len(target_groups) > 2:
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
    
    # Create figure
    from matplotlib.gridspec import GridSpec
    fig = plt.figure(figsize=(14, 10))
    gs = GridSpec(2, 3, height_ratios=[1, 4], width_ratios=[4, 0.3, 1.5],
                  hspace=0.05, wspace=0.1)
    
    # Top bar: cell type response
    ax_bar_top = fig.add_subplot(gs[0, 0])
    response_vals = [response_ratios_ct.get(ct, 0.5) for ct in all_celltypes]
    ax_bar_top.bar(np.arange(len(all_celltypes)), response_vals, color='steelblue', alpha=0.7)
    ax_bar_top.axhline(0.5, color='gray', lw=1, ls='--')
    ax_bar_top.set_xticks([])
    ax_bar_top.set_xlim(-0.5, len(all_celltypes) - 0.5)
    ax_bar_top.set_ylim(0, 1)
    ax_bar_top.set_ylabel('Response\nRatio', fontsize=10)
    ax_bar_top.set_title(f'Cell Type Response - {title_suffix}', fontsize=12)
    
    # Heatmap
    ax_heatmap = fig.add_subplot(gs[1, 0])
    im = ax_heatmap.imshow(ordered_comp_vecs, aspect='auto', cmap='YlOrRd')
    ax_heatmap.set_xticks(np.arange(len(all_celltypes)))
    ax_heatmap.set_xticklabels(all_celltypes, rotation=45, ha='right', fontsize=9)
    ax_heatmap.set_yticks(np.arange(len(ordered_groups)))
    ax_heatmap.set_yticklabels([str(g) for g in ordered_groups], fontsize=9)
    ax_heatmap.set_xlabel('Cell Type', fontsize=11)
    ax_heatmap.set_ylabel(title_suffix, fontsize=11)
    
    # Colorbar
    ax_cbar = fig.add_subplot(gs[1, 1])
    plt.colorbar(im, cax=ax_cbar)
    ax_cbar.set_ylabel('Proportion', fontsize=10)
    
    # Right bar: group response
    ax_bar_right = fig.add_subplot(gs[1, 2])
    response_vals_grp = [response_ratios_grp.get(grp, 0.5) for grp in ordered_groups]
    ax_bar_right.barh(np.arange(len(ordered_groups)), response_vals_grp, color='coral', alpha=0.7)
    ax_bar_right.axvline(0.5, color='gray', lw=1, ls='--')
    ax_bar_right.set_yticks([])
    ax_bar_right.set_xlim(0, 1)
    ax_bar_right.set_ylim(-0.5, len(ordered_groups) - 0.5)
    ax_bar_right.set_xlabel('Response Ratio', fontsize=10)
    ax_bar_right.invert_yaxis()
    
    plt.suptitle(f'Composition and Response Analysis - {title_suffix}', fontsize=14)
    
    save_path = output_dir / f'composition_response_{group_col}.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def plot_spatial_distribution(adata, color_by, output_dir, slice_col='slice_id', 
                              n_slices=6):
    """Plot spatial distribution for selected slices."""
    print(f"Plotting spatial distribution for {color_by}...")
    
    slice_ids = adata.obs[slice_col].unique()[:n_slices]
    ncols = 3
    nrows = int(np.ceil(len(slice_ids) / ncols))
    
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 5 * nrows))
    axes = np.atleast_2d(axes).flatten()
    
    is_categorical = adata.obs[color_by].dtype == 'category' or \
                     adata.obs[color_by].dtype == 'object'
    
    if is_categorical:
        categories = adata.obs[color_by].unique()
        cmap = plt.cm.get_cmap('tab20', len(categories))
        cat_to_color = {cat: cmap(i) for i, cat in enumerate(categories)}
    
    for idx, slice_id in enumerate(slice_ids):
        ax = axes[idx]
        mask = adata.obs[slice_col] == slice_id
        slice_adata = adata[mask]
        
        coords = slice_adata.obsm['spatial']
        x, y = coords[:, 0], coords[:, 1]
        
        if is_categorical:
            colors = [cat_to_color[v] for v in slice_adata.obs[color_by]]
            ax.scatter(x, y, c=colors, s=1, alpha=0.7)
        else:
            values = slice_adata.obs[color_by].values
            scatter = ax.scatter(x, y, c=values, cmap='RdYlBu_r', s=1, alpha=0.7, vmin=0, vmax=1)
            plt.colorbar(scatter, ax=ax, shrink=0.8)
        
        ax.set_title(f'{slice_id}', fontsize=10)
        ax.set_aspect('equal')
        ax.axis('off')
    
    for idx in range(len(slice_ids), len(axes)):
        axes[idx].axis('off')
    
    plt.suptitle(f'Spatial Distribution: {color_by}', fontsize=14)
    plt.tight_layout()
    
    save_path = output_dir / f'spatial_{color_by}.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def plot_interaction_difference(adata, group_col, output_dir, celltype_col='cell_type'):
    """Plot interaction embedding differences between response groups."""
    if 'X_interaction' not in adata.obsm:
        print("Skipping interaction difference plot (no X_interaction)")
        return
    
    print("Plotting interaction differences...")
    
    celltypes = adata.uns.get('interaction_celltypes', 
                              adata.obs[celltype_col].unique().tolist())
    
    # Get unique response values
    if 'response_label' in adata.obs.columns:
        response_col = 'response_label'
    else:
        response_col = 'slice_label'
    
    values = sorted(adata.obs[response_col].unique())
    if len(values) < 2:
        print("Skipping: need at least 2 response groups")
        return
    
    val1, val2 = values[0], values[-1]
    
    def get_mean_interaction_by_celltype(mask):
        result = {}
        for ct in celltypes:
            ct_mask = mask & (adata.obs[celltype_col] == ct)
            if ct_mask.sum() > 0:
                result[ct] = adata.obsm['X_interaction'][ct_mask.values].mean(axis=0)
            else:
                result[ct] = np.zeros(len(celltypes))
        return pd.DataFrame(result, index=celltypes).T
    
    mask1 = adata.obs[response_col] == val1
    mask2 = adata.obs[response_col] == val2
    
    df1 = get_mean_interaction_by_celltype(mask1)
    df2 = get_mean_interaction_by_celltype(mask2)
    df_diff = df1 - df2
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 6))
    
    vmax = max(df1.values.max(), df2.values.max())
    
    sns.heatmap(df1, ax=axes[0], cmap='YlOrRd', vmin=0, vmax=vmax,
                annot=True, fmt='.2f', cbar_kws={'shrink': 0.8})
    axes[0].set_title(f'Response={val1}', fontsize=12)
    axes[0].set_xlabel('Neighbor Cell Type')
    axes[0].set_ylabel('Center Cell Type')
    
    sns.heatmap(df2, ax=axes[1], cmap='YlOrRd', vmin=0, vmax=vmax,
                annot=True, fmt='.2f', cbar_kws={'shrink': 0.8})
    axes[1].set_title(f'Response={val2}', fontsize=12)
    axes[1].set_xlabel('Neighbor Cell Type')
    
    vmax_diff = max(abs(df_diff.values.min()), abs(df_diff.values.max()))
    sns.heatmap(df_diff, ax=axes[2], cmap='RdBu_r', center=0,
                vmin=-vmax_diff, vmax=vmax_diff,
                annot=True, fmt='.2f', cbar_kws={'shrink': 0.8})
    axes[2].set_title(f'Difference ({val1} - {val2})', fontsize=12)
    axes[2].set_xlabel('Neighbor Cell Type')
    
    plt.suptitle('Interaction Embedding Comparison by Response', fontsize=14)
    plt.tight_layout()
    
    save_path = output_dir / 'interaction_difference.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def plot_umap_overview(adata, output_dir, response_col):
    """Plot UMAP overview."""
    print("Plotting UMAP overview...")
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    sc.pl.umap(adata, color='cell_type', ax=axes[0, 0], show=False,
               title='Cell Type', frameon=False)
    
    sc.pl.umap(adata, color='leiden', ax=axes[0, 1], show=False,
               title='Leiden Clusters', frameon=False)
    
    sc.pl.umap(adata, color=response_col, ax=axes[1, 0], show=False,
               title='Response', cmap='RdYlBu_r', vmin=0, vmax=1, frameon=False)
    
    sc.pl.umap(adata, color='slice_id', ax=axes[1, 1], show=False,
               title='Slice ID', legend_loc='none', frameon=False)
    
    plt.tight_layout()
    
    save_path = output_dir / 'umap_overview.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def generate_analysis_report(adata, output_dir, response_col, use_ratio, annotations=None):
    """Generate analysis summary report."""
    report_path = output_dir / 'analysis_report.md'
    
    with open(report_path, 'w') as f:
        f.write("# CartaPA Embedding Analysis Report\n\n")
        
        f.write("## Data Summary\n\n")
        f.write(f"- Total cells: {adata.n_obs:,}\n")
        f.write(f"- Features: {adata.n_vars}\n")
        f.write(f"- Slices: {adata.obs['slice_id'].nunique()}\n")
        f.write(f"- Cell types: {adata.obs['cell_type'].nunique()}\n")
        f.write(f"- Leiden clusters: {adata.obs['leiden'].nunique()}\n\n")
        
        f.write("## Response Metric\n\n")
        if use_ratio:
            f.write("**Note**: Using `response_ratio` from `slice_label` because:\n")
            f.write("- `node_response_prob` values are all zeros/constant\n")
            f.write("- Alternative metric computed as proportion of cells from responder slices\n\n")
        else:
            f.write("Using `node_response_prob` from model predictions.\n\n")
        
        if annotations:
            f.write("## Niche Annotations\n\n")
            f.write("| Cluster | Annotation |\n")
            f.write("|---------|------------|\n")
            for cluster, annotation in sorted(annotations.items(), 
                                              key=lambda x: int(x[0]) if str(x[0]).isdigit() else x[0]):
                f.write(f"| {cluster} | {annotation} |\n")
            f.write("\n")
        
        f.write("## Generated Figures\n\n")
        for fig_file in sorted(output_dir.glob('*.png')):
            f.write(f"- `{fig_file.name}`\n")
    
    print(f"Report saved: {report_path}")


def main():
    args = parse_args()
    
    # Setup output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    print(f"Loading data from {args.h5ad_path}...")
    adata = sc.read_h5ad(args.h5ad_path)
    print(f"Loaded: {adata.shape[0]} cells x {adata.shape[1]} features")
    
    # Check response validity
    response_col, use_ratio = check_response_validity(adata)
    
    # Sample if needed
    if adata.n_obs > args.sample_size:
        print(f"Sampling {args.sample_size} cells...")
        np.random.seed(42)
        sample_idx = np.random.choice(adata.n_obs, args.sample_size, replace=False)
        adata = adata[sample_idx].copy()
    
    # Check for embedding
    if 'X_cartapa' not in adata.obsm:
        raise ValueError("No X_cartapa embedding found in obsm")
    
    # Run UMAP and Leiden
    print("Computing neighbors and UMAP...")
    sc.pp.neighbors(adata, use_rep='X_cartapa', n_neighbors=args.n_neighbors)
    sc.tl.umap(adata)
    
    print(f"Running Leiden clustering (resolution={args.leiden_resolution})...")
    sc.tl.leiden(adata, resolution=args.leiden_resolution)
    
    # Add interaction embeddings if requested
    if args.add_interaction:
        adata = add_interaction_embeddings(adata, n_neighbors=args.n_neighbors,
                                           celltype_col=args.celltype_col,
                                           slice_col=args.slice_col)
    
    # Auto-annotate if requested
    annotations = None
    if args.auto_annotate:
        adata, annotations = auto_annotate_niches(adata, 
                                                  celltype_col=args.celltype_col,
                                                  response_col=response_col)
    
    # Generate plots
    plot_umap_overview(adata, output_dir, response_col)
    
    plot_composition_response(adata, 'leiden', args.celltype_col, response_col,
                             output_dir, title_suffix='Leiden Cluster')
    
    if args.auto_annotate and 'niche_annotation' in adata.obs.columns:
        plot_composition_response(adata, 'niche_annotation', args.celltype_col, response_col,
                                 output_dir, title_suffix='Niche')
    
    plot_spatial_distribution(adata, 'leiden', output_dir, slice_col=args.slice_col)
    plot_spatial_distribution(adata, response_col, output_dir, slice_col=args.slice_col)
    
    if args.add_interaction:
        plot_interaction_difference(adata, 'leiden', output_dir, celltype_col=args.celltype_col)
    
    # Generate report
    generate_analysis_report(adata, output_dir, response_col, use_ratio, annotations)
    
    # Save processed h5ad if requested
    if args.save_h5ad:
        h5ad_out = output_dir / 'analyzed.h5ad'
        adata.write_h5ad(h5ad_out)
        print(f"Saved processed h5ad: {h5ad_out}")
    
    print("\nAnalysis complete!")
    print(f"Results saved to: {output_dir}")


if __name__ == '__main__':
    main()
