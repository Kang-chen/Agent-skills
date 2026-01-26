#!/usr/bin/env python3
"""
Generate CartaPA Analysis Jupyter Notebook

Creates a complete analysis notebook for a specified dataset with all analysis
cells and outputs. The notebook can be executed via command line using
jupyter nbconvert or papermill.

Usage:
    ~/miniconda3/envs/cartaPA/bin/python generate_notebook.py \
        --dataset imc_tnbc_pre \
        --model_name CellPA_COMP_L2D128DR1_on_datasets_all_merged_soft_task6_imc_tnbc_pre \
        --model_seed 9c6b9df1cba45add_seed42 \
        --output_notebook notebooks/imc_tnbc_analysis.ipynb

Author: CartaPA Project
"""

import argparse
import json
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description='Generate CartaPA analysis notebook')
    parser.add_argument('--dataset', type=str, required=True,
                        choices=['codex_hcc_pre', 'codex_tnbc_pre', 'imc_tnbc_pre', 'safe_hnscc_pre'],
                        help='Dataset identifier')
    parser.add_argument('--model_name', type=str, required=True, help='Model name')
    parser.add_argument('--model_seed', type=str, required=True, help='Model seed')
    parser.add_argument('--h5ad_path', type=str, default=None,
                        help='Path to existing h5ad file (skip extraction)')
    parser.add_argument('--output_notebook', type=str, required=True,
                        help='Output notebook path')
    parser.add_argument('--sample_size', type=int, default=50000,
                        help='Sample size for analysis')
    return parser.parse_args()


def create_notebook(dataset, model_name, model_seed, h5ad_path, sample_size):
    """Create notebook structure."""
    
    # Dataset display names
    dataset_names = {
        'codex_hcc_pre': 'CODEX-HCC Pre-treatment',
        'codex_tnbc_pre': 'CODEX-TNBC Pre-treatment',
        'imc_tnbc_pre': 'IMC-TNBC Pre-treatment',
        'safe_hnscc_pre': 'SAFE-HNSCC Pre-treatment',
    }
    
    display_name = dataset_names.get(dataset, dataset)
    
    cells = []
    
    # Title cell
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            f"# CartaPA Embedding Analysis: {display_name}\n",
            "\n",
            "This notebook performs comprehensive analysis of CartaPA embeddings including:\n",
            "- Data loading and exploration\n",
            "- Response metric validation (with automatic fallback to response_ratio)\n",
            "- UMAP dimensionality reduction and Leiden clustering\n",
            "- Cell type composition analysis\n",
            "- Spatial distribution visualization\n",
            "- Interaction embedding analysis\n",
            "- Auto-annotation of niches\n",
            "\n",
            f"**Model**: `{model_name}`\n",
            f"**Seed**: `{model_seed}`\n",
            f"**Dataset**: `{dataset}`"
        ]
    })
    
    # Section 1: Environment Setup
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": ["## 1. Environment Setup"]
    })
    
    cells.append({
        "cell_type": "code",
        "metadata": {},
        "source": [
            "import sys\n",
            "import warnings\n",
            "from pathlib import Path\n",
            "\n",
            "import numpy as np\n",
            "import pandas as pd\n",
            "import matplotlib.pyplot as plt\n",
            "import seaborn as sns\n",
            "import scanpy as sc\n",
            "from scipy.cluster.hierarchy import linkage, dendrogram\n",
            "from scipy.spatial.distance import pdist\n",
            "from sklearn.neighbors import NearestNeighbors\n",
            "\n",
            "warnings.filterwarnings('ignore')\n",
            "\n",
            "# Configure scanpy\n",
            "sc.settings.set_figure_params(dpi=100, frameon=False, figsize=(5, 5))\n",
            "plt.rcParams['axes.grid'] = False\n",
            "\n",
            "# Paths\n",
            f"DATASET = '{dataset}'\n",
            f"MODEL_NAME = '{model_name}'\n",
            f"MODEL_SEED = '{model_seed}'\n",
            f"SAMPLE_SIZE = {sample_size}\n",
            "\n",
            "OUTPUT_DIR = Path(f'../figures/{DATASET.replace(\"_pre\", \"\")}')\n",
            "OUTPUT_DIR.mkdir(parents=True, exist_ok=True)\n",
            "print(f'Output directory: {OUTPUT_DIR}')"
        ],
        "execution_count": None,
        "outputs": []
    })
    
    # Section 2: Data Loading
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": ["## 2. Data Loading"]
    })
    
    if h5ad_path:
        cells.append({
            "cell_type": "code",
            "metadata": {},
            "source": [
                f"# Load pre-built h5ad\n",
                f"H5AD_PATH = '{h5ad_path}'\n",
                "print(f'Loading data from {H5AD_PATH}...')\n",
                "adata = sc.read_h5ad(H5AD_PATH)\n",
                "print(f'Loaded: {adata.shape[0]} cells x {adata.shape[1]} features')"
            ],
            "execution_count": None,
            "outputs": []
        })
    else:
        dataset_short = dataset.replace("_pre", "")
        cells.append({
            "cell_type": "code",
            "metadata": {},
            "source": [
                "# Define h5ad path (adjust as needed)\n",
                f"H5AD_PATH = f'../data/{dataset_short}/{dataset_short}.h5ad'\n",
                "\n",
                "# Check if file exists\n",
                "if not Path(H5AD_PATH).exists():\n",
                f"    print(f'WARNING: {{H5AD_PATH}} not found!')\n",
                "    print('Please run embedding extraction and AnnData building first:')\n",
                f"    print(f'  python scripts/extract_embeddings.py --dataset {dataset} ...')\n",
                f"    print(f'  python scripts/build_anndata.py --dataset {dataset} ...')\n",
                "else:\n",
                "    print(f'Loading data from {H5AD_PATH}...')\n",
                "    adata = sc.read_h5ad(H5AD_PATH)\n",
                "    print(f'Loaded: {adata.shape[0]} cells x {adata.shape[1]} features')"
            ],
            "execution_count": None,
            "outputs": []
        })
    
    # Data exploration
    cells.append({
        "cell_type": "code",
        "metadata": {},
        "source": [
            "# Explore data structure\n",
            "print('=== AnnData Structure ===')\n",
            "print(f'n_obs (cells): {adata.n_obs:,}')\n",
            "print(f'n_vars (features): {adata.n_vars}')\n",
            "print(f'\\nobs columns: {list(adata.obs.columns)}')\n",
            "print(f'\\nobsm keys: {list(adata.obsm.keys())}')\n",
            "\n",
            "if 'X_cartapa' in adata.obsm:\n",
            "    print(f'\\nCartaPA embeddings shape: {adata.obsm[\"X_cartapa\"].shape}')"
        ],
        "execution_count": None,
        "outputs": []
    })
    
    cells.append({
        "cell_type": "code",
        "metadata": {},
        "source": [
            "# Check cell type and slice distributions\n",
            "print('Cell type distribution:')\n",
            "print(adata.obs['cell_type'].value_counts())\n",
            "\n",
            "print(f'\\nNumber of slices: {adata.obs[\"slice_id\"].nunique()}')\n",
            "print('\\nSlice label distribution (0=Non-responder, 1=Responder):')\n",
            "if 'slice_label' in adata.obs.columns:\n",
            "    print(adata.obs['slice_label'].value_counts())"
        ],
        "execution_count": None,
        "outputs": []
    })
    
    # Section 3: Response Metric Validation
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 3. Response Metric Validation\n",
            "\n",
            "**IMPORTANT**: Some models output `node_response_prob` values that are all zeros or constant.\n",
            "In such cases, we use `response_ratio` computed from `slice_label` as an alternative."
        ]
    })
    
    cells.append({
        "cell_type": "code",
        "metadata": {},
        "source": [
            "def check_response_validity(adata):\n",
            "    \"\"\"\n",
            "    Check if node_response_prob is valid and select appropriate metric.\n",
            "    \n",
            "    Returns: (metric_name, use_ratio, reason)\n",
            "    \"\"\"\n",
            "    reason = None\n",
            "    \n",
            "    if 'node_response_prob' not in adata.obs.columns:\n",
            "        reason = 'node_response_prob column not found'\n",
            "        use_ratio = True\n",
            "    else:\n",
            "        probs = adata.obs['node_response_prob']\n",
            "        \n",
            "        # Check for all zeros\n",
            "        if probs.sum() == 0:\n",
            "            reason = 'node_response_prob is all zeros'\n",
            "            use_ratio = True\n",
            "        # Check for constant values\n",
            "        elif probs.std() < 1e-6:\n",
            "            reason = f'node_response_prob is constant (std={probs.std():.6f})'\n",
            "            use_ratio = True\n",
            "        else:\n",
            "            use_ratio = False\n",
            "            print(f'node_response_prob is VALID')\n",
            "            print(f'  Range: [{probs.min():.4f}, {probs.max():.4f}]')\n",
            "            print(f'  Mean: {probs.mean():.4f}, Std: {probs.std():.4f}')\n",
            "    \n",
            "    if use_ratio:\n",
            "        if 'slice_label' not in adata.obs.columns:\n",
            "            raise ValueError('No valid response metric available')\n",
            "        \n",
            "        print(f'WARNING: {reason}')\n",
            "        print('Using response_ratio from slice_label as alternative')\n",
            "        adata.obs['response_label'] = adata.obs['slice_label'].astype(int)\n",
            "        return 'response_label', True, reason\n",
            "    \n",
            "    return 'node_response_prob', False, None\n",
            "\n",
            "response_col, use_ratio, reason = check_response_validity(adata)"
        ],
        "execution_count": None,
        "outputs": []
    })
    
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "### Response Metric Documentation\n",
            "\n",
            "If `node_response_prob` is invalid, the following note applies to all subsequent analysis:"
        ]
    })
    
    cells.append({
        "cell_type": "code",
        "metadata": {},
        "source": [
            "# Document the response metric used\n",
            "if use_ratio:\n",
            "    response_doc = f'''\n",
            "**Note on Response Metric**\n",
            "\n",
            "This analysis uses `response_ratio` (proportion from responder slices) instead of `node_response_prob` because:\n",
            "\n",
            "1. **Evidence**: {reason}\n",
            "2. **Alternative**: `response_ratio` computed from `slice_label` provides biologically meaningful response enrichment scores\n",
            "3. **Interpretation**: Higher ratio indicates cluster enrichment in responding patients\n",
            "\n",
            "This is a valid alternative as cells from the same tissue slice share clinical outcomes.\n",
            "'''\n",
            "    print(response_doc)\n",
            "else:\n",
            "    print('Using model-predicted node_response_prob for analysis.')"
        ],
        "execution_count": None,
        "outputs": []
    })
    
    # Section 4: Sampling and Preprocessing
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": ["## 4. Sampling and Preprocessing"]
    })
    
    cells.append({
        "cell_type": "code",
        "metadata": {},
        "source": [
            "# Subsample for faster computation\n",
            "np.random.seed(42)\n",
            "\n",
            "if adata.n_obs > SAMPLE_SIZE:\n",
            "    print(f'Sampling {SAMPLE_SIZE:,} cells from {adata.n_obs:,} total cells...')\n",
            "    sample_idx = np.random.choice(adata.n_obs, SAMPLE_SIZE, replace=False)\n",
            "    adata_sample = adata[sample_idx].copy()\n",
            "else:\n",
            "    print(f'Using all {adata.n_obs:,} cells')\n",
            "    adata_sample = adata.copy()\n",
            "\n",
            "print(f'Working with {adata_sample.n_obs:,} cells')"
        ],
        "execution_count": None,
        "outputs": []
    })
    
    # Section 5: UMAP and Clustering
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": ["## 5. UMAP and Leiden Clustering"]
    })
    
    cells.append({
        "cell_type": "code",
        "metadata": {},
        "source": [
            "# Check for embeddings\n",
            "if 'X_cartapa' not in adata_sample.obsm:\n",
            "    raise ValueError('No X_cartapa embedding found in obsm')\n",
            "\n",
            "print('Computing neighbors using CartaPA embeddings...')\n",
            "sc.pp.neighbors(adata_sample, use_rep='X_cartapa', n_neighbors=15)\n",
            "\n",
            "print('Running UMAP...')\n",
            "sc.tl.umap(adata_sample, min_dist=0.3)"
        ],
        "execution_count": None,
        "outputs": []
    })
    
    cells.append({
        "cell_type": "code",
        "metadata": {},
        "source": [
            "# Run Leiden clustering at multiple resolutions\n",
            "resolutions = [0.3, 0.5, 1.0]\n",
            "\n",
            "for res in resolutions:\n",
            "    key = f'leiden_{res}'\n",
            "    sc.tl.leiden(adata_sample, resolution=res, key_added=key)\n",
            "    n_clusters = adata_sample.obs[key].nunique()\n",
            "    print(f'Resolution {res}: {n_clusters} clusters')\n",
            "\n",
            "# Use resolution 0.5 as default\n",
            "adata_sample.obs['leiden'] = adata_sample.obs['leiden_0.5']\n",
            "print(f'\\nUsing leiden_0.5 with {adata_sample.obs[\"leiden\"].nunique()} clusters')"
        ],
        "execution_count": None,
        "outputs": []
    })
    
    # Section 6: UMAP Visualization
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": ["## 6. UMAP Visualization"]
    })
    
    cells.append({
        "cell_type": "code",
        "metadata": {},
        "source": [
            "# UMAP overview\n",
            "fig, axes = plt.subplots(2, 2, figsize=(14, 12))\n",
            "\n",
            "sc.pl.umap(adata_sample, color='cell_type', ax=axes[0, 0], show=False,\n",
            "           title='Cell Type', frameon=False)\n",
            "\n",
            "sc.pl.umap(adata_sample, color='leiden', ax=axes[0, 1], show=False,\n",
            "           title='Leiden Clusters', frameon=False)\n",
            "\n",
            "sc.pl.umap(adata_sample, color=response_col, ax=axes[1, 0], show=False,\n",
            "           title='Response', cmap='RdYlBu_r', vmin=0, vmax=1, frameon=False)\n",
            "\n",
            "sc.pl.umap(adata_sample, color='slice_id', ax=axes[1, 1], show=False,\n",
            "           title='Slice ID', legend_loc='none', frameon=False)\n",
            "\n",
            "plt.tight_layout()\n",
            "plt.savefig(OUTPUT_DIR / 'umap_overview.png', dpi=150, bbox_inches='tight')\n",
            "plt.show()"
        ],
        "execution_count": None,
        "outputs": []
    })
    
    # Section 7: Composition Analysis
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": ["## 7. Cell Type Composition Analysis"]
    })
    
    cells.append({
        "cell_type": "code",
        "metadata": {},
        "source": [
            "def plot_composition_response_v2(df, group_col, celltype_col='cell_type',\n",
            "                                  response_col='response_label',\n",
            "                                  title_suffix='', cluster_rows=True,\n",
            "                                  save_path=None):\n",
            "    \"\"\"\n",
            "    Plot cell composition and response ratio combined figure.\n",
            "    \"\"\"\n",
            "    from matplotlib.gridspec import GridSpec\n",
            "    \n",
            "    def smart_sort(items):\n",
            "        try:\n",
            "            return sorted(items, key=lambda x: int(x))\n",
            "        except (ValueError, TypeError):\n",
            "            return sorted(items, key=str)\n",
            "    \n",
            "    target_groups = smart_sort(df[group_col].dropna().unique())\n",
            "    all_celltypes = smart_sort(df[celltype_col].dropna().unique())\n",
            "    \n",
            "    # Calculate composition\n",
            "    avg_comp_vecs = []\n",
            "    for grp in target_groups:\n",
            "        subset = df[df[group_col] == grp]\n",
            "        celltype_counts = subset[celltype_col].value_counts()\n",
            "        total = len(subset)\n",
            "        comp_vec = np.array([celltype_counts.get(ct, 0) / total for ct in all_celltypes])\n",
            "        avg_comp_vecs.append(comp_vec)\n",
            "    avg_comp_vecs = np.array(avg_comp_vecs)\n",
            "    \n",
            "    # Calculate response ratios\n",
            "    response_ratios_grp = {grp: df[df[group_col] == grp][response_col].mean() \n",
            "                          for grp in target_groups}\n",
            "    response_ratios_ct = {ct: df[df[celltype_col] == ct][response_col].mean() \n",
            "                         for ct in all_celltypes}\n",
            "    \n",
            "    # Row clustering\n",
            "    if cluster_rows and len(target_groups) > 2:\n",
            "        try:\n",
            "            distances = pdist(avg_comp_vecs, metric='euclidean')\n",
            "            link = linkage(distances, method='ward')\n",
            "            dendro = dendrogram(link, no_plot=True)\n",
            "            row_order = dendro['leaves']\n",
            "        except:\n",
            "            row_order = list(range(len(target_groups)))\n",
            "    else:\n",
            "        row_order = list(range(len(target_groups)))\n",
            "    \n",
            "    ordered_groups = [target_groups[i] for i in row_order]\n",
            "    ordered_comp_vecs = avg_comp_vecs[row_order]\n",
            "    \n",
            "    # Create figure\n",
            "    fig = plt.figure(figsize=(14, 10))\n",
            "    gs = GridSpec(2, 3, height_ratios=[1, 4], width_ratios=[4, 0.3, 1.5],\n",
            "                  hspace=0.05, wspace=0.1)\n",
            "    \n",
            "    # Top bar\n",
            "    ax_bar_top = fig.add_subplot(gs[0, 0])\n",
            "    response_vals = [response_ratios_ct.get(ct, 0.5) for ct in all_celltypes]\n",
            "    ax_bar_top.bar(np.arange(len(all_celltypes)), response_vals, color='steelblue', alpha=0.7)\n",
            "    ax_bar_top.axhline(0.5, color='gray', lw=1, ls='--')\n",
            "    ax_bar_top.set_xticks([])\n",
            "    ax_bar_top.set_xlim(-0.5, len(all_celltypes) - 0.5)\n",
            "    ax_bar_top.set_ylim(0, 1)\n",
            "    ax_bar_top.set_ylabel('Response\\nRatio', fontsize=10)\n",
            "    ax_bar_top.set_title(f'Cell Type Response - {title_suffix}', fontsize=12)\n",
            "    \n",
            "    # Heatmap\n",
            "    ax_heatmap = fig.add_subplot(gs[1, 0])\n",
            "    im = ax_heatmap.imshow(ordered_comp_vecs, aspect='auto', cmap='YlOrRd')\n",
            "    ax_heatmap.set_xticks(np.arange(len(all_celltypes)))\n",
            "    ax_heatmap.set_xticklabels(all_celltypes, rotation=45, ha='right', fontsize=9)\n",
            "    ax_heatmap.set_yticks(np.arange(len(ordered_groups)))\n",
            "    ax_heatmap.set_yticklabels([str(g) for g in ordered_groups], fontsize=9)\n",
            "    ax_heatmap.set_xlabel('Cell Type', fontsize=11)\n",
            "    ax_heatmap.set_ylabel(title_suffix, fontsize=11)\n",
            "    \n",
            "    # Colorbar\n",
            "    ax_cbar = fig.add_subplot(gs[1, 1])\n",
            "    plt.colorbar(im, cax=ax_cbar)\n",
            "    ax_cbar.set_ylabel('Proportion', fontsize=10)\n",
            "    \n",
            "    # Right bar\n",
            "    ax_bar_right = fig.add_subplot(gs[1, 2])\n",
            "    response_vals_grp = [response_ratios_grp.get(grp, 0.5) for grp in ordered_groups]\n",
            "    ax_bar_right.barh(np.arange(len(ordered_groups)), response_vals_grp, color='coral', alpha=0.7)\n",
            "    ax_bar_right.axvline(0.5, color='gray', lw=1, ls='--')\n",
            "    ax_bar_right.set_yticks([])\n",
            "    ax_bar_right.set_xlim(0, 1)\n",
            "    ax_bar_right.set_ylim(-0.5, len(ordered_groups) - 0.5)\n",
            "    ax_bar_right.set_xlabel('Response Ratio', fontsize=10)\n",
            "    ax_bar_right.invert_yaxis()\n",
            "    \n",
            "    plt.suptitle(f'Composition and Response - {title_suffix}', fontsize=14)\n",
            "    \n",
            "    if save_path:\n",
            "        plt.savefig(save_path, dpi=150, bbox_inches='tight')\n",
            "    \n",
            "    return fig"
        ],
        "execution_count": None,
        "outputs": []
    })
    
    cells.append({
        "cell_type": "code",
        "metadata": {},
        "source": [
            "# Plot composition and response by Leiden cluster\n",
            "df = adata_sample.obs.copy()\n",
            "\n",
            "fig = plot_composition_response_v2(\n",
            "    df,\n",
            "    group_col='leiden',\n",
            "    celltype_col='cell_type',\n",
            "    response_col=response_col,\n",
            "    title_suffix='Leiden Cluster',\n",
            "    save_path=OUTPUT_DIR / 'composition_response_leiden.png'\n",
            ")\n",
            "plt.show()"
        ],
        "execution_count": None,
        "outputs": []
    })
    
    # Section 8: Spatial Visualization
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": ["## 8. Spatial Distribution"]
    })
    
    cells.append({
        "cell_type": "code",
        "metadata": {},
        "source": [
            "def plot_spatial_distribution(adata, color_by, slice_ids=None, n_slices=6,\n",
            "                             ncols=3, point_size=1, save_path=None):\n",
            "    \"\"\"Plot spatial distribution for selected slices.\"\"\"\n",
            "    if slice_ids is None:\n",
            "        slice_ids = adata.obs['slice_id'].unique()[:n_slices]\n",
            "    \n",
            "    nrows = int(np.ceil(len(slice_ids) / ncols))\n",
            "    fig, axes = plt.subplots(nrows, ncols, figsize=(5*ncols, 5*nrows))\n",
            "    axes = np.atleast_2d(axes).flatten()\n",
            "    \n",
            "    is_categorical = adata.obs[color_by].dtype == 'category' or \\\n",
            "                     adata.obs[color_by].dtype == 'object'\n",
            "    \n",
            "    if is_categorical:\n",
            "        categories = adata.obs[color_by].unique()\n",
            "        cmap = plt.cm.get_cmap('tab20', len(categories))\n",
            "        cat_to_color = {cat: cmap(i) for i, cat in enumerate(categories)}\n",
            "    \n",
            "    for idx, slice_id in enumerate(slice_ids):\n",
            "        ax = axes[idx]\n",
            "        mask = adata.obs['slice_id'] == slice_id\n",
            "        slice_data = adata[mask]\n",
            "        \n",
            "        if 'spatial' not in slice_data.obsm:\n",
            "            ax.text(0.5, 0.5, 'No spatial data', ha='center', va='center')\n",
            "            ax.axis('off')\n",
            "            continue\n",
            "        \n",
            "        coords = slice_data.obsm['spatial']\n",
            "        x, y = coords[:, 0], coords[:, 1]\n",
            "        \n",
            "        if is_categorical:\n",
            "            colors = [cat_to_color[v] for v in slice_data.obs[color_by]]\n",
            "            ax.scatter(x, y, c=colors, s=point_size, alpha=0.7)\n",
            "        else:\n",
            "            values = slice_data.obs[color_by].values\n",
            "            scatter = ax.scatter(x, y, c=values, cmap='RdYlBu_r', \n",
            "                                s=point_size, alpha=0.7, vmin=0, vmax=1)\n",
            "            plt.colorbar(scatter, ax=ax, shrink=0.8)\n",
            "        \n",
            "        ax.set_title(f'{slice_id[:20]}...' if len(str(slice_id)) > 20 else slice_id, fontsize=10)\n",
            "        ax.set_aspect('equal')\n",
            "        ax.axis('off')\n",
            "    \n",
            "    for idx in range(len(slice_ids), len(axes)):\n",
            "        axes[idx].axis('off')\n",
            "    \n",
            "    plt.suptitle(f'Spatial Distribution: {color_by}', fontsize=14)\n",
            "    plt.tight_layout()\n",
            "    \n",
            "    if save_path:\n",
            "        plt.savefig(save_path, dpi=150, bbox_inches='tight')\n",
            "    \n",
            "    return fig"
        ],
        "execution_count": None,
        "outputs": []
    })
    
    cells.append({
        "cell_type": "code",
        "metadata": {},
        "source": [
            "# Spatial distribution by cluster\n",
            "fig = plot_spatial_distribution(\n",
            "    adata_sample, \n",
            "    color_by='leiden',\n",
            "    save_path=OUTPUT_DIR / 'spatial_leiden.png'\n",
            ")\n",
            "plt.show()"
        ],
        "execution_count": None,
        "outputs": []
    })
    
    cells.append({
        "cell_type": "code",
        "metadata": {},
        "source": [
            "# Spatial distribution by response\n",
            "fig = plot_spatial_distribution(\n",
            "    adata_sample,\n",
            "    color_by=response_col,\n",
            "    save_path=OUTPUT_DIR / f'spatial_{response_col}.png'\n",
            ")\n",
            "plt.show()"
        ],
        "execution_count": None,
        "outputs": []
    })
    
    # Section 9: Interaction Embedding
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": ["## 9. Interaction Embedding Analysis"]
    })
    
    cells.append({
        "cell_type": "code",
        "metadata": {},
        "source": [
            "def add_interaction_embeddings(adata, n_neighbors=15, celltype_col='cell_type',\n",
            "                               slice_col='slice_id'):\n",
            "    \"\"\"Add 1-hop neighbor cell type count embeddings.\"\"\"\n",
            "    print(f'Computing interaction embeddings with {n_neighbors} neighbors...')\n",
            "    \n",
            "    celltypes = sorted(adata.obs[celltype_col].unique())\n",
            "    celltype_to_idx = {ct: i for i, ct in enumerate(celltypes)}\n",
            "    \n",
            "    interaction_matrix = np.zeros((adata.n_obs, len(celltypes)), dtype=np.float32)\n",
            "    \n",
            "    for slice_id in adata.obs[slice_col].unique():\n",
            "        mask = adata.obs[slice_col] == slice_id\n",
            "        slice_indices = np.where(mask)[0]\n",
            "        \n",
            "        if len(slice_indices) < n_neighbors + 1:\n",
            "            continue\n",
            "        \n",
            "        if 'spatial' not in adata.obsm:\n",
            "            continue\n",
            "        \n",
            "        coords = adata.obsm['spatial'][slice_indices]\n",
            "        cell_types = adata.obs[celltype_col].iloc[slice_indices].values\n",
            "        \n",
            "        nn = NearestNeighbors(n_neighbors=min(n_neighbors + 1, len(coords)))\n",
            "        nn.fit(coords)\n",
            "        _, indices = nn.kneighbors(coords)\n",
            "        \n",
            "        for local_idx, global_idx in enumerate(slice_indices):\n",
            "            neighbor_local_indices = indices[local_idx, 1:]\n",
            "            neighbor_types = cell_types[neighbor_local_indices]\n",
            "            \n",
            "            for nt in neighbor_types:\n",
            "                if nt in celltype_to_idx:\n",
            "                    ct_idx = celltype_to_idx[nt]\n",
            "                    interaction_matrix[global_idx, ct_idx] += 1\n",
            "        \n",
            "        # Normalize\n",
            "        row_sums = interaction_matrix[slice_indices].sum(axis=1, keepdims=True)\n",
            "        row_sums[row_sums == 0] = 1\n",
            "        interaction_matrix[slice_indices] /= row_sums\n",
            "    \n",
            "    adata.obsm['X_interaction'] = interaction_matrix\n",
            "    adata.uns['interaction_celltypes'] = celltypes\n",
            "    \n",
            "    print(f'Added obsm[\"X_interaction\"] with shape {interaction_matrix.shape}')\n",
            "    return adata\n",
            "\n",
            "# Add interaction embeddings\n",
            "if 'X_interaction' not in adata_sample.obsm:\n",
            "    adata_sample = add_interaction_embeddings(adata_sample)\n",
            "else:\n",
            "    print('Interaction embeddings already present')"
        ],
        "execution_count": None,
        "outputs": []
    })
    
    cells.append({
        "cell_type": "code",
        "metadata": {},
        "source": [
            "def plot_interaction_difference(adata, response_col, celltype_col='cell_type', save_path=None):\n",
            "    \"\"\"Plot interaction embedding differences between response groups.\"\"\"\n",
            "    if 'X_interaction' not in adata.obsm:\n",
            "        print('No X_interaction found')\n",
            "        return None\n",
            "    \n",
            "    celltypes = adata.uns.get('interaction_celltypes', \n",
            "                              adata.obs[celltype_col].unique().tolist())\n",
            "    \n",
            "    # Split by response\n",
            "    high_mask = adata.obs[response_col] >= 0.5\n",
            "    low_mask = adata.obs[response_col] < 0.5\n",
            "    \n",
            "    if high_mask.sum() == 0 or low_mask.sum() == 0:\n",
            "        print('Not enough cells in both groups')\n",
            "        return None\n",
            "    \n",
            "    # Mean interaction for each cell type center\n",
            "    def get_mean_interaction_by_celltype(mask):\n",
            "        result = {}\n",
            "        for ct in celltypes:\n",
            "            ct_mask = mask & (adata.obs[celltype_col] == ct)\n",
            "            if ct_mask.sum() > 0:\n",
            "                result[ct] = adata.obsm['X_interaction'][ct_mask.values].mean(axis=0)\n",
            "            else:\n",
            "                result[ct] = np.zeros(len(celltypes))\n",
            "        return pd.DataFrame(result, index=celltypes).T\n",
            "    \n",
            "    df_high = get_mean_interaction_by_celltype(high_mask)\n",
            "    df_low = get_mean_interaction_by_celltype(low_mask)\n",
            "    df_diff = df_high - df_low\n",
            "    \n",
            "    # Plot\n",
            "    fig, axes = plt.subplots(1, 3, figsize=(15, 6))\n",
            "    \n",
            "    vmax = max(df_high.values.max(), df_low.values.max())\n",
            "    \n",
            "    sns.heatmap(df_high, ax=axes[0], cmap='YlOrRd', vmin=0, vmax=vmax,\n",
            "                annot=True, fmt='.2f', cbar_kws={'shrink': 0.8})\n",
            "    axes[0].set_title('High Response', fontsize=12)\n",
            "    axes[0].set_xlabel('Neighbor Cell Type')\n",
            "    axes[0].set_ylabel('Center Cell Type')\n",
            "    \n",
            "    sns.heatmap(df_low, ax=axes[1], cmap='YlOrRd', vmin=0, vmax=vmax,\n",
            "                annot=True, fmt='.2f', cbar_kws={'shrink': 0.8})\n",
            "    axes[1].set_title('Low Response', fontsize=12)\n",
            "    axes[1].set_xlabel('Neighbor Cell Type')\n",
            "    \n",
            "    vmax_diff = max(abs(df_diff.values.min()), abs(df_diff.values.max()))\n",
            "    sns.heatmap(df_diff, ax=axes[2], cmap='RdBu_r', center=0,\n",
            "                vmin=-vmax_diff, vmax=vmax_diff,\n",
            "                annot=True, fmt='.2f', cbar_kws={'shrink': 0.8})\n",
            "    axes[2].set_title('Difference (High - Low)', fontsize=12)\n",
            "    axes[2].set_xlabel('Neighbor Cell Type')\n",
            "    \n",
            "    plt.suptitle('Interaction Embedding Comparison by Response', fontsize=14)\n",
            "    plt.tight_layout()\n",
            "    \n",
            "    if save_path:\n",
            "        plt.savefig(save_path, dpi=150, bbox_inches='tight')\n",
            "    \n",
            "    return fig\n",
            "\n",
            "# Plot interaction difference\n",
            "fig = plot_interaction_difference(\n",
            "    adata_sample,\n",
            "    response_col=response_col,\n",
            "    save_path=OUTPUT_DIR / 'interaction_difference.png'\n",
            ")\n",
            "if fig:\n",
            "    plt.show()"
        ],
        "execution_count": None,
        "outputs": []
    })
    
    # Section 10: Auto-annotation
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": ["## 10. Auto-Annotation of Niches"]
    })
    
    cells.append({
        "cell_type": "code",
        "metadata": {},
        "source": [
            "def auto_annotate_niches(adata, leiden_col='leiden', celltype_col='cell_type',\n",
            "                         response_col='response_label', threshold=0.3):\n",
            "    \"\"\"Automatically annotate niches based on dominant cell types.\"\"\"\n",
            "    print('Auto-annotating niches...')\n",
            "    \n",
            "    clusters = sorted(adata.obs[leiden_col].unique(),\n",
            "                      key=lambda x: int(x) if str(x).isdigit() else x)\n",
            "    \n",
            "    annotations = {}\n",
            "    \n",
            "    for cluster in clusters:\n",
            "        mask = adata.obs[leiden_col] == cluster\n",
            "        subset = adata.obs[mask]\n",
            "        \n",
            "        # Dominant cell types\n",
            "        celltype_props = subset[celltype_col].value_counts(normalize=True)\n",
            "        dominant = celltype_props[celltype_props >= threshold].index.tolist()\n",
            "        \n",
            "        if not dominant:\n",
            "            dominant = [celltype_props.index[0]]\n",
            "        \n",
            "        # Response level\n",
            "        if response_col in subset.columns:\n",
            "            ratio = subset[response_col].mean()\n",
            "            if ratio > 0.6:\n",
            "                response_tag = 'HighResp'\n",
            "            elif ratio < 0.4:\n",
            "                response_tag = 'LowResp'\n",
            "            else:\n",
            "                response_tag = 'MidResp'\n",
            "        else:\n",
            "            response_tag = ''\n",
            "        \n",
            "        type_str = '_'.join([d[:8] for d in dominant[:2]])  # Truncate names\n",
            "        annotation = f'N{cluster}_{type_str}'\n",
            "        if response_tag:\n",
            "            annotation += f'_{response_tag}'\n",
            "        \n",
            "        annotations[cluster] = annotation\n",
            "    \n",
            "    adata.obs['niche_annotation'] = adata.obs[leiden_col].map(annotations)\n",
            "    \n",
            "    return adata, annotations\n",
            "\n",
            "# Auto-annotate\n",
            "adata_sample, annotations = auto_annotate_niches(\n",
            "    adata_sample,\n",
            "    response_col=response_col\n",
            ")\n",
            "\n",
            "print('\\nNiche Annotations:')\n",
            "for cluster, annotation in sorted(annotations.items(),\n",
            "                                   key=lambda x: int(x[0]) if str(x[0]).isdigit() else x[0]):\n",
            "    n_cells = (adata_sample.obs['leiden'] == cluster).sum()\n",
            "    print(f'  {cluster}: {annotation} (n={n_cells:,})')"
        ],
        "execution_count": None,
        "outputs": []
    })
    
    # Section 11: Summary
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": ["## 11. Analysis Summary"]
    })
    
    cells.append({
        "cell_type": "code",
        "metadata": {},
        "source": [
            "# Print summary\n",
            "print('=' * 60)\n",
            "print('ANALYSIS SUMMARY')\n",
            "print('=' * 60)\n",
            "print(f'Dataset: {DATASET}')\n",
            "print(f'Model: {MODEL_NAME}')\n",
            "print(f'Seed: {MODEL_SEED}')\n",
            "print(f'\\nData:')\n",
            "print(f'  Total cells analyzed: {adata_sample.n_obs:,}')\n",
            "print(f'  Features: {adata_sample.n_vars}')\n",
            "print(f'  Cell types: {adata_sample.obs[\"cell_type\"].nunique()}')\n",
            "print(f'  Leiden clusters: {adata_sample.obs[\"leiden\"].nunique()}')\n",
            "print(f'\\nResponse Metric:')\n",
            "print(f'  Column used: {response_col}')\n",
            "print(f'  Using ratio fallback: {use_ratio}')\n",
            "if use_ratio:\n",
            "    print(f'  Reason: {reason}')\n",
            "print(f'\\nOutput files saved to: {OUTPUT_DIR}')\n",
            "for f in sorted(OUTPUT_DIR.glob('*.png')):\n",
            "    print(f'  - {f.name}')"
        ],
        "execution_count": None,
        "outputs": []
    })
    
    # Build notebook structure
    notebook = {
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3 (cartaPA)",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "name": "python",
                "version": "3.10.0"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 5,
        "cells": cells
    }
    
    return notebook


def main():
    args = parse_args()
    
    # Create notebook
    notebook = create_notebook(
        dataset=args.dataset,
        model_name=args.model_name,
        model_seed=args.model_seed,
        h5ad_path=args.h5ad_path,
        sample_size=args.sample_size
    )
    
    # Save notebook
    output_path = Path(args.output_notebook)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(notebook, f, indent=1)
    
    print(f'Notebook saved to: {output_path}')
    print(f'Total cells: {len(notebook["cells"])}')
    print(f'\nTo execute the notebook:')
    print(f'  jupyter nbconvert --to notebook --execute {output_path}')
    print(f'Or:')
    print(f'  papermill {output_path} {output_path.stem}_executed.ipynb')


if __name__ == '__main__':
    main()
