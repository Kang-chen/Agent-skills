#!/usr/bin/env python3
"""
Build AnnData from CartaPA Embeddings

Convert embedding pickle files to AnnData format with full metadata.

Usage:
    ~/miniconda3/envs/cartaPA/bin/python build_anndata.py \
        --dataset DATASET_ID \
        --embedding_path embeddings.pkl \
        --output_path output.h5ad \
        --add_interaction_embeddings

Author: CartaPA Project
"""

import argparse
import pickle
from pathlib import Path
from typing import Dict, List, Optional

import anndata as ad
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm

MODEL_REPO_PATH = Path('/home/kang/project/CartaPA_model/read_only_repo')
RAW_PATH = MODEL_REPO_PATH / 'data' / 'raw'
PROCESSED_PATH = MODEL_REPO_PATH / 'data' / 'processed'


def parse_args():
    parser = argparse.ArgumentParser(description='Build AnnData from CartaPA embeddings')
    parser.add_argument('--dataset', type=str, required=True,
                        choices=['codex_hcc_pre', 'codex_tnbc_pre', 'imc_tnbc_pre', 'safe_hnscc_pre'],
                        help='Dataset identifier')
    parser.add_argument('--embedding_path', type=str, required=True,
                        help='Path to embedding pickle file')
    parser.add_argument('--output_path', type=str, required=True,
                        help='Output h5ad path')
    parser.add_argument('--add_interaction_embeddings', action='store_true',
                        help='Add 1-hop interaction embeddings')
    parser.add_argument('--n_neighbors', type=int, default=15,
                        help='Number of neighbors for interaction (default: 15)')
    return parser.parse_args()


def load_embeddings(embedding_path: str) -> Dict:
    """Load embeddings from pickle file."""
    print(f"Loading embeddings from: {embedding_path}")
    with open(embedding_path, 'rb') as f:
        embeddings = pickle.load(f)
    print(f"Loaded {len(embeddings)} slices")
    return embeddings


def load_label_encoders():
    """Load label encoders for cell type decoding."""
    encoder_path = PROCESSED_PATH / 'datasets_all_merged_soft_delaunay_label_encoders.pkl'
    print(f"Loading label encoders from: {encoder_path}")
    with open(encoder_path, 'rb') as f:
        label_encoders = pickle.load(f)
    return label_encoders


def load_source_adata(dataset_id: str) -> ad.AnnData:
    """Load source AnnData for metadata."""
    h5ad_path = RAW_PATH / 'datasets_all_merged_soft.h5ad'
    print(f"Loading source data from: {h5ad_path}")
    
    adata = ad.read_h5ad(h5ad_path)
    mask = adata.obs['dataset'] == dataset_id
    dataset_adata = adata[mask].copy()
    dataset_adata.obs_names_make_unique()
    
    print(f"Loaded {dataset_id}: {dataset_adata.shape}")
    return dataset_adata


def build_anndata(embeddings: Dict, source_adata: ad.AnnData, 
                  label_encoders: Dict) -> ad.AnnData:
    """Build AnnData from embeddings and source metadata."""
    print("Building AnnData...")
    
    # Get cell type decoder
    cell_type_encoder = label_encoders.get('cell_type')
    if cell_type_encoder is not None:
        celltype_classes = cell_type_encoder.classes_
    else:
        celltype_classes = None
    
    # Collect data from all slices
    all_panels = []
    all_dense_emb = []
    all_sparse_emb = []
    all_response_probs = []
    all_celltypes = []
    all_slice_ids = []
    all_patient_ids = []
    all_slice_labels = []
    all_slice_response_probs = []
    
    slice_id_mapping = {}
    
    for slice_idx, slice_data in tqdm(embeddings.items(), desc="Processing slices"):
        n_cells = slice_data['node_panels'].shape[0]
        
        all_panels.append(slice_data['node_panels'])
        all_dense_emb.append(slice_data['dense_node_embeddings'])
        all_sparse_emb.append(slice_data['sparse_node_embeddings'])
        all_response_probs.append(slice_data['node_response_probs'])
        all_celltypes.append(slice_data['node_celltypes'])
        
        slice_id = slice_data['raw_img_id']
        slice_id_mapping[slice_idx] = slice_id
        all_slice_ids.extend([slice_id] * n_cells)
        all_patient_ids.extend([slice_data['patient_id']] * n_cells)
        
        label = int(slice_data['label']) if hasattr(slice_data['label'], 'item') else int(slice_data['label'])
        all_slice_labels.extend([label] * n_cells)
        
        slice_resp = slice_data['response_prob']
        if hasattr(slice_resp, '__len__'):
            slice_resp = float(slice_resp[0])
        all_slice_response_probs.extend([slice_resp] * n_cells)
    
    # Concatenate arrays
    X = np.vstack(all_panels).astype(np.float32)
    dense_emb = np.vstack(all_dense_emb).astype(np.float32)
    sparse_emb = np.vstack(all_sparse_emb).astype(np.float32)
    response_probs = np.concatenate(all_response_probs).astype(np.float32)
    celltypes_encoded = np.concatenate(all_celltypes)
    
    # Decode cell types
    if celltype_classes is not None:
        celltypes = [celltype_classes[i] if i < len(celltype_classes) else f"Unknown_{i}" 
                     for i in celltypes_encoded]
    else:
        celltypes = [f"CellType_{i}" for i in celltypes_encoded]
    
    # Get protein names from source adata
    if source_adata is not None and source_adata.var_names is not None:
        var_names = source_adata.var_names.tolist()
        if len(var_names) != X.shape[1]:
            var_names = [f"Protein_{i}" for i in range(X.shape[1])]
    else:
        var_names = [f"Protein_{i}" for i in range(X.shape[1])]
    
    # Build obs DataFrame
    obs = pd.DataFrame({
        'dataset': source_adata.obs['dataset'].iloc[0] if source_adata is not None else 'unknown',
        'patient_id': all_patient_ids,
        'slice_id': all_slice_ids,
        'slice_label': all_slice_labels,
        'cell_type': celltypes,
        'cell_type_encoded': celltypes_encoded,
        'node_response_prob': response_probs,
        'slice_response_prob': all_slice_response_probs,
    })
    
    # Create AnnData
    adata = ad.AnnData(
        X=X,
        obs=obs,
        var=pd.DataFrame(index=var_names)
    )
    
    # Add embeddings to obsm
    adata.obsm['X_cartapa'] = dense_emb
    adata.obsm['X_cartapa_sparse'] = sparse_emb
    
    # Try to get spatial coordinates from source
    if source_adata is not None:
        spatial_coords = []
        for slice_idx, slice_data in embeddings.items():
            slice_id = slice_data['raw_img_id']
            source_mask = source_adata.obs['slice_id'] == slice_id
            if source_mask.sum() > 0 and 'spatial' in source_adata.obsm:
                coords = source_adata.obsm['spatial'][source_mask.values]
                n_needed = slice_data['node_panels'].shape[0]
                if len(coords) >= n_needed:
                    spatial_coords.append(coords[:n_needed])
                else:
                    # Pad with zeros if needed
                    padded = np.zeros((n_needed, 2))
                    padded[:len(coords)] = coords
                    spatial_coords.append(padded)
            else:
                # Generate placeholder coordinates
                n_cells = slice_data['node_panels'].shape[0]
                spatial_coords.append(np.random.rand(n_cells, 2) * 1000)
        
        adata.obsm['spatial'] = np.vstack(spatial_coords)
    
    # Add metadata
    adata.uns['cartapa_info'] = {
        'n_slices': len(embeddings),
        'embedding_dim': dense_emb.shape[1],
        'n_proteins': X.shape[1],
        'slice_id_mapping': slice_id_mapping,
    }
    
    print(f"Built AnnData: {adata.shape}")
    print(f"  obsm keys: {list(adata.obsm.keys())}")
    print(f"  obs columns: {list(adata.obs.columns)}")
    
    return adata


def add_interaction_embeddings(adata: ad.AnnData, n_neighbors: int = 15) -> ad.AnnData:
    """Add 1-hop neighbor cell type count embeddings."""
    print(f"Computing interaction embeddings with {n_neighbors} neighbors...")
    
    celltypes = sorted(adata.obs['cell_type'].unique())
    celltype_to_idx = {ct: i for i, ct in enumerate(celltypes)}
    
    interaction_matrix = np.zeros((adata.n_obs, len(celltypes)), dtype=np.float32)
    
    for slice_id in tqdm(adata.obs['slice_id'].unique(), desc="Processing slices"):
        mask = adata.obs['slice_id'] == slice_id
        slice_indices = np.where(mask)[0]
        
        if len(slice_indices) < n_neighbors + 1:
            continue
        
        coords = adata.obsm['spatial'][slice_indices]
        cell_types = adata.obs['cell_type'].iloc[slice_indices].values
        
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
        
        # Normalize
        row_sums = interaction_matrix[slice_indices].sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        interaction_matrix[slice_indices] /= row_sums
    
    adata.obsm['X_interaction'] = interaction_matrix
    adata.uns['interaction_celltypes'] = celltypes
    
    print(f"Added obsm['X_interaction'] with shape {interaction_matrix.shape}")
    return adata


def main():
    args = parse_args()
    
    # Load data
    embeddings = load_embeddings(args.embedding_path)
    label_encoders = load_label_encoders()
    source_adata = load_source_adata(args.dataset)
    
    # Build AnnData
    adata = build_anndata(embeddings, source_adata, label_encoders)
    
    # Add interaction embeddings if requested
    if args.add_interaction_embeddings:
        adata = add_interaction_embeddings(adata, n_neighbors=args.n_neighbors)
    
    # Save
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    adata.write_h5ad(output_path)
    print(f"\nSaved AnnData to: {output_path}")
    print(f"  Shape: {adata.shape}")
    print(f"  Size: {output_path.stat().st_size / 1024 / 1024:.1f} MB")


if __name__ == '__main__':
    main()
