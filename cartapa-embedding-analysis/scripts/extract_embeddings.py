#!/usr/bin/env python3
"""
Generic CartaPA Embedding Extraction Script

Extract embeddings from trained CartaPA models for any supported dataset.

Usage:
    ~/miniconda3/envs/purturb/bin/python extract_embeddings.py \
        --model_name MODEL_NAME \
        --model_seed SEED_HASH \
        --dataset DATASET_ID \
        --output_dir OUTPUT_DIR \
        --output_name embeddings.pkl

Supported datasets:
    - codex_hcc_pre
    - codex_tnbc_pre
    - imc_tnbc_pre
    - safe_hnscc_pre

Author: CartaPA Project
"""

import sys
import os
import unittest.mock as mock

# Mock GCL imports to avoid dependency issues
sys.modules['GCL'] = mock.MagicMock()
sys.modules['GCL.losses'] = mock.MagicMock()
sys.modules['GCL.augmentors'] = mock.MagicMock()
sys.modules['GCL.models'] = mock.MagicMock()
sys.modules['GCL.eval'] = mock.MagicMock()

# Add model repo to path
MODEL_REPO_PATH = '/home/kang/project/CartaPA_model/read_only_repo'
sys.path.insert(0, MODEL_REPO_PATH)

import pickle
import argparse
import numpy as np
import torch
import anndata as ad
from scipy.spatial import Delaunay

from copy import deepcopy
from tqdm import tqdm
from pathlib import Path
import torch_geometric
from torch_geometric.data import Data
from torch_geometric.nn.conv import GINConv, GATv2Conv
from torch_geometric.nn.pool import global_mean_pool, global_add_pool
from sklearn.preprocessing import LabelEncoder
from types import FunctionType

# Paths
DATA_PATH = Path(MODEL_REPO_PATH) / 'data'
RAW_PATH = DATA_PATH / 'raw'
PROCESSED_PATH = DATA_PATH / 'processed'


def parse_args():
    parser = argparse.ArgumentParser(description='Extract CartaPA embeddings')
    parser.add_argument('--model_name', type=str, required=True, help='Model name')
    parser.add_argument('--model_seed', type=str, required=True, help='Model seed subfolder')
    parser.add_argument('--dataset', type=str, required=True, 
                        choices=['codex_hcc_pre', 'codex_tnbc_pre', 'imc_tnbc_pre', 'safe_hnscc_pre'],
                        help='Dataset identifier')
    parser.add_argument('--device', type=str, default=None, help='Device to use')
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory')
    parser.add_argument('--output_name', type=str, default='embeddings.pkl', help='Output filename')
    return parser.parse_args()


def get_free_gpu():
    try:
        import subprocess
        result = subprocess.run(['nvidia-smi', '--query-gpu=memory.free', '--format=csv,nounits,noheader'], 
                               capture_output=True, text=True)
        memory_free = [int(x) for x in result.stdout.strip().split('\n')]
        return memory_free.index(max(memory_free))
    except:
        return 0


def seed_everything(seed=42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch_geometric.seed_everything(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


class FeatureExtractor(torch.nn.Module):
    """Wrapper model that extracts dense embeddings before BinaryActivation."""
    def __init__(self, original_model):
        super().__init__()
        self.model = deepcopy(original_model)
        self.encoder = self.model.encoder
        self.encoder.gnn.forward = self.custom_gnn_forward
        self.config = self.model.config
        self.original_out_act = self.encoder.gnn.out_act
        
    def custom_gnn_forward(self, x, edge_index, batch):
        outputs = []
        if self.encoder.gnn.use_virtual_node:
            vn_emb = self.encoder.gnn.virtualnode_embedding(
                torch.zeros(batch[-1].item() + 1).to(edge_index.dtype).to(edge_index.device))

        for model_idx, model in enumerate(self.encoder.gnn.combined_models):
            layer_xs = []
            curr_x = x
            for i, layer in enumerate(model):
                if isinstance(layer, (GINConv, GATv2Conv)):
                    curr_x = layer(curr_x, edge_index)
                else:
                    curr_x = layer(curr_x)

                if self.encoder.gnn.use_virtual_node:
                    vn_emb_tmp = global_add_pool(self.original_out_act(curr_x), batch) + vn_emb
                    vn_emb_abs = vn_emb_tmp.abs()
                    vn_emb_normalized = vn_emb_tmp / (vn_emb_abs + 1e-8)
                    vn_emb_tmp = torch.log1p(vn_emb_abs) * vn_emb_normalized
                    vn_emb = self.encoder.gnn.virtualnode_mlps[i](vn_emb_tmp)
                    curr_x = self.original_out_act(curr_x + vn_emb[batch])
                else:
                    if i < len(model) - 1:
                        curr_x = self.original_out_act(curr_x)

                if not isinstance(layer, GATv2Conv):
                    layer_xs.append(curr_x)
            
            if self.encoder.gnn.layers_combination == 'mean':
                outputs.append(torch.stack(layer_xs, dim=0).mean(dim=0))
            elif self.encoder.gnn.layers_combination == 'cat':
                outputs.append(torch.cat(layer_xs, dim=-1))
            elif self.encoder.gnn.layers_combination == 'last':
                outputs.append(layer_xs[-1])
        
        z = torch.stack(outputs, dim=0).mean(dim=0)

        if isinstance(self.encoder.gnn.pool, FunctionType):
            g = self.encoder.gnn.pool(z, batch)
        else:
            (x_pooled, edge_index_pooled, edge_attr, batch_pooled, perm, score) = \
                self.encoder.gnn.pool(z, edge_index=edge_index, batch=batch)
            g = global_mean_pool(x_pooled, batch_pooled)
            
        return z, g
    
    def forward(self, batch_data):
        return self.encoder(batch_data)


def load_model(model_path, device):
    print(f"Loading model from: {model_path}")
    
    if torch.cuda.is_available():
        map_location = device
    else:
        map_location = torch.device('cpu')
    
    model = torch.load(model_path, weights_only=False, map_location=map_location)
    
    if hasattr(model, 'config'):
        model.config.device = device
    model.to(device)
    model.eval()
    
    feature_extractor = FeatureExtractor(model)
    feature_extractor.to(device)
    feature_extractor.eval()
    
    print(f"Model loaded: {type(model).__name__}")
    return model, feature_extractor


def load_label_encoders(pt_name='datasets_all_merged_soft_delaunay.pt'):
    encoder_path = PROCESSED_PATH / pt_name.replace('.pt', '_label_encoders.pkl')
    print(f"Loading label encoders from: {encoder_path}")
    
    with open(encoder_path, 'rb') as f:
        label_encoders = pickle.load(f)
    
    return label_encoders


def load_dataset_adata(dataset_id):
    h5ad_path = RAW_PATH / 'datasets_all_merged_soft.h5ad'
    print(f"Loading data from: {h5ad_path}")
    
    adata = ad.read_h5ad(h5ad_path)
    
    # Filter to target dataset
    mask = adata.obs['dataset'] == dataset_id
    dataset_adata = adata[mask].copy()
    dataset_adata.obs_names_make_unique()
    
    print(f"Loaded {dataset_id} data: {dataset_adata.shape}")
    print(f"Unique slices: {dataset_adata.obs['slice_id'].nunique()}")
    
    return dataset_adata


def build_delaunay_edges(coords):
    """Build Delaunay triangulation edges from coordinates."""
    tri = Delaunay(coords)
    
    edges = set()
    for simplex in tri.simplices:
        for i in range(3):
            for j in range(i + 1, 3):
                edges.add((simplex[i], simplex[j]))
                edges.add((simplex[j], simplex[i]))
    
    edges = np.array(list(edges), dtype=np.int64).T
    return edges


def build_graph_data(adata_slice, label_encoders):
    """Build graph data from slice AnnData."""
    if 'spatial' in adata_slice.obsm:
        coords = adata_slice.obsm['spatial']
    else:
        raise ValueError("No spatial coordinates found in adata.obsm['spatial']")
    
    edge_index = build_delaunay_edges(coords)
    edge_index = torch.tensor(edge_index, dtype=torch.long)
    
    cell_type_encoder = label_encoders.get('cell_type', None)
    if cell_type_encoder is None:
        cell_type_encoder = LabelEncoder()
        cell_type_encoder.fit(adata_slice.obs['cell_type'].unique())
    
    cell_types = cell_type_encoder.transform(adata_slice.obs['cell_type'].values)
    
    panel = adata_slice.X.copy()
    if hasattr(panel, 'toarray'):
        panel = panel.toarray()
    panel = panel.astype(np.float32)
    
    data = Data(
        edge_index=edge_index,
        num_nodes=adata_slice.n_obs,
        cell_type=torch.tensor(cell_types, dtype=torch.long),
        panel=torch.tensor(panel, dtype=torch.float32),
    )
    
    return data


def prepare_batch_data(data, config, device):
    batch = torch.zeros(data.num_nodes, dtype=torch.long, device=device)
    
    edge_index = data.edge_index.to(device)
    cell_type = data.cell_type.to(device)
    panel = data.panel.to(device)
    
    batch_data = Data(
        edge_index=edge_index,
        batch=batch,
        cell_type=cell_type,
        panel=panel,
        num_nodes=data.num_nodes,
    )
    
    batch_data.x_cell_type = cell_type
    batch_data.x_panel = panel
    batch_data.ptr = torch.tensor([0, data.num_nodes], dtype=torch.long, device=device)
    
    return batch_data


def extract_embeddings(model, feature_extractor, dataset_adata, label_encoders, device):
    seed_everything(42)
    
    unique_slices = dataset_adata.obs['slice_id'].unique()
    print(f"Processing {len(unique_slices)} slices...")
    
    slice_embeddings = {}
    
    for i, slice_id in enumerate(tqdm(unique_slices, desc="Extracting embeddings")):
        slice_mask = dataset_adata.obs['slice_id'] == slice_id
        slice_adata = dataset_adata[slice_mask].copy()
        
        try:
            data = build_graph_data(slice_adata, label_encoders)
        except Exception as e:
            print(f"Warning: Failed to build graph for slice {slice_id}: {e}")
            continue
        
        batch_data = prepare_batch_data(data, model.config, device)
        
        with torch.no_grad():
            try:
                sparse_node_embeddings, graph_embeddings = model(batch_data)
                
                dense_node_embeddings, _ = feature_extractor(batch_data)
                dense_node_embeddings = torch.tanh(dense_node_embeddings * 0.05)
                
                if hasattr(model, 'slice_labels_predictors') and 'y' in model.slice_labels_predictors:
                    node_response_probs = torch.sigmoid(
                        model.slice_labels_predictors['y'](dense_node_embeddings))
                    graph_response_probs = torch.sigmoid(
                        model.slice_labels_predictors['y'](graph_embeddings))
                else:
                    node_response_probs = torch.zeros(data.num_nodes, 1, device=device)
                    graph_response_probs = torch.zeros(1, 1, device=device)
                
                dense_node_embeddings = np.round(dense_node_embeddings.cpu().numpy(), decimals=6)
                sparse_node_embeddings = np.round(sparse_node_embeddings.cpu().numpy(), decimals=6)
                graph_embeddings = np.round(graph_embeddings.cpu().numpy(), decimals=6)
                node_response_probs_np = np.round(node_response_probs.cpu().numpy(), decimals=6)
                graph_response_probs_np = np.round(graph_response_probs.cpu().numpy(), decimals=6)
                
                celltypes = data.cell_type.cpu().numpy()
                panels = data.panel.cpu().numpy()
                
                patient_id = slice_adata.obs['patient_id'].iloc[0]
                slice_label = slice_adata.obs.get('slice_label', slice_adata.obs.get('label', [0])).iloc[0]
                slice_label = int(slice_label) if isinstance(slice_label, (int, float, np.integer, np.floating)) else 0
                
                slice_embedding = {
                    'patient_id': patient_id,
                    'label': np.array(slice_label),
                    'is_test': False,
                    'is_post_treatment': False,
                    'raw_img_id': slice_id,
                    'embedding': graph_embeddings[0],
                    'response_prob': graph_response_probs_np[0],
                    'node_panels': panels,
                    'node_celltypes': celltypes,
                    'node_response_probs': node_response_probs_np[:, 0],
                    'dense_node_embeddings': dense_node_embeddings,
                    'sparse_node_embeddings': sparse_node_embeddings,
                }
                
                slice_embeddings[i] = slice_embedding
                
            except Exception as e:
                print(f"Warning: Failed to extract embeddings for slice {slice_id}: {e}")
                import traceback
                traceback.print_exc()
                continue
    
    print(f"Successfully extracted embeddings for {len(slice_embeddings)} slices")
    return slice_embeddings


def main():
    args = parse_args()
    
    if args.device is None:
        if torch.cuda.is_available():
            gpu_id = get_free_gpu()
            device = f'cuda:{gpu_id}'
        else:
            device = 'cpu'
    else:
        device = args.device
    
    print(f"Using device: {device}")
    
    model_path = os.path.join(
        MODEL_REPO_PATH, 
        'saved_models', 
        args.model_name,
        args.model_seed,
        'model_best',
        'model.pth'
    )
    
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        sys.exit(1)
    
    model, feature_extractor = load_model(model_path, device)
    label_encoders = load_label_encoders()
    dataset_adata = load_dataset_adata(args.dataset)
    
    slice_embeddings = extract_embeddings(
        model, feature_extractor, dataset_adata, label_encoders, device
    )
    
    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, args.output_name)
    
    with open(output_path, 'wb') as f:
        pickle.dump(slice_embeddings, f)
    
    print(f"\nSaved embeddings to: {output_path}")
    print(f"Total slices: {len(slice_embeddings)}")
    
    if slice_embeddings:
        first_key = list(slice_embeddings.keys())[0]
        first_val = slice_embeddings[first_key]
        print("\nEmbedding structure:")
        for k, v in first_val.items():
            if hasattr(v, 'shape'):
                print(f"  {k}: shape={v.shape}, dtype={v.dtype}")
            else:
                print(f"  {k}: type={type(v).__name__}, value={v}")


if __name__ == '__main__':
    main()
