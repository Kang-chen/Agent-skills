# Response Probability Alternatives

## Problem Description

Some CartaPA models output `node_response_prob` values that are all zeros or constant values. This occurs when:

1. The model architecture doesn't include node-level response prediction
2. The model was trained without node-level supervision
3. The dataset lacks sufficient response signal for node-level prediction

### How to Detect

```python
# Check if node_response_prob is valid
response_probs = adata.obs['node_response_prob']

# Check for all zeros
if response_probs.sum() == 0:
    print("WARNING: node_response_prob is all zeros")
    use_response_ratio = True

# Check for constant values
if response_probs.std() < 1e-6:
    print("WARNING: node_response_prob is constant")
    use_response_ratio = True

# Check for valid probability range
if response_probs.min() < 0 or response_probs.max() > 1:
    print("WARNING: node_response_prob out of [0,1] range")
```

## Alternative: Response Ratio from Slice Labels

### Concept

Use `slice_label` (slice-level response label: 0=Non-responder, 1=Responder) to compute a response ratio for each group:

```
response_ratio(group) = mean(slice_label) for cells in group
                      = # cells from responder slices / # total cells in group
```

### Justification

1. **Clinical Validity**: All cells from the same tissue slice share the same clinical response outcome
2. **Biological Meaning**: High response_ratio indicates the cluster is enriched in cells from responding patients
3. **Statistical Robustness**: Uses actual clinical labels rather than model predictions

### Implementation

```python
def calculate_response_ratio(df, group_col, response_col='slice_label'):
    """
    Calculate response ratio (proportion of cells from responder slices) per group.
    
    Parameters
    ----------
    df : DataFrame
        Must contain group_col and response_col
    group_col : str
        Column to group by (e.g., 'leiden', 'cell_type')
    response_col : str
        Binary response column (0/1), default 'slice_label'
    
    Returns
    -------
    dict : {group_value: response_ratio}
    """
    # Ensure response column is numeric
    df = df.copy()
    df[response_col] = df[response_col].astype(int)
    
    response_ratios = {}
    for grp in df[group_col].unique():
        subset = df[df[group_col] == grp]
        response_ratios[grp] = subset[response_col].mean()
    
    return response_ratios
```

### Interpretation

| Response Ratio | Interpretation |
|----------------|----------------|
| > 0.7 | Strongly enriched in responders |
| 0.5 - 0.7 | Slightly enriched in responders |
| 0.3 - 0.5 | Slightly enriched in non-responders |
| < 0.3 | Strongly enriched in non-responders |

### Documentation in Analysis

When using response_ratio instead of node_response_prob, add this note to your analysis:

```markdown
**Note on Response Metric**

This analysis uses `response_ratio` (proportion of cells from responder slices) 
instead of `node_response_prob` because:

1. **Evidence**: `node_response_prob` values are all zeros/constant in this dataset
2. **Alternative**: `response_ratio` computed from `slice_label` provides biologically 
   meaningful response enrichment scores
3. **Interpretation**: Higher ratio indicates cluster enrichment in responding patients

This is a valid alternative as cells from the same tissue slice share clinical outcomes.
```

## Comparison Table

| Metric | Source | Range | Best For |
|--------|--------|-------|----------|
| `node_response_prob` | Model prediction | [0, 1] | When model outputs valid probabilities |
| `response_ratio` | `slice_label` aggregation | [0, 1] | When node_response_prob is invalid |
| `slice_response_prob` | Model slice-level prediction | [0, 1] | Slice-level analysis |

## Code Example: Auto-Detection

```python
def get_response_metric(adata):
    """
    Automatically select appropriate response metric.
    
    Returns
    -------
    tuple : (metric_name, metric_values, use_ratio)
    """
    response_probs = adata.obs.get('node_response_prob')
    
    # Check validity
    if response_probs is None:
        use_ratio = True
    elif response_probs.std() < 1e-6:  # Constant or all zeros
        use_ratio = True
    else:
        use_ratio = False
    
    if use_ratio:
        # Use response_ratio from slice_label
        if 'slice_label' not in adata.obs.columns:
            raise ValueError("Neither node_response_prob nor slice_label available")
        
        metric_name = 'response_ratio'
        # Convert slice_label to int and use as metric
        adata.obs['response_ratio'] = adata.obs['slice_label'].astype(int)
        metric_values = adata.obs['response_ratio']
        
        print("Using response_ratio from slice_label (node_response_prob invalid)")
    else:
        metric_name = 'node_response_prob'
        metric_values = response_probs
        print("Using node_response_prob from model predictions")
    
    return metric_name, metric_values, use_ratio
```
