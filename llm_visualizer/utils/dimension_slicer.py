"""
Level 3: Dimension slicing — select arbitrary subsets of embedding dimensions.
Supports syntax like:
  "2,45,6"            -> specific dimensions
  "0:100"             -> range
  "0:100:2"           -> every 2nd dimension from 0 to 100
  "all"               -> all dimensions
  "0:768:3"           -> every 3rd dimension
"""
import numpy as np
from typing import List, Union, Dict


def parse_dim_spec(spec: str, max_dim: int) -> List[int]:
    """
    Parse a dimension specification string into a list of dimension indices.

    Supported formats:
    - "2,45,6"              -> [2, 45, 6]
    - "0:100"               -> [0, 1, 2, ..., 99]
    - "0:100:2"             -> [0, 2, 4, ..., 98]
    - "all"                 -> [0, 1, ..., max_dim-1]
    - "0:768:3"             -> every 3rd from 0 to 768
    - "10:50,100:150"       -> combine ranges
    - "even"                -> [0, 2, 4, ...]
    - "odd"                 -> [1, 3, 5, ...]
    - "first_half"          -> [0, ..., max_dim//2 - 1]
    - "second_half"         -> [max_dim//2, ..., max_dim-1]
    """
    spec = spec.strip().lower()

    if spec == "all":
        return list(range(max_dim))
    elif spec == "even":
        return list(range(0, max_dim, 2))
    elif spec == "odd":
        return list(range(1, max_dim, 2))
    elif spec == "first_half":
        return list(range(max_dim // 2))
    elif spec == "second_half":
        return list(range(max_dim // 2, max_dim))

    dims = []
    parts = spec.split(",")
    for part in parts:
        part = part.strip()
        if ":" in part:
            # Range notation: start:end or start:end:step
            range_parts = part.split(":")
            if len(range_parts) == 2:
                start, end = int(range_parts[0]), int(range_parts[1])
                dims.extend(range(start, min(end, max_dim)))
            elif len(range_parts) == 3:
                start, end, step = int(range_parts[0]), int(range_parts[1]), int(range_parts[2])
                dims.extend(range(start, min(end, max_dim), step))
        else:
            d = int(part)
            if 0 <= d < max_dim:
                dims.append(d)

    # Remove duplicates, keep order
    seen = set()
    unique_dims = []
    for d in dims:
        if d not in seen and 0 <= d < max_dim:
            seen.add(d)
            unique_dims.append(d)
    return unique_dims


def slice_dimensions(data: np.ndarray, dim_spec: str) -> np.ndarray:
    """
    Slice specific dimensions from data.
    data shape: (..., d_model)
    Returns: (..., len(selected_dims))
    """
    max_dim = data.shape[-1]
    dims = parse_dim_spec(dim_spec, max_dim)
    return data[..., dims]


def get_dimension_stats(data: np.ndarray, dim_indices: List[int]) -> List[Dict]:
    """
    Compute per-dimension statistics for selected dimensions.
    data shape: (n_tokens, d_model)
    """
    stats = []
    for d in dim_indices:
        col = data[:, d]
        stats.append({
            "dim": d,
            "mean": float(np.mean(col)),
            "std": float(np.std(col)),
            "min": float(np.min(col)),
            "max": float(np.max(col)),
            "range": float(np.max(col) - np.min(col)),
        })
    return stats

