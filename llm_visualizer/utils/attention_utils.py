"""
Level 2: Attention weight extraction and analysis utilities.
"""
import numpy as np
from typing import Dict, List, Optional


def get_attention_weights(trace: Dict, layer_idx: int, head_idx: Optional[int] = None) -> np.ndarray:
    """
    Get attention weights for a specific layer.
    If head_idx is None, return all heads: shape (n_heads, seq_len, seq_len).
    If head_idx is specified, return that head: shape (seq_len, seq_len).
    """
    attn = trace["attentions"][layer_idx]
    if head_idx is not None:
        return attn[head_idx]
    return attn


def get_attention_to_token(trace: Dict, layer_idx: int, token_idx: int,
                            head_idx: Optional[int] = None) -> np.ndarray:
    """
    How much does each token attend TO a specific token?
    Returns a vector of attention scores from all positions to token_idx.
    """
    attn = get_attention_weights(trace, layer_idx, head_idx)
    if head_idx is not None:
        return attn[:, token_idx]  # (seq_len,)
    return attn[:, :, token_idx]   # (n_heads, seq_len)


def get_attention_from_token(trace: Dict, layer_idx: int, token_idx: int,
                              head_idx: Optional[int] = None) -> np.ndarray:
    """
    Where does a specific token attend?
    Returns the attention distribution FROM token_idx to all other positions.
    """
    attn = get_attention_weights(trace, layer_idx, head_idx)
    if head_idx is not None:
        return attn[token_idx, :]  # (seq_len,)
    return attn[:, token_idx, :]   # (n_heads, seq_len)


def get_attention_rollout(trace: Dict, max_layer: Optional[int] = None) -> np.ndarray:
    """
    Compute attention rollout: multiply attention matrices across layers
    to see the cumulative attention flow from input to any layer.
    Returns shape: (seq_len, seq_len).
    """
    n_layers = len(trace["attentions"])
    if max_layer is None:
        max_layer = n_layers

    # Average across heads for each layer
    rollout = np.eye(trace["attentions"][0].shape[-1])
    for i in range(min(max_layer, n_layers)):
        attn = trace["attentions"][i].mean(axis=0)  # avg over heads
        # Add residual connection (identity)
        attn_with_residual = 0.5 * attn + 0.5 * np.eye(attn.shape[0])
        # Normalize rows
        attn_with_residual /= attn_with_residual.sum(axis=-1, keepdims=True)
        rollout = rollout @ attn_with_residual
    return rollout


def get_head_importance(trace: Dict, layer_idx: int) -> List[Dict]:
    """
    Measure how 'diverse' each head's attention pattern is (entropy).
    Higher entropy = more diffuse attention; lower = more focused.
    """
    attn = trace["attentions"][layer_idx]  # (n_heads, seq, seq)
    results = []
    for h in range(attn.shape[0]):
        # Compute average entropy across query positions
        head_attn = attn[h]  # (seq, seq)
        entropy = -np.sum(head_attn * np.log(head_attn + 1e-10), axis=-1).mean()
        results.append({
            "head": h,
            "avg_entropy": float(entropy),
            "max_attn": float(head_attn.max()),
        })
    return results

