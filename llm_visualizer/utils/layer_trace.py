"""
Level 2: Full forward pass that captures ALL intermediate hidden states
and attention weights at every layer.
"""
import torch
import numpy as np
from typing import Dict, List
from utils.model_loader import load_model, get_device


def forward_full_trace(token_ids: list, model_key: str) -> Dict:
    """
    Run a full forward pass and capture:
    - hidden_states: list of (n_tokens, d_model) arrays, one per layer + input
    - attentions: list of (n_heads, n_tokens, n_tokens) arrays, one per layer
    - logits: (n_tokens, vocab_size) output logits

    hidden_states[0] = input embeddings (before any layer)
    hidden_states[i+1] = output of layer i
    attentions[i] = attention weights of layer i
    """
    model = load_model(model_key)
    device = get_device(model_key)
    input_ids = torch.tensor([token_ids], device=device)

    with torch.no_grad():
        outputs = model(input_ids, output_hidden_states=True, output_attentions=True)

    hidden_states = [h[0].detach().cpu().numpy() for h in outputs.hidden_states]
    attentions = [a[0].detach().cpu().numpy() for a in outputs.attentions]
    logits = outputs.logits[0].detach().cpu().numpy()

    return {
        "hidden_states": hidden_states,   # len = n_layers + 1
        "attentions": attentions,          # len = n_layers
        "logits": logits,
    }


def get_layer_output(trace: Dict, layer_idx: int) -> np.ndarray:
    """Get hidden state output of a specific layer. Shape: (n_tokens, d_model)."""
    return trace["hidden_states"][layer_idx]


def get_layer_delta(trace: Dict, layer_idx: int) -> np.ndarray:
    """
    Get the CHANGE made by a specific layer:
    delta = hidden_states[layer_idx+1] - hidden_states[layer_idx]
    Shows what information that layer added/moved.
    """
    return trace["hidden_states"][layer_idx + 1] - trace["hidden_states"][layer_idx]


def get_all_layer_deltas(trace: Dict) -> List[np.ndarray]:
    """Get deltas for all layers."""
    n_layers = len(trace["attentions"])
    return [get_layer_delta(trace, i) for i in range(n_layers)]


def get_token_trajectory(trace: Dict, token_idx: int) -> np.ndarray:
    """
    Get the trajectory of a single token through all layers.
    Returns shape: (n_layers+1, d_model) — the representation of that token
    at each stage.
    """
    return np.array([hs[token_idx] for hs in trace["hidden_states"]])


def compute_layer_norms(trace: Dict) -> List[Dict]:
    """
    Compute statistics for each layer's output and delta.
    Useful for understanding how much each layer changes things.
    """
    stats = []
    n_layers = len(trace["attentions"])
    for i in range(n_layers):
        output = trace["hidden_states"][i + 1]
        delta = get_layer_delta(trace, i)
        stats.append({
            "layer": i,
            "output_norm": float(np.linalg.norm(output)),
            "delta_norm": float(np.linalg.norm(delta)),
            "delta_mean": float(np.mean(np.abs(delta))),
            "delta_max": float(np.max(np.abs(delta))),
        })
    return stats

