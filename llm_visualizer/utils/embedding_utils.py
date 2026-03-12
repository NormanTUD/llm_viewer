"""
Level 1: Embedding extraction — get the raw word and position embeddings
before any transformer layers process them.
"""
import torch
import numpy as np
from utils.model_loader import load_model, load_tokenizer, get_device


def get_token_embedding_matrix(model_key: str) -> np.ndarray:
    """
    Return the full token embedding matrix: shape (vocab_size, d_model).
    This is the lookup table that maps token IDs to vectors.
    """
    model = load_model(model_key)
    wte = model.transformer.wte.weight.detach().cpu().numpy()
    return wte


def get_token_embeddings(token_ids: list, model_key: str) -> np.ndarray:
    """
    Given a list of token IDs, return their embeddings: shape (n_tokens, d_model).
    """
    model = load_model(model_key)
    device = get_device(model_key)
    ids_tensor = torch.tensor([token_ids], device=device)
    with torch.no_grad():
        embeddings = model.transformer.wte(ids_tensor)
    return embeddings[0].detach().cpu().numpy()


def get_position_embeddings(seq_len: int, model_key: str) -> np.ndarray:
    """
    Return position embeddings for positions 0..seq_len-1.
    Shape: (seq_len, d_model).
    """
    model = load_model(model_key)
    device = get_device(model_key)
    positions = torch.arange(0, seq_len, device=device).unsqueeze(0)
    with torch.no_grad():
        pos_emb = model.transformer.wpe(positions)
    return pos_emb[0].detach().cpu().numpy()


def get_combined_input_embeddings(token_ids: list, model_key: str) -> np.ndarray:
    """
    Return token_embedding + position_embedding (the actual input to layer 0).
    Shape: (n_tokens, d_model).
    """
    tok_emb = get_token_embeddings(token_ids, model_key)
    pos_emb = get_position_embeddings(len(token_ids), model_key)
    return tok_emb + pos_emb


def get_nearest_tokens_to_vector(vector: np.ndarray, model_key: str, top_k: int = 10):
    """
    Find the top_k tokens whose embeddings are closest to a given vector.
    Uses cosine similarity.
    """
    emb_matrix = get_token_embedding_matrix(model_key)
    # Cosine similarity
    vec_norm = vector / (np.linalg.norm(vector) + 1e-9)
    emb_norms = emb_matrix / (np.linalg.norm(emb_matrix, axis=1, keepdims=True) + 1e-9)
    similarities = emb_norms @ vec_norm
    top_indices = np.argsort(similarities)[-top_k:][::-1]

    tokenizer = load_tokenizer(model_key)
    results = []
    for idx in top_indices:
        results.append({
            "token_id": int(idx),
            "token": tokenizer.decode([int(idx)]),
            "similarity": float(similarities[idx]),
        })
    return results

