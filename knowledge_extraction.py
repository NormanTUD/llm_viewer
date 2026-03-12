# knowledge_extraction.py
# Füge ganz oben hinzu falls nicht vorhanden:

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import defaultdict
from datetime import datetime



def extract_weight_subspace(model, layer_idx, feature_indices, sae):
    """
    Projiziert die Modellgewichte auf den Subspace
    der identifizierten Features (SVD-basiert).
    """
    W = model.layers[layer_idx].mlp.weight.detach()

    # Feature-Richtungen als Basis des Subspace
    V = torch.stack([
        sae.decoder.weight[:, i] for i in feature_indices
    ])  # [n_cluster_features, d_model]

    # Orthogonalisierung via QR-Zerlegung
    Q, R = torch.linalg.qr(V.T)

    # Projektion der Gewichte auf den Subspace
    W_projected = Q @ Q.T @ W

    # Low-Rank Approximation via SVD
    U, S, Vh = torch.linalg.svd(W_projected, full_matrices=False)

    # Behalte nur signifikante Singulärwerte
    rank = (S > S[0] * 0.01).sum()

    return {
        'U': U[:, :rank],
        'S': S[:rank],
        'Vh': Vh[:rank, :],
        'effective_rank': rank.item()
    }

class SparseAutoencoder(nn.Module):
    """
    Sparse Autoencoder for decomposing holographic activations
    into interpretable feature directions.
    (Based on Anthropic's "Scaling Monosemanticity", 2024)
    """
    def __init__(self, d_model, n_features, sparsity_coeff=1e-3):
        super().__init__()
        self.encoder = nn.Linear(d_model, n_features)
        self.decoder = nn.Linear(n_features, d_model)
        self.sparsity_coeff = sparsity_coeff

    def forward(self, x):
        feature_acts = F.relu(self.encoder(x))
        reconstructed = self.decoder(feature_acts)
        recon_loss = F.mse_loss(reconstructed, x)
        sparsity_loss = self.sparsity_coeff * feature_acts.abs().sum()
        return reconstructed, feature_acts, recon_loss + sparsity_loss

    def extract_features(self, x, threshold=0.1):
        """
        Returns active features, their strengths, and decoder directions.
        x: [batch, d_model] or [d_model]
        """
        with torch.no_grad():
            if x.dim() == 1:
                x = x.unsqueeze(0)
            feature_acts = F.relu(self.encoder(x))        # [batch, n_features]
            active = (feature_acts > threshold)             # [batch, n_features]

            # ── FIX: squeeze batch dimension for boolean indexing ──
            active_squeezed = active.squeeze(0)             # [n_features] 1-D bool

            indices = active.nonzero(as_tuple=False)        # [N, 2]
            strengths = feature_acts[active]                # [N]

            # Safe column selection with 1-D mask
            if active_squeezed.any():
                directions = self.decoder.weight[:, active_squeezed]  # [d_model, K]
            else:
                directions = torch.empty(self.decoder.weight.shape[0], 0)

            return {
                'feature_indices': indices,
                'feature_strengths': strengths,
                'feature_directions': directions,
            }


# ── Helper stubs referenced elsewhere (kept for import compat) ──

def extract_knowledge_subcluster(model, sae, domain_prompts, layer_idx):
    """Extract a knowledge sub-cluster for a domain."""
    all_features = defaultdict(list)
    for prompt in domain_prompts:
        activations = model.get_activations(prompt, layer=layer_idx)
        if activations is None:
            continue
        features = sae.extract_features(activations)
        indices = features['feature_indices']
        strengths = features['feature_strengths']
        if indices.numel() == 0:
            continue
        for row in range(indices.shape[0]):
            idx_val = indices[row, -1].item()       # last column = feature idx
            str_val = strengths[row].item()
            all_features[idx_val].append(str_val)

    cluster_features = {}
    for feat_idx, strs in all_features.items():
        rate = len(strs) / max(len(domain_prompts), 1)
        if rate > 0.7:
            cluster_features[feat_idx] = {
                'mean_strength': float(np.mean(strs)),
                'direction': sae.decoder.weight[:, feat_idx].detach(),
                'activation_rate': rate,
            }
    return cluster_features


def serialize_knowledge_cluster(cluster_features, sae, metadata=None):
    """Serialize a cluster to a plain dict (JSON-safe)."""
    if metadata is None:
        metadata = {}
    serialized = {
        'metadata': {**metadata, 'n_features': len(cluster_features),
                     'timestamp': datetime.now().isoformat()},
        'features': {},
    }
    for idx, data in cluster_features.items():
        direction = data['direction'] if isinstance(data['direction'], torch.Tensor) \
            else sae.decoder.weight[:, idx].detach()
        serialized['features'][str(idx)] = {
            'direction': direction.half().numpy().tolist(),
            'norm': float(direction.norm()),
        }
    return serialized


def deserialize_and_inject(serialized_data, target_model, layer_idx):
    """
    Injiziert extrahiertes Wissenscluster in ein anderes Modell.
    (Ähnlich zu LoRA-Adaptern, aber subcluster-spezifisch)
    """
    data = msgpack.unpackb(serialized_data)

    U = torch.from_numpy(np.frombuffer(data['numeric']['U']))
    S = torch.from_numpy(np.frombuffer(data['numeric']['S']))
    Vh = torch.from_numpy(np.frombuffer(data['numeric']['Vh']))

    # Rekonstruiere den Gewichts-Delta für dieses Subcluster
    W_delta = U @ torch.diag(S) @ Vh

    # Addiere zum Zielmodell (wie bei einem holographischen Overlay)
    with torch.no_grad():
        target_model.layers[layer_idx].mlp.weight += W_delta * alpha

    return target_model

