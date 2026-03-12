# app.py — with 3-D visualization endpoints & fixed cluster extraction
import json, os, io, base64, traceback, math
import numpy as np
import torch
from flask import Flask, render_template, request, jsonify

from model_wrapper import UnifiedModelWrapper, MODEL_REGISTRY
from knowledge_extraction import SparseAutoencoder

app = Flask(__name__)

wrapper = None
sae_models = {}  # layer_idx → trained SAE


def get_wrapper():
    global wrapper
    if wrapper is None:
        wrapper = UnifiedModelWrapper('gpt2')
    return wrapper


# ── Torch-based PCA (no sklearn needed) ────────────────────
def pca_reduce(data_tensor, n_components=3):
    """data_tensor: [N, D] → [N, n_components] numpy"""
    data = data_tensor.float()
    data = data - data.mean(0, keepdim=True)
    U, S, Vh = torch.linalg.svd(data, full_matrices=False)
    projected = data @ Vh[:n_components].T
    explained = (S[:n_components] ** 2) / (S ** 2).sum()
    return projected.numpy(), explained.numpy()


# ════════════════════════════════════════════════════════════
#  Pages
# ════════════════════════════════════════════════════════════

@app.route('/')
def index():
    return render_template('index.html')


# ════════════════════════════════════════════════════════════
#  Model management
# ════════════════════════════════════════════════════════════

@app.route('/api/available_models', methods=['GET'])
def available_models():
    """Return the list of models that can be loaded."""
    models = []
    for key, info in MODEL_REGISTRY.items():
        models.append({'key': key, 'hf': info['hf'],
                       'label': info['label'], 'family': info['family']})
    return jsonify(models)


@app.route('/api/load_model', methods=['POST'])
def load_model():
    try:
        data = request.json
        model_key = data.get('model_name', 'gpt2')
        global wrapper, sae_models
        sae_models = {}                       # reset SAEs on model change
        wrapper = UnifiedModelWrapper(model_key)
        return jsonify({'status': 'ok', 'info': wrapper.get_model_info()})
    except Exception as e:
        traceback.print_exc()
        return jsonify({'status': 'error', 'message': str(e)}), 500


@app.route('/api/model_info', methods=['GET'])
def model_info():
    try:
        return jsonify(get_wrapper().get_model_info())
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500


# ════════════════════════════════════════════════════════════
#  Forward pass
# ════════════════════════════════════════════════════════════

@app.route('/api/forward', methods=['POST'])
def forward_pass():
    try:
        w = get_wrapper()
        text = request.json.get('text', '')
        if not text:
            return jsonify({'status': 'error', 'message': 'Kein Text'}), 400

        result = w.forward_with_activations(text)

        act_summary = {}
        for key, tensor in result['activations'].items():
            flat = tensor.numpy().flatten()
            act_summary[key] = {
                'shape': list(tensor.shape),
                'mean': float(tensor.mean()),
                'std': float(tensor.std()),
                'l2_norm': float(tensor.norm()),
                'histogram': np.histogram(flat, bins=50)[0].tolist(),
            }

        attn_data = {}
        if result['attentions']:
            indices = sorted(set([0, len(result['attentions']) - 1]))
            for i in indices:
                att = result['attentions'][i]
                attn_data[f'layer_{i}'] = {
                    'shape': list(att.shape),
                    'values': att[0, 0].tolist(),
                }

        return jsonify({
            'status': 'ok',
            'tokens': result['tokens'],
            'predictions': result['predictions'],
            'activations': act_summary,
            'attentions': attn_data,
        })
    except Exception as e:
        traceback.print_exc()
        return jsonify({'status': 'error', 'message': str(e)}), 500


# ════════════════════════════════════════════════════════════
#  Causal trace
# ════════════════════════════════════════════════════════════

@app.route('/api/causal_trace', methods=['POST'])
def causal_trace():
    try:
        w = get_wrapper()
        prompt = request.json.get('prompt', '')
        if not prompt:
            return jsonify({'status': 'error', 'message': 'Kein Prompt'}), 400
        result = w.causal_trace(prompt)
        return jsonify({'status': 'ok', 'result': result})
    except Exception as e:
        traceback.print_exc()
        return jsonify({'status': 'error', 'message': str(e)}), 500


# ════════════════════════════════════════════════════════════
#  SAE training
# ════════════════════════════════════════════════════════════

@app.route('/api/train_sae', methods=['POST'])
def train_sae():
    try:
        w = get_wrapper()
        data = request.json
        layer_idx = data.get('layer_idx', 6)
        n_features = data.get('n_features', 2048)
        n_steps = data.get('n_steps', 200)
        sample_texts = data.get('sample_texts', [
            "The capital of France is",
            "Berlin is the capital of",
            "Water boils at a temperature of",
            "The largest planet in our solar system is",
            "Python is a programming language that",
            "The speed of light is approximately",
            "Shakespeare wrote the play",
            "The chemical formula for water is",
            "Machine learning is a subset of",
            "The Earth revolves around the",
        ])

        all_act = []
        for text in sample_texts:
            act = w.get_activations(text, layer=layer_idx)
            if act is not None:
                all_act.append(act)
        if not all_act:
            return jsonify({'status': 'error',
                            'message': 'Keine Aktivierungen'}), 400

        act_tensor = torch.stack(all_act)
        sae = SparseAutoencoder(w.d_model, n_features)
        opt = torch.optim.Adam(sae.parameters(), lr=1e-3)

        losses = []
        for step in range(n_steps):
            opt.zero_grad()
            _, _, loss = sae(act_tensor)
            loss.backward()
            opt.step()
            losses.append(float(loss))

        sae_models[layer_idx] = sae
        return jsonify({
            'status': 'ok', 'layer_idx': layer_idx,
            'n_features': n_features, 'final_loss': losses[-1],
            'loss_history': losses,
            'message': f'SAE Layer {layer_idx} trained ({n_steps} steps)',
        })
    except Exception as e:
        traceback.print_exc()
        return jsonify({'status': 'error', 'message': str(e)}), 500


# ════════════════════════════════════════════════════════════
#  Cluster extraction — FIXED
# ════════════════════════════════════════════════════════════

@app.route('/api/extract_cluster', methods=['POST'])
def extract_cluster():
    try:
        w = get_wrapper()
        data = request.json
        layer_idx = data.get('layer_idx', 6)
        domain_prompts = data.get('domain_prompts', [])
        threshold = data.get('threshold', 0.1)

        if not domain_prompts:
            return jsonify({'status': 'error',
                            'message': 'Keine Domain-Prompts'}), 400
        if layer_idx not in sae_models:
            return jsonify({'status': 'error',
                            'message': f'Kein SAE für Layer {layer_idx}'}), 400

        sae = sae_models[layer_idx]
        cluster_data = {}
        prompt_features = []

        for prompt in domain_prompts:
            act = w.get_activations(prompt, layer=layer_idx)
            if act is None:
                continue
            features = sae.extract_features(act.unsqueeze(0), threshold)
            indices = features['feature_indices']   # [N, 2]
            strengths = features['feature_strengths']

            if indices.numel() == 0:
                prompt_features.append({
                    'prompt': prompt,
                    'n_active_features': 0,
                    'top_features': [],
                })
                continue

            # ── FIX: safe column indexing ──
            feat_idx_list = indices[:, 1].tolist() if indices.dim() == 2 \
                else indices.tolist()
            feat_str_list = strengths.tolist()

            prompt_features.append({
                'prompt': prompt,
                'n_active_features': len(feat_idx_list),
                'top_features': sorted(
                    zip(feat_idx_list, feat_str_list),
                    key=lambda x: -x[1])[:20],
            })

            for idx, strength in zip(feat_idx_list, feat_str_list):
                cluster_data.setdefault(idx, []).append(strength)

        # Shared features
        shared_features = []
        for feat_idx, strs in cluster_data.items():
            rate = len(strs) / len(domain_prompts)
            if rate > 0.5:
                shared_features.append({
                    'feature_idx': int(feat_idx),
                    'activation_rate': rate,
                    'mean_strength': float(np.mean(strs)),
                    'std_strength': float(np.std(strs)),
                })
        shared_features.sort(key=lambda x: -x['mean_strength'])

        # SVD
        if shared_features:
            feat_idxs = [f['feature_idx'] for f in shared_features]
            V = torch.stack([sae.decoder.weight[:, i] for i in feat_idxs])
            Q, R = torch.linalg.qr(V.T)
            sv = torch.linalg.svdvals(Q[:, :len(feat_idxs)])
            svd_info = {
                'effective_rank': int((sv > 0.01).sum()),
                'singular_values': sv.tolist()[:20],
                'subspace_dim': len(feat_idxs),
            }
        else:
            svd_info = {'effective_rank': 0, 'singular_values': [],
                        'subspace_dim': 0}

        return jsonify({
            'status': 'ok',
            'prompt_features': prompt_features,
            'shared_features': shared_features[:30],
            'svd_info': svd_info,
            'n_total_features': len(cluster_data),
            'n_shared_features': len(shared_features),
        })
    except Exception as e:
        traceback.print_exc()
        return jsonify({'status': 'error', 'message': str(e)}), 500


@app.route('/api/serialize_cluster', methods=['POST'])
def serialize_cluster_endpoint():
    try:
        w = get_wrapper()
        data = request.json
        layer_idx = data.get('layer_idx', 6)
        feature_indices = data.get('feature_indices', [])

        if layer_idx not in sae_models:
            return jsonify({'status': 'error',
                            'message': 'Kein SAE trainiert'}), 400

        sae = sae_models[layer_idx]
        serialized = {
            'metadata': {
                'model': w.hf_name, 'layer': layer_idx,
                'n_features': len(feature_indices), 'd_model': w.d_model,
            },
            'features': {},
        }
        for idx in feature_indices:
            d = sae.decoder.weight[:, idx].detach()
            serialized['features'][str(idx)] = {
                'direction': d.half().numpy().tolist(),
                'norm': float(d.norm()),
            }

        if feature_indices:
            V = torch.stack([sae.decoder.weight[:, i] for i in feature_indices])
            U, S, Vh = torch.linalg.svd(V, full_matrices=False)
            serialized['svd'] = {
                'U': U.half().numpy().tolist(),
                'S': S.numpy().tolist(),
                'Vh': Vh.half().numpy().tolist(),
            }

        json_bytes = json.dumps(serialized, indent=2).encode()
        encoded = base64.b64encode(json_bytes).decode()
        return jsonify({
            'status': 'ok', 'data_b64': encoded,
            'size_kb': len(json_bytes) / 1024,
            'filename': f'cluster_L{layer_idx}_F{len(feature_indices)}.json',
        })
    except Exception as e:
        traceback.print_exc()
        return jsonify({'status': 'error', 'message': str(e)}), 500


# ════════════════════════════════════════════════════════════
#  3-D VISUALIZATION ENDPOINTS
# ════════════════════════════════════════════════════════════

@app.route('/api/vis3d/layer_trajectory', methods=['POST'])
def vis3d_layer_trajectory():
    """
    PCA-3D trajectory of the last-token activation through every layer.
    Shows how the representation evolves from input to output.
    """
    try:
        w = get_wrapper()
        text = request.json.get('text', '')
        if not text:
            return jsonify({'status': 'error', 'message': 'Kein Text'}), 400

        acts = w.get_all_layer_activations(text)  # {layer_idx: [d_model]}
        if len(acts) < 3:
            return jsonify({'status': 'error',
                            'message': 'Mindestens 3 Layer nötig'}), 400

        indices = sorted(acts.keys())
        mat = torch.stack([acts[i] for i in indices])  # [n_layers, d_model]
        projected, explained = pca_reduce(mat, 3)

        points = []
        for i, idx in enumerate(indices):
            points.append({
                'x': float(projected[i, 0]),
                'y': float(projected[i, 1]),
                'z': float(projected[i, 2]),
                'layer': int(idx),
                'norm': float(mat[i].norm()),
            })

        return jsonify({
            'status': 'ok', 'points': points,
            'explained_variance': explained.tolist(),
        })
    except Exception as e:
        traceback.print_exc()
        return jsonify({'status': 'error', 'message': str(e)}), 500


@app.route('/api/vis3d/feature_space', methods=['POST'])
def vis3d_feature_space():
    """
    PCA-3D scatter of SAE decoder directions (feature vectors).
    Colour / size encode the activation strength for the given text.
    """
    try:
        w = get_wrapper()
        data = request.json
        text = data.get('text', '')
        layer_idx = data.get('layer_idx', 6)

        if layer_idx not in sae_models:
            return jsonify({'status': 'error',
                            'message': f'Kein SAE für Layer {layer_idx}'}), 400
        sae = sae_models[layer_idx]

        # All decoder directions
        W = sae.decoder.weight.detach().T  # [n_features, d_model]
        n_feat = W.shape[0]

        # Activation strengths for this text
        act = w.get_activations(text, layer=layer_idx) if text else None
        if act is not None:
            with torch.no_grad():
                fa = torch.relu(sae.encoder(act.unsqueeze(0))).squeeze(0)  # [n_feat]
            strengths = fa.numpy()
        else:
            strengths = np.zeros(n_feat)

        # PCA down to 3D (subsample if too many features)
        max_plot = 1000
        if n_feat > max_plot:
            # Keep top-activated features + random sample
            top_k = min(200, n_feat)
            top_idx = np.argsort(-strengths)[:top_k]
            rest = np.setdiff1d(np.arange(n_feat), top_idx)
            sample_idx = np.random.choice(rest, max_plot - top_k, replace=False)
            sel = np.concatenate([top_idx, sample_idx])
        else:
            sel = np.arange(n_feat)

        W_sel = W[sel]
        projected, explained = pca_reduce(W_sel, 3)

        points = []
        for i, fi in enumerate(sel):
            points.append({
                'x': float(projected[i, 0]),
                'y': float(projected[i, 1]),
                'z': float(projected[i, 2]),
                'feature_idx': int(fi),
                'strength': float(strengths[fi]),
            })

        return jsonify({
            'status': 'ok', 'points': points,
            'explained_variance': explained.tolist(),
        })
    except Exception as e:
        traceback.print_exc()
        return jsonify({'status': 'error', 'message': str(e)}), 500


@app.route('/api/vis3d/activation_landscape', methods=['POST'])
def vis3d_activation_landscape():
    """
    3-D surface plot of a layer's MLP activation vector,
    reshaped to a 2-D grid so we can see the 'terrain'.
    """
    try:
        w = get_wrapper()
        data = request.json
        text = data.get('text', '')
        layer_idx = data.get('layer_idx', 0)

        act = w.get_activations(text, layer=layer_idx)
        if act is None:
            return jsonify({'status': 'error',
                            'message': 'Keine Aktivierung'}), 400

        vals = act.numpy()
        d = len(vals)
        side = int(math.ceil(math.sqrt(d)))
        padded = np.zeros(side * side)
        padded[:d] = vals
        Z = padded.reshape(side, side)

        return jsonify({
            'status': 'ok',
            'z': Z.tolist(),
            'side': side,
            'd_model': d,
            'layer_idx': layer_idx,
            'stats': {
                'mean': float(vals.mean()),
                'std': float(vals.std()),
                'max': float(vals.max()),
                'min': float(vals.min()),
            },
        })
    except Exception as e:
        traceback.print_exc()
        return jsonify({'status': 'error', 'message': str(e)}), 500


@app.route('/api/vis3d/multi_prompt', methods=['POST'])
def vis3d_multi_prompt():
    """
    PCA-3D scatter of multiple prompts' activations at one layer.
    Shows how different inputs cluster in the representation space.
    """
    try:
        w = get_wrapper()
        data = request.json
        prompts = data.get('prompts', [])
        layer_idx = data.get('layer_idx', 6)

        if len(prompts) < 3:
            return jsonify({'status': 'error',
                            'message': 'Mindestens 3 Prompts nötig'}), 400

        acts = []
        labels = []
        for p in prompts:
            a = w.get_activations(p, layer=layer_idx)
            if a is not None:
                acts.append(a)
                labels.append(p[:40])

        if len(acts) < 3:
            return jsonify({'status': 'error',
                            'message': 'Zu wenige Aktivierungen'}), 400

        mat = torch.stack(acts)
        projected, explained = pca_reduce(mat, 3)

        points = [{'x': float(projected[i, 0]),
                    'y': float(projected[i, 1]),
                    'z': float(projected[i, 2]),
                    'label': labels[i]}
                   for i in range(len(labels))]

        return jsonify({
            'status': 'ok', 'points': points,
            'explained_variance': explained.tolist(),
        })
    except Exception as e:
        traceback.print_exc()
        return jsonify({'status': 'error', 'message': str(e)}), 500


@app.route('/api/weight_stats/<int:layer_idx>', methods=['GET'])
def weight_stats(layer_idx):
    try:
        return jsonify({'status': 'ok',
                        'stats': get_wrapper().get_weight_stats(layer_idx)})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500


# ════════════════════════════════════════════════════════════

if __name__ == '__main__':
    print("=" * 60)
    print("  Holographic Knowledge Extraction GUI")
    print("  http://localhost:5000")
    print("=" * 60)
    app.run(debug=True, port=5000)
