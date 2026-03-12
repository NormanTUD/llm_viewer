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

# ── Add to the top of app.py, after existing imports ────────
from sklearn.manifold import TSNE

# Optional UMAP — graceful fallback
try:
    from umap import UMAP
    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False


def reduce_dims(data_tensor, n_components=3, method='pca', **kwargs):
    """
    Unified dimensionality reduction.
    method: 'pca' | 'tsne' | 'umap'
    Returns (projected_np, meta_dict)
    """
    data = data_tensor.float()
    N = data.shape[0]

    if method == 'pca':
        proj, expl = pca_reduce(data, n_components)
        return proj, {'explained_variance': expl.tolist()}

    elif method == 'tsne':
        perplexity = kwargs.get('perplexity', min(30, max(5, N - 1)))
        tsne = TSNE(n_components=min(n_components, 3),
                     perplexity=perplexity,
                     learning_rate='auto',
                     init='pca',
                     random_state=42)
        proj = tsne.fit_transform(data.numpy())
        return proj, {'kl_divergence': float(tsne.kl_divergence_),
                      'perplexity': perplexity}

    elif method == 'umap':
        if not HAS_UMAP:
            # Fall back to t-SNE if umap not installed
            return reduce_dims(data_tensor, n_components, 'tsne', **kwargs)
        n_neighbors = kwargs.get('n_neighbors', min(15, max(2, N - 1)))
        min_dist = kwargs.get('min_dist', 0.1)
        reducer = UMAP(n_components=n_components,
                       n_neighbors=n_neighbors,
                       min_dist=min_dist,
                       random_state=42)
        proj = reducer.fit_transform(data.numpy())
        return proj, {'n_neighbors': n_neighbors, 'min_dist': min_dist}

    else:
        raise ValueError(f"Unknown method: {method}")

# ── Add to app.py ───────────────────────────────────────────
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram as scipy_dendro
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score


@app.route('/api/vis3d/subclusters', methods=['POST'])
def vis3d_subclusters():
    """
    Hierarchical subclustering of prompts at a given layer.
    Returns cluster assignments, dendrogram data, and 3D scatter.
    """
    try:
        w = get_wrapper()
        data = request.json
        prompts = data.get('prompts', [])
        layer_idx = data.get('layer_idx', 6)
        method = data.get('reduction', 'pca')       # pca | tsne | umap
        n_clusters = data.get('n_clusters', None)    # None = auto via DBSCAN
        linkage_method = data.get('linkage', 'ward') # ward | average | complete

        if len(prompts) < 4:
            return jsonify({'status': 'error',
                            'message': 'Need ≥ 4 prompts for subclustering'}), 400

        # Gather activations
        acts, labels = [], []
        for p in prompts:
            a = w.get_activations(p, layer=layer_idx)
            if a is not None:
                acts.append(a)
                labels.append(p[:50])
        if len(acts) < 4:
            return jsonify({'status': 'error',
                            'message': 'Too few valid activations'}), 400

        mat = torch.stack(acts)           # [N, d_model]
        mat_np = mat.numpy()

        # ── Hierarchical linkage ──
        Z = linkage(mat_np, method=linkage_method, metric='cosine')

        # ── Cluster assignment ──
        if n_clusters is not None:
            cluster_ids = fcluster(Z, t=n_clusters, criterion='maxclust')
        else:
            # Auto-detect with DBSCAN on the cosine distance matrix
            from sklearn.metrics.pairwise import cosine_distances
            dist_mat = cosine_distances(mat_np)
            eps = float(np.median(dist_mat[dist_mat > 0]) * 0.6)
            db = DBSCAN(eps=eps, min_samples=2, metric='precomputed')
            cluster_ids = db.fit_predict(dist_mat)
            # Relabel noise (-1) as its own cluster
            if -1 in cluster_ids:
                cluster_ids[cluster_ids == -1] = cluster_ids.max() + 1
            cluster_ids = cluster_ids + 1   # 1-based

        n_found = len(set(cluster_ids))

        # ── Silhouette score (if ≥ 2 clusters) ──
        sil = -1.0
        if n_found >= 2 and n_found < len(acts):
            sil = float(silhouette_score(mat_np, cluster_ids, metric='cosine'))

        # ── Dimensionality reduction for scatter ──
        projected, red_meta = reduce_dims(mat, n_components=3, method=method)

        # ── Dendrogram structure (serialisable) ──
        dendro = scipy_dendro(Z, no_plot=True, labels=labels)
        dendro_data = {
            'icoord': [list(map(float, x)) for x in dendro['icoord']],
            'dcoord': [list(map(float, x)) for x in dendro['dcoord']],
            'ivl':    dendro['ivl'],
            'color_list': dendro.get('color_list', []),
        }

        # ── Per-cluster centroid in 3D ──
        centroids_3d = []
        for cid in sorted(set(cluster_ids)):
            mask = cluster_ids == cid
            centroid = projected[mask].mean(axis=0)
            centroids_3d.append({
                'cluster': int(cid),
                'x': float(centroid[0]),
                'y': float(centroid[1]),
                'z': float(centroid[2]),
                'size': int(mask.sum()),
            })

        points = []
        for i in range(len(labels)):
            points.append({
                'x': float(projected[i, 0]),
                'y': float(projected[i, 1]),
                'z': float(projected[i, 2]),
                'label': labels[i],
                'cluster': int(cluster_ids[i]),
            })

        return jsonify({
            'status': 'ok',
            'points': points,
            'centroids': centroids_3d,
            'n_clusters': n_found,
            'silhouette': sil,
            'dendrogram': dendro_data,
            'reduction_meta': red_meta,
            'method': method,
        })
    except Exception as e:
        traceback.print_exc()
        return jsonify({'status': 'error', 'message': str(e)}), 500

# ── Add to app.py ───────────────────────────────────────────

@app.route('/api/vis3d/spiral_analysis', methods=['POST'])
def vis3d_spiral_analysis():
    """
    Detect and visualise spiral structure in SAE feature space.
    Projects to 2D, converts to polar coordinates, fits Archimedean
    spiral arms, and returns both Cartesian and polar views.
    """
    try:
        w = get_wrapper()
        data = request.json
        text = data.get('text', '')
        layer_idx = data.get('layer_idx', 6)
        method = data.get('reduction', 'pca')
        n_arms = data.get('n_arms', 3)              # max spiral arms to detect

        if layer_idx not in sae_models:
            return jsonify({'status': 'error',
                            'message': f'No SAE for layer {layer_idx}'}), 400
        sae = sae_models[layer_idx]

        # Decoder directions
        W = sae.decoder.weight.detach().T            # [n_features, d_model]
        n_feat = W.shape[0]

        # Activation strengths
        act = w.get_activations(text, layer=layer_idx) if text else None
        if act is not None:
            with torch.no_grad():
                fa = torch.relu(sae.encoder(act.unsqueeze(0))).squeeze(0)
            strengths = fa.numpy()
        else:
            strengths = np.zeros(n_feat)

        # Subsample if huge
        max_pts = 1500
        if n_feat > max_pts:
            top_k = min(300, n_feat)
            top_idx = np.argsort(-strengths)[:top_k]
            rest = np.setdiff1d(np.arange(n_feat), top_idx)
            sample_idx = np.random.choice(rest, max_pts - top_k, replace=False)
            sel = np.concatenate([top_idx, sample_idx])
        else:
            sel = np.arange(n_feat)

        W_sel = W[sel]
        str_sel = strengths[sel]

        # ── 2D projection for polar analysis ──
        proj2d, meta2d = reduce_dims(W_sel, n_components=2, method=method)

        # ── 3D projection for scatter ──
        proj3d, meta3d = reduce_dims(W_sel, n_components=3, method=method)

        # ── Convert to polar coordinates ──
        cx, cy = proj2d.mean(axis=0)                 # centroid
        dx = proj2d[:, 0] - cx
        dy = proj2d[:, 1] - cy
        r = np.sqrt(dx**2 + dy**2)
        theta = np.arctan2(dy, dx)                   # [-π, π]

        # ── Spiral arm detection ──
        # Sort by angle, fit piecewise linear r(θ) to find arms
        order = np.argsort(theta)
        theta_sorted = theta[order]
        r_sorted = r[order]

        # Bin into angular slices and find radial peaks
        n_bins = 60
        bin_edges = np.linspace(-np.pi, np.pi, n_bins + 1)
        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
        bin_r_mean = np.zeros(n_bins)
        bin_r_std = np.zeros(n_bins)
        bin_count = np.zeros(n_bins)
        for bi in range(n_bins):
            mask = (theta >= bin_edges[bi]) & (theta < bin_edges[bi + 1])
            if mask.sum() > 0:
                bin_r_mean[bi] = r[mask].mean()
                bin_r_std[bi] = r[mask].std()
                bin_count[bi] = mask.sum()

        # Fit Archimedean spiral: r = a + b*θ
        valid = bin_count > 2
        spiral_fits = []
        if valid.sum() > 10:
            from numpy.polynomial import polynomial as P
            coeffs = P.polyfit(bin_centers[valid], bin_r_mean[valid], deg=1)
            a_fit, b_fit = float(coeffs[0]), float(coeffs[1])

            # Generate fitted spiral curve(s)
            for arm in range(n_arms):
                offset = arm * (2 * np.pi / n_arms)
                t_curve = np.linspace(-np.pi, np.pi, 200)
                r_curve = a_fit + b_fit * (t_curve + offset)
                r_curve = np.clip(r_curve, 0, r.max() * 1.2)
                x_curve = cx + r_curve * np.cos(t_curve)
                y_curve = cy + r_curve * np.sin(t_curve)
                spiral_fits.append({
                    'arm': arm,
                    'x': x_curve.tolist(),
                    'y': y_curve.tolist(),
                    'r': r_curve.tolist(),
                    'theta': t_curve.tolist(),
                })

            # Spiral coherence score: how well r correlates with θ
            corr = float(np.corrcoef(theta[r > 0], r[r > 0])[0, 1]) \
                if (r > 0).sum() > 10 else 0.0
        else:
            a_fit, b_fit, corr = 0, 0, 0
            spiral_fits = []

        # ── Build point arrays ──
        points_2d = []
        points_3d = []
        points_polar = []
        for i, fi in enumerate(sel):
            points_2d.append({
                'x': float(proj2d[i, 0]), 'y': float(proj2d[i, 1]),
                'feature_idx': int(fi), 'strength': float(str_sel[i]),
            })
            points_3d.append({
                'x': float(proj3d[i, 0]),
                'y': float(proj3d[i, 1]),
                'z': float(proj3d[i, 2]),
                'feature_idx': int(fi), 'strength': float(str_sel[i]),
            })
            points_polar.append({
                'r': float(r[i]), 'theta': float(theta[i]),
                'feature_idx': int(fi), 'strength': float(str_sel[i]),
            })

        return jsonify({
            'status': 'ok',
            'points_2d': points_2d,
            'points_3d': points_3d,
            'points_polar': points_polar,
            'spiral_fits': spiral_fits,
            'spiral_params': {
                'a': a_fit, 'b': b_fit,
                'coherence': corr,
                'n_arms_fitted': len(spiral_fits),
            },
            'angular_profile': {
                'bin_centers': bin_centers.tolist(),
                'mean_r': bin_r_mean.tolist(),
                'std_r': bin_r_std.tolist(),
                'count': bin_count.tolist(),
            },
            'reduction_meta': meta3d,
        })
    except Exception as e:
        traceback.print_exc()
        return jsonify({'status': 'error', 'message': str(e)}), 500


# ── Add to app.py ───────────────────────────────────────────

@app.route('/api/vis3d/slice_view', methods=['POST'])
def vis3d_slice_view():
    """
    True 2D or 3D slice through activation space.
    Axes = actual neuron indices (not projected).
    Plots multiple prompts in this real subspace.
    """
    try:
        w = get_wrapper()
        data = request.json
        prompts = data.get('prompts', [])
        layer_idx = data.get('layer_idx', 6)
        # Neuron indices to use as axes (2 or 3)
        axes = data.get('axes', None)              # e.g. [0, 128, 512]
        auto_select = data.get('auto_select', True)  # pick highest-variance neurons

        if len(prompts) < 2:
            return jsonify({'status': 'error',
                            'message': 'Need ≥ 2 prompts'}), 400

        acts, labels = [], []
        for p in prompts:
            a = w.get_activations(p, layer=layer_idx)
            if a is not None:
                acts.append(a)
                labels.append(p[:50])
        if len(acts) < 2:
            return jsonify({'status': 'error',
                            'message': 'Too few activations'}), 400

        mat = torch.stack(acts)  # [N, d_model]
        mat_np = mat.numpy()

        # ── Auto-select axes by variance ──
        if axes is None or auto_select:
            variances = mat_np.var(axis=0)
            top_var = np.argsort(-variances)
            n_axes = 3 if len(data.get('axes', [0, 0, 0])) == 3 else 3
            axes = top_var[:n_axes].tolist()

        n_dims = len(axes)
        if n_dims < 2:
            axes = axes + [0] * (2 - n_dims)
        axes = axes[:3]

        points = []
        for i in range(len(labels)):
            pt = {'label': labels[i]}
            pt['x'] = float(mat_np[i, axes[0]])
            pt['y'] = float(mat_np[i, axes[1]])
            if len(axes) >= 3:
                pt['z'] = float(mat_np[i, axes[2]])
            points.append(pt)

        # ── Decision boundary heatmap (2D grid sampling) ──
        # Interpolate a grid in the 2D plane defined by axes[0], axes[1]
        # holding all other dims at the mean
        grid_res = data.get('grid_resolution', 40)
        mean_act = mat_np.mean(axis=0)

        x_vals = mat_np[:, axes[0]]
        y_vals = mat_np[:, axes[1]]
        x_range = np.linspace(x_vals.min() - 0.5, x_vals.max() + 0.5, grid_res)
        y_range = np.linspace(y_vals.min() - 0.5, y_vals.max() + 0.5, grid_res)

        # For each grid point, find nearest prompt (Voronoi-style decision boundary)
        grid_labels = np.zeros((grid_res, grid_res), dtype=int)
        grid_distances = np.zeros((grid_res, grid_res))

        from scipy.spatial import KDTree
        prompt_coords = np.column_stack([x_vals, y_vals])
        tree = KDTree(prompt_coords)

        for xi, xv in enumerate(x_range):
            for yi, yv in enumerate(y_range):
                dist, idx = tree.query([xv, yv])
                grid_labels[yi, xi] = idx
                grid_distances[yi, xi] = dist

        return jsonify({
            'status': 'ok',
            'points': points,
            'axes': axes,
            'axes_variances': [float(mat_np[:, a].var()) for a in axes],
            'n_dims': len(axes),
            'decision_grid': {
                'labels': grid_labels.tolist(),
                'distances': grid_distances.tolist(),
                'x_range': x_range.tolist(),
                'y_range': y_range.tolist(),
            },
        })
    except Exception as e:
        traceback.print_exc()
        return jsonify({'status': 'error', 'message': str(e)}), 500


@app.route('/api/vis3d/neuron_pair_heatmap', methods=['POST'])
def vis3d_neuron_pair_heatmap():
    """
    Heatmap of activation co-occurrence for two neurons across many prompts.
    Shows correlation structure in *real* neuron space.
    """
    try:
        w = get_wrapper()
        data = request.json
        prompts = data.get('prompts', [])
        layer_idx = data.get('layer_idx', 6)
        top_k = data.get('top_k', 20)  # how many neuron pairs to show

        acts = []
        for p in prompts:
            a = w.get_activations(p, layer=layer_idx)
            if a is not None:
                acts.append(a.numpy())
        if len(acts) < 3:
            return jsonify({'status': 'error',
                            'message': 'Need ≥ 3 prompts'}), 400

        mat = np.stack(acts)  # [N, d_model]

        # Correlation matrix of neurons (across prompts)
        # Only keep high-variance neurons
        variances = mat.var(axis=0)
        top_neurons = np.argsort(-variances)[:top_k]

        sub = mat[:, top_neurons]  # [N, top_k]
        corr = np.corrcoef(sub.T)  # [top_k, top_k]

        return jsonify({
            'status': 'ok',
            'correlation_matrix': corr.tolist(),
            'neuron_indices': top_neurons.tolist(),
            'neuron_variances': variances[top_neurons].tolist(),
            'n_prompts': len(acts),
        })
    except Exception as e:
        traceback.print_exc()
        return jsonify({'status': 'error', 'message': str(e)}), 500

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


# ── Replace the existing vis3d_multi_prompt in app.py ───────

@app.route('/api/vis3d/multi_prompt', methods=['POST'])
def vis3d_multi_prompt():
    """
    PCA / t-SNE / UMAP 3D scatter of multiple prompts' activations.
    Now with cluster colouring and method selection.
    """
    try:
        w = get_wrapper()
        data = request.json
        prompts = data.get('prompts', [])
        layer_idx = data.get('layer_idx', 6)
        method = data.get('reduction', 'pca')

        if len(prompts) < 3:
            return jsonify({'status': 'error',
                            'message': 'Need ≥ 3 prompts'}), 400

        acts, labels = [], []
        for p in prompts:
            a = w.get_activations(p, layer=layer_idx)
            if a is not None:
                acts.append(a)
                labels.append(p[:40])

        if len(acts) < 3:
            return jsonify({'status': 'error',
                            'message': 'Too few activations'}), 400

        mat = torch.stack(acts)
        projected, meta = reduce_dims(mat, 3, method=method)

        points = [{'x': float(projected[i, 0]),
                    'y': float(projected[i, 1]),
                    'z': float(projected[i, 2]),
                    'label': labels[i]}
                   for i in range(len(labels))]

        return jsonify({
            'status': 'ok', 'points': points,
            'reduction_meta': meta,
            'method': method,
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
