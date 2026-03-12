/**
 * slicer_viz.js — Dimension slicing and visualization
 */

document.getElementById('btn-slice')?.addEventListener('click', sliceAndVisualize);

// Token trajectory
document.getElementById('btn-trajectory')?.addEventListener('click', showTrajectory);

// Clustering
document.getElementById('btn-cluster')?.addEventListener('click', runClustering);

async function sliceAndVisualize() {
    if (!APP_STATE.traceKey) return;

    const layer = parseInt(document.getElementById('slicer-layer-select').value);
    const dimSpec = document.getElementById('slicer-dim-spec').value;
    const projection = document.getElementById('slicer-projection').value;

    const data = await apiPost('/api/slice', {
        trace_key: APP_STATE.traceKey,
        layer: layer,
        dim_spec: dimSpec,
        projection: projection,
    });

    // 3D plot
    const trace3d = {
        x: data.projected_3d.map(p => p[0]),
        y: data.projected_3d.map(p => p[1]),
        z: data.projected_3d.map(p => p[2]),
        mode: 'markers+text',
        type: 'scatter3d',
        text: APP_STATE.tokens,
        textposition: 'top center',
        textfont: { size: 10, color: '#c9d1d9' },
        hovertext: APP_STATE.tokens.map((t, i) =>
            `<b>"${t}"</b><br>Dims: ${data.n_dims_selected} selected`
        ),
        hoverinfo: 'text',
        marker: {
            size: 8,
            color: APP_STATE.tokens.map((_, i) => i),
            colorscale: 'Plasma',
            opacity: 0.9,
        },
    };

    Plotly.newPlot('slicer-plot-3d', [trace3d], {
        paper_bgcolor: '#161b22', plot_bgcolor: '#0d1117',
        font: { color: '#c9d1d9' },
        scene: { bgcolor: '#0d1117',
            xaxis: { gridcolor: '#30363d' },
            yaxis: { gridcolor: '#30363d' },
            zaxis: { gridcolor: '#30363d' },
        },
        title: `Layer ${layer} — Dims: ${dimSpec} (${data.n_dims_selected} dims) [3D]`,
        margin: { l: 0, r: 0, t: 40, b: 0 },
    }, { responsive: true });

    // 2D plot
    const trace2d = {
        x: data.projected_2d.map(p => p[0]),
        y: data.projected_2d.map(p => p[1]),
        mode: 'markers+text',
        type: 'scatter',
        text: APP_STATE.tokens,
        textposition: 'top center',
        textfont: { size: 10, color: '#c9d1d9' },
        hovertext: APP_STATE.tokens.map(t => `"${t}"`),
        hoverinfo: 'text',
        marker: {
            size: 10,
            color: APP_STATE.tokens.map((_, i) => i),
            colorscale: 'Plasma',
        },
    };

    Plotly.newPlot('slicer-plot-2d', [trace2d], {
        paper_bgcolor: '#161b22', plot_bgcolor: '#0d1117',
        font: { color: '#c9d1d9' },
        xaxis: { gridcolor: '#30363d' },
        yaxis: { gridcolor: '#30363d' },
        title: `Layer ${layer} — Dims: ${dimSpec} [2D]`,
        margin: { l: 40, r: 20, t: 40, b: 40 },
    }, { responsive: true });

    // Stats
    const statsDiv = document.getElementById('slicer-stats');
    statsDiv.innerHTML = `
        <h3>Dimension Statistics</h3>
        <p>Selected: ${data.n_dims_selected} dimensions</p>
        <table style="width:100%; border-collapse:collapse; margin-top:8px;">
            <tr style="border-bottom:1px solid #30363d;">
                <th>Dim</th><th>Mean</th><th>Std</th><th>Min</th><th>Max</th>
            </tr>
            ${data.dim_stats.slice(0, 30).map(s => `
                <tr style="border-bottom:1px solid #21262d;">
                    <td>${s.dim}</td>
                    <td>${s.mean.toFixed(3)}</td>
                    <td>${s.std.toFixed(3)}</td>
                    <td>${s.min.toFixed(3)}</td>
                    <td>${s.max.toFixed(3)}</td>
                </tr>
            `).join('')}
        </table>
        ${data.dim_stats.length > 30 ? `<p>... and ${data.dim_stats.length - 30} more</p>` : ''}
    `;

    // PCA variance
    if (data.pca_variance && data.pca_variance.length > 0) {
        const varTrace = {
            x: data.pca_variance.map((_, i) => `PC${i + 1}`),
            y: data.pca_variance,
            type: 'bar',
            marker: { color: '#8b5cf6' },
        };
        Plotly.newPlot('slicer-variance', [varTrace], {
            paper_bgcolor: '#161b22', plot_bgcolor: '#0d1117',
            font: { color: '#c9d1d9', size: 10 },
            title: 'PCA Explained Variance (sliced dims)',
            margin: { l: 40, r: 10, t: 40, b: 40 },
        }, { responsive: true });
    }
}

async function showTrajectory() {
    if (!APP_STATE.traceKey) return;

    const tokenIdx = parseInt(document.getElementById('traj-token-select').value);
    const projection = document.getElementById('traj-projection').value;

    const data = await apiPost('/api/token_trajectory', {
        trace_key: APP_STATE.traceKey,
        token_idx: tokenIdx,
        projection: projection,
    });

    const pts = data.projected_3d;
    const hoverTexts = data.hover_info.map(h => {
        let text = `<b>${h.layer === 'input' ? 'Input' : 'Layer ' + h.layer}</b>`;
        if (h.nearest_tokens) {
            text += '<br>Nearest:';
            h.nearest_tokens.slice(0, 5).forEach(n => {
                text += `<br>  "${n.token}" (${n.similarity.toFixed(3)})`;
            });
        }
        return text;
    });

    // Points
    const pointTrace = {
        x: pts.map(p => p[0]),
        y: pts.map(p => p[1]),
        z: pts.map(p => p[2]),
        mode: 'markers+text',
        type: 'scatter3d',
        text: data.hover_info.map(h => h.layer === 'input' ? 'IN' : `L${h.layer}`),
        textfont: { size: 9, color: '#c9d1d9' },
        hovertext: hoverTexts,
        hoverinfo: 'text',
        marker: {
            size: 6,
            color: data.hover_info.map((_, i) => i),
            colorscale: 'Turbo',
            colorbar: { title: 'Layer' },
        },
        name: 'Layer positions',
    };

    // Path (line connecting the dots)
    const lineTrace = {
        x: pts.map(p => p[0]),
        y: pts.map(p => p[1]),
        z: pts.map(p => p[2]),
        mode: 'lines',
        type: 'scatter3d',
        line: { color: '#58a6ff', width: 3 },
        hoverinfo: 'skip',
        name: 'Trajectory',
    };

    Plotly.newPlot('traj-plot-3d', [lineTrace, pointTrace], {
        paper_bgcolor: '#161b22', plot_bgcolor: '#0d1117',
        font: { color: '#c9d1d9' },
        scene: { bgcolor: '#0d1117',
            xaxis: { gridcolor: '#30363d' },
            yaxis: { gridcolor: '#30363d' },
            zaxis: { gridcolor: '#30363d' },
        },
        title: `Trajectory of "${APP_STATE.tokens[tokenIdx]}" through ${data.n_layers} layers`,
        margin: { l: 0, r: 0, t: 40, b: 0 },
    }, { responsive: true });

    // Info
    const infoPanel = document.getElementById('traj-info');
    infoPanel.innerHTML = `
        <h3>Token Trajectory</h3>
        <p><b>Token:</b> "${APP_STATE.tokens[tokenIdx]}" (position ${tokenIdx})</p>
        <p><b>Layers traversed:</b> ${data.n_layers}</p>
        <hr style="border-color:#30363d; margin:10px 0">
        <h4>Nearest tokens at each layer:</h4>
        ${data.hover_info.map(h => `
            <div style="margin:6px 0; padding:4px; border-left:2px solid #30363d; padding-left:8px;">
                <b>${h.layer === 'input' ? 'Input' : 'Layer ' + h.layer}:</b>
                ${h.nearest_tokens ? h.nearest_tokens.slice(0, 3).map(n =>
                    `"${n.token}" (${n.similarity.toFixed(3)})`
                ).join(', ') : 'N/A'}
            </div>
        `).join('')}
    `;
}

async function runClustering() {
    const mode = document.getElementById('cluster-mode').value;
    const nClusters = parseInt(document.getElementById('cluster-n').value);
    const sampleSize = parseInt(document.getElementById('cluster-sample').value);

    let data;

    if (mode === 'vocab') {
        data = await apiPost('/api/cluster_vocab', {
            model: APP_STATE.model,
            n_clusters: nClusters,
            sample_size: sampleSize,
            projection: 'pca',
        });

        const trace = {
            x: data.projected_3d.map(p => p[0]),
            y: data.projected_3d.map(p => p[1]),
            z: data.projected_3d.map(p => p[2]),
            mode: 'markers',
            type: 'scatter3d',
            hovertext: data.tokens.map((t, i) =>
                `"${t}" (cluster ${data.cluster_labels[i]})`
            ),
            hoverinfo: 'text',
            marker: {
                size: 3,
                color: data.cluster_labels,
                colorscale: 'Rainbow',
                opacity: 0.7,
            },
        };

        Plotly.newPlot('cluster-plot-3d', [trace], {
            paper_bgcolor: '#161b22', plot_bgcolor: '#0d1117',
            font: { color: '#c9d1d9' },
            scene: { bgcolor: '#0d1117',
                xaxis: { gridcolor: '#30363d' },
                yaxis: { gridcolor: '#30363d' },
                zaxis: { gridcolor: '#30363d' },
            },
            title: `Vocabulary Clusters (${data.n_clusters} clusters, ${data.tokens.length} tokens)`,
            margin: { l: 0, r: 0, t: 40, b: 0 },
        }, { responsive: true });

        // Info
        const clusterCounts = {};
        data.cluster_labels.forEach(l => { clusterCounts[l] = (clusterCounts[l] || 0) + 1; });
        const infoPanel = document.getElementById('cluster-info');
        infoPanel.innerHTML = `
            <h3>Cluster Info</h3>
            <p><b>Silhouette Score:</b> ${data.silhouette_score.toFixed(3)}</p>
            <p><b>Tokens sampled:</b> ${data.tokens.length}</p>
            <hr style="border-color:#30363d; margin:10px 0">
            <h4>Cluster sizes:</h4>
            ${Object.entries(clusterCounts).map(([k, v]) => `<p>Cluster ${k}: ${v} tokens</p>`).join('')}
            <hr style="border-color:#30363d; margin:10px 0">
            <h4>Sample tokens per cluster:</h4>
            ${Object.keys(clusterCounts).map(c => {
                const tokensInCluster = data.tokens.filter((_, i) => data.cluster_labels[i] == c).slice(0, 10);
                return `<p><b>Cluster ${c}:</b> ${tokensInCluster.map(t => `"${t}"`).join(', ')}</p>`;
            }).join('')}
        `;

    } else {
        // Cluster layer hidden states
        if (!APP_STATE.traceKey) { alert('Run analysis first!'); return; }
        const layer = parseInt(document.getElementById('slicer-layer-select')?.value || 0);
        data = await apiPost('/api/cluster_layer', {
            trace_key: APP_STATE.traceKey,
            layer: layer,
            n_clusters: Math.min(nClusters, APP_STATE.tokens.length),
        });

        const trace = {
            x: data.projected_3d.map(p => p[0]),
            y: data.projected_3d.map(p => p[1]),
            z: data.projected_3d.map(p => p[2]),
            mode: 'markers+text',
            type: 'scatter3d',
            text: APP_STATE.tokens,
            textfont: { size: 10, color: '#c9d1d9' },
            hovertext: APP_STATE.tokens.map((t, i) =>
                `"${t}" — cluster ${data.cluster_labels[i]}`
            ),
            hoverinfo: 'text',
            marker: {
                size: 10,
                color: data.cluster_labels,
                colorscale: 'Rainbow',
            },
        };

        Plotly.newPlot('cluster-plot-3d', [trace], {
            paper_bgcolor: '#161b22', plot_bgcolor: '#0d1117',
            font: { color: '#c9d1d9' },
            scene: { bgcolor: '#0d1117',
                xaxis: { gridcolor: '#30363d' },
                yaxis: { gridcolor: '#30363d' },
                zaxis: { gridcolor: '#30363d' },
            },
            title: `Layer ${data.layer} Hidden State Clusters`,
            margin: { l: 0, r: 0, t: 40, b: 0 },
        }, { responsive: true });
    }
}

