/**
 * pattern_viz.js — Real embedding space pattern discovery.
 *
 * Core feature: search the actual vocabulary embedding space for
 * subsets of tokens whose positions form geometric patterns
 * (spirals, helices, etc.) on manifolds in the high-D space.
 *
 * Shows:
 *  - Background cloud of all/sampled tokens (grey, mouseover shows token)
 *  - Highlighted matched tokens (colored, connected in order)
 *  - Pattern template overlay (wireframe ghost)
 */

// ─── Button bindings ─────────────────────────────────────

document.getElementById('btn-preview-pattern')?.addEventListener('click', previewPattern);
document.getElementById('btn-preview-filter')?.addEventListener('click', previewFilter);
document.getElementById('btn-search-pattern')?.addEventListener('click', searchPattern);
document.getElementById('btn-scan-all-patterns')?.addEventListener('click', scanAllPatterns);

// Show/hide regex input based on filter selection
document.getElementById('pattern-filter')?.addEventListener('change', () => {
    const regexInput = document.getElementById('pattern-regex');
    regexInput.style.display =
        document.getElementById('pattern-filter').value === 'regex' ? 'inline-block' : 'none';
});


// ─── Pattern Preview ─────────────────────────────────────

async function previewPattern() {
    const pattern = document.getElementById('pattern-select').value;
    const nPoints = parseInt(document.getElementById('pattern-n-points').value) || 50;

    const data = await apiPost('/api/patterns/generate', {
        pattern: pattern,
        n_points: nPoints * 3,   // denser preview
    });

    const pts = data.points;
    const trace = {
        x: pts.map(p => p[0]),
        y: pts.map(p => p[1]),
        z: pts.map(p => p[2]),
        mode: 'markers+lines',
        type: 'scatter3d',
        marker: {
            size: 3,
            color: pts.map((_, i) => i),
            colorscale: 'Turbo',
            opacity: 0.8,
        },
        line: { width: 2, color: '#58a6ff' },
        hovertext: pts.map((p, i) =>
            `Point ${i}<br>x: ${p[0].toFixed(3)}<br>y: ${p[1].toFixed(3)}<br>z: ${p[2].toFixed(3)}`
        ),
        hoverinfo: 'text',
        name: `${pattern} template`,
    };

    Plotly.newPlot('pattern-main-plot', [trace], darkLayout(`Pattern Preview: ${pattern}`), { responsive: true });
}


// ─── Filter Preview ──────────────────────────────────────

async function previewFilter() {
    const model = APP_STATE.model;
    const filter = document.getElementById('pattern-filter').value;
    const regex = document.getElementById('pattern-regex').value || null;

    const data = await apiPost('/api/patterns/filter_preview', {
        model: model,
        filter: filter,
        regex: filter === 'regex' ? regex : null,
        max_show: 200,
    });

    const preview = document.getElementById('filter-preview');
    preview.innerHTML = `
        <b>${data.count} tokens match filter "${filter}"</b>
        ${data.count > 200 ? ` (showing first 200)` : ''}<br>
        <span style="color:#a5f3fc;">
            ${data.tokens.map(t => `"${escapeHtml(t.text)}"`).join(', ')}
        </span>
    `;
}


// ─── MAIN: Search for pattern in real embeddings ─────────

async function searchPattern() {
    const model = APP_STATE.model;
    const pattern = document.getElementById('pattern-select').value;
    const filter = document.getElementById('pattern-filter').value;
    const regex = document.getElementById('pattern-regex').value || null;
    const nPoints = parseInt(document.getElementById('pattern-n-points').value) || 40;
    const sampleSize = parseInt(document.getElementById('pattern-sample-size').value) || 10000;
    const method = document.getElementById('pattern-search-method').value;
    const searchSub = document.getElementById('pattern-search-subspaces').checked;

    const infoPanel = document.getElementById('pattern-info');
    infoPanel.innerHTML = `<h3>🔎 Searching...</h3>
        <p>Finding real tokens that form a <b>${pattern}</b> in embedding space...</p>
        <p>Filter: <b>${filter}</b> | Sample: <b>${sampleSize}</b> tokens</p>
        <p><span class="loading"></span> This may take a moment...</p>`;

    const data = await apiPost('/api/patterns/search', {
        model: model,
        pattern: pattern,
        n_pattern_points: nPoints,
        filter: filter,
        regex: filter === 'regex' ? regex : null,
        sample_size: sampleSize,
        search_method: method,
        projection: 'pca',
        search_subspaces: searchSub,
        trace_key: APP_STATE.traceKey || null,
    });

    if (data.error) {
        infoPanel.innerHTML = `<h3>❌ Error</h3><p>${data.error}</p>`;
        return;
    }

    // ── Build 3D visualization ────────────────────────

    const traces = [];

    // 1. Background context cloud (grey, small, mouseover reveals token)
    if (data.context_positions_3d && data.context_positions_3d.length > 0) {
        traces.push({
            x: data.context_positions_3d.map(p => p[0]),
            y: data.context_positions_3d.map(p => p[1]),
            z: data.context_positions_3d.map(p => p[2]),
            mode: 'markers',
            type: 'scatter3d',
            marker: { size: 2, color: '#2d333b', opacity: 0.4 },
            hovertext: data.context_tokens.map((t, i) =>
                `<b>"${t}"</b><br>ID: ${data.context_token_ids[i]}<br><i>(background)</i>`
            ),
            hoverinfo: 'text',
            name: `All tokens (${data.context_tokens.length} sampled)`,
        });
    }

    // 2. Pattern template overlay (ghost wireframe)
    if (data.pattern_overlay_3d && data.pattern_overlay_3d.length > 0) {
        traces.push({
            x: data.pattern_overlay_3d.map(p => p[0]),
            y: data.pattern_overlay_3d.map(p => p[1]),
            z: data.pattern_overlay_3d.map(p => p[2]),
            mode: 'lines',
            type: 'scatter3d',
            line: { width: 3, color: 'rgba(255,255,255,0.15)', dash: 'dot' },
            hoverinfo: 'skip',
            name: `${pattern} template (aligned)`,
        });
    }

    // 3. Matched tokens — the discovered pattern!
    if (data.matched_positions_3d && data.matched_positions_3d.length > 0) {
        // Line connecting matched tokens in order
        traces.push({
            x: data.matched_positions_3d.map(p => p[0]),
            y: data.matched_positions_3d.map(p => p[1]),
            z: data.matched_positions_3d.map(p => p[2]),
            mode: 'lines',
            type: 'scatter3d',
            line: { width: 4, color: '#f97316' },
            hoverinfo: 'skip',
            name: 'Matched path',
        });

        // Points with full hover info
        traces.push({
            x: data.matched_positions_3d.map(p => p[0]),
            y: data.matched_positions_3d.map(p => p[1]),
            z: data.matched_positions_3d.map(p => p[2]),
            mode: 'markers+text',
            type: 'scatter3d',
            text: data.matched_tokens.map(t => t.trim().length <= 8 ? t.trim() : ''),
            textposition: 'top center',
            textfont: { size: 9, color: '#fbbf24' },
            hovertext: data.matched_tokens.map((t, i) =>
                `<b>"${t}"</b><br>` +
                `ID: ${data.matched_token_ids[i]}<br>` +
                `Position in pattern: ${i + 1} / ${data.n_matched}<br>` +
                `<i style="color:#f97316;">★ MATCHED</i>`
            ),
            hoverinfo: 'text',
            marker: {
                size: 7,
                color: data.matched_tokens.map((_, i) => i),
                colorscale: 'Hot',
                opacity: 1.0,
                line: { width: 1, color: '#fff' },
                colorbar: { title: 'Order', len: 0.5 },
            },
            name: `Matched tokens (${data.n_matched})`,
        });
    }

    const title = `"${pattern}" pattern in ${filter} tokens — ` +
                  `${data.n_matched} matches (sim: ${data.match_similarity.toFixed(3)})`;

    Plotly.newPlot('pattern-main-plot', traces, darkLayout(title), { responsive: true });

    // ── Info panel ────────────────────────────────────

    infoPanel.innerHTML = `
        <h3>✅ Pattern Found!</h3>
        <p><b>Pattern:</b> ${pattern}</p>
        <p><b>Filter:</b> ${filter}</p>
        <p><b>Tokens searched:</b> ${data.n_searched.toLocaleString()}</p>
        <p><b>Tokens matched:</b> ${data.n_matched}</p>
        <p><b>Similarity score:</b> <span style="color:${
            data.match_similarity > 0.5 ? '#238636' :
            data.match_similarity > 0.3 ? '#d29922' : '#da3633'
        }; font-weight:bold;">${data.match_similarity.toFixed(4)}</span></p>
        <p><b>Search strategy:</b> ${data.search_info.strategy || 'N/A'}
           ${data.search_info.best_subspace ? ` (${data.search_info.best_subspace})` : ''}</p>

        <hr style="border-color:#30363d; margin:12px 0">
        <h4>Matched Tokens (in pattern order):</h4>
        <div style="max-height:300px; overflow-y:auto;">
            <table style="width:100%; border-collapse:collapse;">
                <tr style="border-bottom:1px solid #30363d; position:sticky; top:0; background:#161b22;">
                    <th style="padding:4px; text-align:left;">#</th>
                    <th style="padding:4px; text-align:left;">Token</th>
                    <th style="padding:4px; text-align:left;">ID</th>
                </tr>
                ${data.matched_tokens.map((t, i) => `
                    <tr style="border-bottom:1px solid #21262d;"
                        onmouseover="this.style.background='#1e3a5f'"
                        onmouseout="this.style.background='transparent'">
                        <td style="padding:4px; color:#6b7280;">${i + 1}</td>
                        <td style="padding:4px; color:#fbbf24; font-weight:bold;">
                            "${escapeHtml(t)}"
                        </td>
                        <td style="padding:4px; color:#6b7280;">${data.matched_token_ids[i]}</td>
                    </tr>
                `).join('')}
            </table>
        </div>
    `;
}


// ─── Scan all patterns against a filter ──────────────────

async function scanAllPatterns() {
    const model = APP_STATE.model;
    const filter = document.getElementById('pattern-filter').value;
    const sampleSize = parseInt(document.getElementById('pattern-sample-size').value) || 5000;

    const ranking = document.getElementById('pattern-ranking');
    ranking.innerHTML = `<p>📊 Scanning all patterns against <b>${filter}</b> tokens...
        <span class="loading"></span></p>`;

    const data = await apiPost('/api/patterns/scan_all', {
        model: model,
        filter: filter,
        sample_size: sampleSize,
        projection: 'pca',
    });

    if (!data.results || data.results.length === 0) {
        ranking.innerHTML = '<p>No results.</p>';
        return;
    }

    ranking.innerHTML = `
        <h3>All Patterns Ranked — "${filter}" tokens</h3>
        <p>Which geometric shape do these tokens' embeddings most resemble?</p>
        <table style="width:100%; border-collapse:collapse; margin-top:8px;">
            <tr style="border-bottom:1px solid #30363d;">
                <th style="text-align:left; padding:6px;">Rank</th>
                <th style="text-align:left; padding:6px;">Pattern</th>
                <th style="text-align:left; padding:6px;">Similarity</th>
                <th style="text-align:left; padding:6px;">Matched</th>
                <th style="text-align:left; padding:6px;">Sample tokens</th>
                <th style="text-align:left; padding:6px;">Bar</th>
            </tr>
            ${data.results.map((r, i) => `
                <tr style="border-bottom:1px solid #21262d; cursor:pointer;"
                    onclick="quickSearch('${r.pattern}')"
                    onmouseover="this.style.background='#21262d'"
                    onmouseout="this.style.background='transparent'">
                    <td style="padding:6px;">${i + 1}</td>
                    <td style="padding:6px; text-transform:capitalize; font-weight:bold;">
                        ${r.pattern}
                    </td>
                    <td style="padding:6px; color:${
                        r.similarity > 0.5 ? '#238636' :
                        r.similarity > 0.3 ? '#d29922' : '#da3633'
                    };">${(r.similarity || 0).toFixed(4)}</td>
                    <td style="padding:6px;">${r.n_matched || '?'}</td>
                    <td style="padding:6px; color:#8b949e; font-size:0.8rem;">
                        ${(r.sample_tokens || []).slice(0, 5).map(t => `"${t.trim()}"`).join(', ')}
                    </td>
                    <td style="padding:6px;">
                        <div style="height:14px; width:${((r.similarity || 0) * 250).toFixed(0)}px;
                            border-radius:3px; background: linear-gradient(90deg, #1e3a5f, ${
                                r.similarity > 0.5 ? '#238636' : r.similarity > 0.3 ? '#d29922' : '#da3633'
                            });"></div>
                    </td>
                </tr>
            `).join('')}
        </table>
        <p class="hint">Click a row to run a full search for that pattern.</p>
    `;

    // Bar chart
    const barTrace = {
        x: data.results.map(r => r.pattern),
        y: data.results.map(r => r.similarity || 0),
        type: 'bar',
        marker: {
            color: data.results.map(r => r.similarity || 0),
            colorscale: [[0, '#da3633'], [0.5, '#d29922'], [1, '#238636']],
        },
        hovertext: data.results.map(r =>
            `${r.pattern}: ${(r.similarity || 0).toFixed(4)}<br>` +
            `Matched: ${r.n_matched || '?'} tokens<br>` +
            `Sample: ${(r.sample_tokens || []).slice(0, 3).join(', ')}`
        ),
        hoverinfo: 'text',
    };

    Plotly.newPlot('pattern-bar-chart', [barTrace], {
        paper_bgcolor: '#161b22',
        plot_bgcolor: '#0d1117',
        font: { color: '#c9d1d9' },
        title: `Pattern similarity for "${filter}" token embeddings`,
        xaxis: { title: 'Pattern', color: '#8b949e' },
        yaxis: { title: 'Similarity', color: '#8b949e' },
        margin: { l: 50, r: 20, t: 50, b: 80 },
    }, { responsive: true });
}


// ─── Quick search (from scan results table click) ────────

function quickSearch(patternName) {
    document.getElementById('pattern-select').value = patternName;
    searchPattern();
}


// ─── Dark layout helper ──────────────────────────────────

function darkLayout(title) {
    return {
        paper_bgcolor: '#161b22',
        plot_bgcolor: '#0d1117',
        font: { color: '#c9d1d9' },
        scene: {
            bgcolor: '#0d1117',
            xaxis: { gridcolor: '#30363d', color: '#6b7280' },
            yaxis: { gridcolor: '#30363d', color: '#6b7280' },
            zaxis: { gridcolor: '#30363d', color: '#6b7280' },
        },
        title: { text: title, font: { size: 14 } },
        margin: { l: 0, r: 0, t: 50, b: 0 },
        legend: {
            bgcolor: 'rgba(22,27,34,0.8)',
            bordercolor: '#30363d',
            font: { color: '#c9d1d9', size: 11 },
        },
    };
}


// ─── Auto-preview on load ────────────────────────────────

setTimeout(() => { previewPattern(); }, 500);
