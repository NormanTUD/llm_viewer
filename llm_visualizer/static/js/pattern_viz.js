/**
 * pattern_viz.js v3 — Continuous curve fitting visualization.
 *
 * Key visual changes from v2:
 *  - Shows the ALIGNED CURVE as a smooth line (not a ghost template)
 *  - Draws projection lines from each inlier token to its nearest
 *    point on the curve ("whiskers"), making it visually clear
 *    that tokens are ON the curve
 *  - Color encodes distance-to-curve (blue=close, red=far)
 */

let _searchAbort = null;
let _searchRunning = false;

// ─── Bindings ────────────────────────────────────────────

document.getElementById('btn-preview-pattern')?.addEventListener('click', previewPattern);
document.getElementById('btn-preview-filter')?.addEventListener('click', previewFilter);
document.getElementById('btn-search-pattern')?.addEventListener('click', searchPatternLive);
document.getElementById('btn-scan-all-patterns')?.addEventListener('click', scanAllPatterns);

document.getElementById('pattern-filter')?.addEventListener('change', () => {
    document.getElementById('pattern-regex').style.display =
        document.getElementById('pattern-filter').value === 'regex' ? 'inline-block' : 'none';
});

document.getElementById('pattern-strictness')?.addEventListener('input', (e) => {
    document.getElementById('strictness-label').textContent = e.target.value;
});


// ─── Preview ─────────────────────────────────────────────

async function previewPattern() {
    const pattern = document.getElementById('pattern-select').value;
    const n = parseInt(document.getElementById('pattern-n-points').value) || 40;
    const data = await apiPost('/api/patterns/generate', { pattern, n_points: n * 4 });
    const pts = data.points;

    Plotly.newPlot('pattern-main-plot', [{
        x: pts.map(p => p[0]), y: pts.map(p => p[1]), z: pts.map(p => p[2]),
        mode: 'markers+lines', type: 'scatter3d',
        marker: { size: 3, color: pts.map((_, i) => i), colorscale: 'Turbo', opacity: 0.8 },
        line: { width: 3, color: '#58a6ff' },
        hovertext: pts.map((p, i) => `Point ${i}`),
        hoverinfo: 'text', name: pattern,
    }], darkLayout(`Pattern: ${pattern}`), { responsive: true });
}

async function previewFilter() {
    const data = await apiPost('/api/patterns/filter_preview', {
        model: APP_STATE.model,
        filter: document.getElementById('pattern-filter').value,
        regex: document.getElementById('pattern-regex').value || null,
        max_show: 200,
    });
    document.getElementById('filter-preview').innerHTML =
        `<b>${data.count} tokens</b> ` +
        `<span style="color:#a5f3fc">${data.tokens.map(t => `"${escapeHtml(t.text)}"`).join(', ')}</span>`;
}


// ─── LIVE SEARCH ─────────────────────────────────────────

async function searchPatternLive() {
    if (_searchRunning) {
        if (_searchAbort) _searchAbort.abort();
        _searchRunning = false;
        updateSearchButton(false);
        return;
    }

    const params = {
        model: APP_STATE.model,
        pattern: document.getElementById('pattern-select').value,
        n_pattern_points: parseInt(document.getElementById('pattern-n-points').value) || 40,
        filter: document.getElementById('pattern-filter').value,
        regex: document.getElementById('pattern-regex').value || null,
        sample_size: parseInt(document.getElementById('pattern-sample-size').value) || 10000,
        projection: 'pca',
        search_subspaces: document.getElementById('pattern-search-subspaces')?.checked || false,
        strictness: parseFloat(document.getElementById('pattern-strictness').value) || 0.5,
        trace_key: APP_STATE.traceKey || null,
    };
    if (params.filter === 'regex') params.regex = params.regex;
    else delete params.regex;

    _searchRunning = true;
    updateSearchButton(true);

    const infoPanel = document.getElementById('pattern-info');
    infoPanel.innerHTML = `
        <h3>🔎 Searching...</h3>
        <p>Pattern: <b>${params.pattern}</b> | Filter: <b>${params.filter}</b> |
           Strictness: <b>${params.strictness}</b></p>
        <p><span class="loading"></span> Watch tokens snap onto the curve in real time!</p>
    `;
    document.getElementById('pattern-ranking').innerHTML = '';

    const controller = new AbortController();
    _searchAbort = controller;
    let updateCount = 0;

    try {
        const response = await fetch('/api/patterns/search_stream', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(params),
            signal: controller.signal,
        });

        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let buffer = '';

        while (true) {
            const { done, value } = await reader.read();
            if (done) break;
            buffer += decoder.decode(value, { stream: true });

            const messages = buffer.split('\n\n');
            buffer = messages.pop();

            for (const msg of messages) {
                const line = msg.trim();
                if (!line.startsWith('data: ')) continue;
                let data;
                try { data = JSON.parse(line.substring(6)); } catch { continue; }

                if (data.status === 'progress') {
                    handleProgress(data);
                } else if (data.status === 'best_update' || data.status === 'done') {
                    updateCount++;
                    renderMatch(data, infoPanel, updateCount, data.status === 'done');
                } else if (data.status === 'error') {
                    infoPanel.innerHTML = `<h3>❌ Error</h3><p>${data.message}</p>`;
                }
            }
        }
    } catch (e) {
        if (e.name !== 'AbortError')
            infoPanel.innerHTML += `<p style="color:#da3633;">Error: ${e.message}</p>`;
    }

    _searchRunning = false;
    updateSearchButton(false);
}

function updateSearchButton(running) {
    const btn = document.getElementById('btn-search-pattern');
    btn.innerHTML = running ? '⏹ Stop' : '🔎 Find Pattern in Embeddings';
    btn.style.background = running ? '#da3633' : '#238636';
}


// ─── Progress ────────────────────────────────────────────

function handleProgress(data) {
    const colors = { loading:'#58a6ff', coarse:'#f97316', fine:'#a855f7',
                     refine:'#22c55e', subspace:'#ec4899' };
    const c = colors[data.phase] || '#8b949e';
    const pct = data.total ? Math.round((data.trial || 0) / data.total * 100) : 0;

    document.getElementById('pattern-ranking').innerHTML = `
        <div style="display:flex; align-items:center; gap:12px; padding:8px;">
            <span class="loading"></span>
            <span style="color:${c}; font-weight:bold; text-transform:uppercase;">${data.phase || ''}</span>
            <span>${data.message || ''}</span>
            <div style="flex:1; background:#21262d; border-radius:4px; height:8px; min-width:80px;">
                <div style="background:${c}; width:${pct}%; height:100%; border-radius:4px;
                    transition:width 0.2s;"></div>
            </div>
        </div>`;
}


// ─── RENDER MATCH (the key visualization) ────────────────

function renderMatch(data, infoPanel, updateCount, isFinal) {
    const traces = [];

    // 1. Background cloud
    const ctx = data.context;
    if (ctx?.positions_3d?.length) {
        traces.push({
            x: ctx.positions_3d.map(p => p[0]),
            y: ctx.positions_3d.map(p => p[1]),
            z: ctx.positions_3d.map(p => p[2]),
            mode: 'markers', type: 'scatter3d',
            marker: { size: 1.5, color: '#2d333b', opacity: 0.3 },
            hovertext: ctx.tokens.map((t, i) =>
                `<b>"${t}"</b><br>ID: ${ctx.token_ids[i]}`),
            hoverinfo: 'text',
            name: `Background (${ctx.tokens.length})`,
        });
    }

    // 2. Aligned curve (the continuous fitted curve)
    if (data.aligned_curve_3d?.length > 1) {
        traces.push({
            x: data.aligned_curve_3d.map(p => p[0]),
            y: data.aligned_curve_3d.map(p => p[1]),
            z: data.aligned_curve_3d.map(p => p[2]),
            mode: 'lines', type: 'scatter3d',
            line: { width: 5, color: 'rgba(88,166,255,0.4)' },
            hoverinfo: 'skip',
            name: 'Fitted curve',
        });
    }

    // 3. Projection whiskers: lines from each inlier to its projection on curve
    //    This is the key visual that shows tokens are ON the curve
    if (data.matched_positions_3d?.length && data.projection_points_3d?.length) {
        const whiskerX = [], whiskerY = [], whiskerZ = [];
        for (let i = 0; i < data.matched_positions_3d.length; i++) {
            const tok = data.matched_positions_3d[i];
            const proj = data.projection_points_3d[i];
            if (tok && proj) {
                whiskerX.push(tok[0], proj[0], null);
                whiskerY.push(tok[1], proj[1], null);
                whiskerZ.push(tok[2], proj[2], null);
            }
        }
        if (whiskerX.length > 0) {
            traces.push({
                x: whiskerX, y: whiskerY, z: whiskerZ,
                mode: 'lines', type: 'scatter3d',
                line: { width: 1.5, color: 'rgba(251,191,36,0.3)' },
                hoverinfo: 'skip',
                name: 'Distance to curve',
                showlegend: false,
            });
        }
    }

    // 4. Matched tokens — colored by distance to curve
    if (data.matched_positions_3d?.length) {
        const dists = data.inlier_distances || [];
        const maxDist = Math.max(...dists, 0.001);

        traces.push({
            x: data.matched_positions_3d.map(p => p[0]),
            y: data.matched_positions_3d.map(p => p[1]),
            z: data.matched_positions_3d.map(p => p[2]),
            mode: 'markers+text', type: 'scatter3d',
            text: data.matched_tokens.map(t =>
                t.trim().length <= 12 ? t.trim() : ''),
            textposition: 'top center',
            textfont: { size: 9, color: '#fbbf24' },
            hovertext: data.matched_tokens.map((t, i) =>
                `<b>"${t}"</b><br>` +
                `ID: ${data.matched_token_ids[i]}<br>` +
                `#${i + 1} / ${data.n_matched}<br>` +
                `Dist to curve: ${(dists[i] || 0).toFixed(4)}<br>` +
                `Curve param: ${(data.curve_parameters?.[i] || 0).toFixed(2)}<br>` +
                `<span style="color:#22c55e">● ON CURVE</span>`),
            hoverinfo: 'text',
            marker: {
                size: 6,
                color: dists.map(d => 1 - d / maxDist), // 1=close(blue), 0=far(red)
                colorscale: [[0, '#ef4444'], [0.5, '#fbbf24'], [1, '#22c55e']],
                cmin: 0, cmax: 1,
                opacity: 0.95,
                line: { width: 0.5, color: '#fff' },
                colorbar: { title: 'Fit', tickvals: [0, 1], ticktext: ['Far', 'Close'],
                            len: 0.4, x: 1.02 },
            },
            name: `Inliers (${data.n_matched})`,
        });

        // 5. Path connecting matched tokens in curve order
        traces.push({
            x: data.matched_positions_3d.map(p => p[0]),
            y: data.matched_positions_3d.map(p => p[1]),
            z: data.matched_positions_3d.map(p => p[2]),
            mode: 'lines', type: 'scatter3d',
            line: { width: 2, color: 'rgba(249,115,22,0.5)' },
            hoverinfo: 'skip',
            name: 'Token path',
            showlegend: false,
        });
    }

    const sim = (data.match_similarity || 0).toFixed(3);
    const icon = isFinal ? '✅' : '🔄';
    const title = `${icon} "${data.pattern_name}" in ${data.filter_type} — ` +
                  `${data.n_matched} tokens on curve (sim: ${sim})`;

    Plotly.react('pattern-main-plot', traces, darkLayout(title), { responsive: true });

    // ── Info panel ────────────────────────────────────
    const simColor = data.match_similarity > 0.7 ? '#22c55e' :
                     data.match_similarity > 0.4 ? '#d29922' : '#da3633';
    const avgDist = data.inlier_distances?.length ?
        (data.inlier_distances.reduce((a, b) => a + b, 0) / data.inlier_distances.length).toFixed(4) : '?';

    infoPanel.innerHTML = `
        <h3>${isFinal ? '✅ Done!' : '🔄 Live Update #' + updateCount}</h3>
        <div style="display:grid; grid-template-columns:auto 1fr; gap:3px 14px; margin:8px 0; font-size:0.88rem;">
            <span style="color:#8b949e;">Pattern:</span> <b>${data.pattern_name}</b>
            <span style="color:#8b949e;">Filter:</span> <b>${data.filter_type}</b>
            <span style="color:#8b949e;">Strictness:</span> <b>${data.strictness}</b>
            <span style="color:#8b949e;">Searched:</span> <b>${(data.n_searched||0).toLocaleString()} tokens</b>
            <span style="color:#8b949e;">On curve:</span> <b style="color:${simColor};font-size:1.1em;">${data.n_matched}</b>
            <span style="color:#8b949e;">Avg dist to curve:</span> <b>${avgDist}</b>
            <span style="color:#8b949e;">Threshold:</span> <b>${(data.inlier_threshold || 0).toFixed(4)}</b>
            <span style="color:#8b949e;">Similarity:</span>
                <b style="color:${simColor};font-size:1.1em;">${sim}</b>
            <span style="color:#8b949e;">Strategy:</span> <b>${data.search_info?.strategy || '?'}</b>
            <span style="color:#8b949e;">Elapsed:</span> <b>${data.elapsed || '?'}s</b>
        </div>

        <hr style="border-color:#30363d; margin:12px 0">
        <h4>Tokens on the curve (ordered by curve parameter):</h4>
        <div style="max-height:280px; overflow-y:auto;">
            <table style="width:100%; border-collapse:collapse;">
                <tr style="border-bottom:1px solid #30363d; position:sticky; top:0; background:#161b22;">
                    <th style="padding:3px; text-align:left;">#</th>
                    <th style="padding:3px; text-align:left;">Token</th>
                    <th style="padding:3px; text-align:left;">ID</th>
                    <th style="padding:3px; text-align:left;">Dist</th>
                    <th style="padding:3px; text-align:left;">Param</th>
                </tr>
                ${(data.matched_tokens || []).map((t, i) => {
                    const d = (data.inlier_distances || [])[i] || 0;
                    const p = (data.curve_parameters || [])[i] || 0;
                    const dColor = d < (data.inlier_threshold || 1) * 0.3 ? '#22c55e' :
                                   d < (data.inlier_threshold || 1) * 0.7 ? '#fbbf24' : '#ef4444';
                    return `
                    <tr style="border-bottom:1px solid #21262d;"
                        onmouseover="this.style.background='#1e3a5f'"
                        onmouseout="this.style.background='transparent'">
                        <td style="padding:3px; color:#6b7280;">${i + 1}</td>
                        <td style="padding:3px; color:#fbbf24; font-weight:bold;">"${escapeHtml(t)}"</td>
                        <td style="padding:3px; color:#6b7280;">${data.matched_token_ids[i]}</td>
                        <td style="padding:3px; color:${dColor};">${d.toFixed(4)}</td>
                        <td style="padding:3px; color:#8b949e;">${p.toFixed(2)}</td>
                    </tr>`;
                }).join('')}
            </table>
        </div>

        ${isFinal ? `
        <hr style="border-color:#30363d; margin:12px 0">
        <div style="display:flex; gap:8px; flex-wrap:wrap;">
            <button class="btn-secondary" onclick="rerunWithStrictness(0.3)">Re-run loose (0.3)</button>
            <button class="btn-secondary" onclick="rerunWithStrictness(0.5)">Re-run balanced (0.5)</button>
            <button class="btn-secondary" onclick="rerunWithStrictness(0.8)">Re-run strict (0.8)</button>
        </div>` : ''}
    `;
}


// ─── Quick re-run with different strictness ──────────────

function rerunWithStrictness(val) {
    document.getElementById('pattern-strictness').value = val;
    document.getElementById('strictness-label').textContent = val;
    searchPatternLive();
}


// ─── Scan all patterns ───────────────────────────────────

async function scanAllPatterns() {
    const filter = document.getElementById('pattern-filter').value;
    const sampleSize = parseInt(document.getElementById('pattern-sample-size').value) || 5000;
    const ranking = document.getElementById('pattern-ranking');

    ranking.innerHTML = `<p>📊 Scanning all patterns against <b>${filter}</b>...
        <span class="loading"></span></p>`;

    const data = await apiPost('/api/patterns/scan_all', {
        model: APP_STATE.model, filter, sample_size: sampleSize, projection: 'pca',
    });

    if (!data.results?.length) {
        ranking.innerHTML = '<p>No results.</p>';
        return;
    }

    ranking.innerHTML = `
        <h3>All Patterns — "${filter}" tokens</h3>
        <p>Which geometric shape do these token embeddings best lie on?</p>
        <table style="width:100%; border-collapse:collapse; margin-top:8px;">
            <tr style="border-bottom:1px solid #30363d;">
                <th style="padding:6px; text-align:left;">#</th>
                <th style="padding:6px; text-align:left;">Pattern</th>
                <th style="padding:6px; text-align:left;">Sim</th>
                <th style="padding:6px; text-align:left;">On curve</th>
                <th style="padding:6px; text-align:left;">Sample tokens</th>
                <th style="padding:6px; text-align:left;"></th>
            </tr>
            ${data.results.map((r, i) => {
                const s = r.similarity || 0;
                const c = s > 0.6 ? '#22c55e' : s > 0.35 ? '#d29922' : '#da3633';
                return `
                <tr style="border-bottom:1px solid #21262d; cursor:pointer;"
                    onclick="quickSearch('${r.pattern}')"
                    onmouseover="this.style.background='#21262d'"
                    onmouseout="this.style.background='transparent'">
                    <td style="padding:6px;">${i + 1}</td>
                    <td style="padding:6px; font-weight:bold; text-transform:capitalize;">${r.pattern}</td>
                    <td style="padding:6px; color:${c}; font-weight:bold;">${s.toFixed(4)}</td>
                    <td style="padding:6px;">${r.n_matched || '?'}</td>
                    <td style="padding:6px; color:#8b949e; font-size:0.8rem; max-width:250px; overflow:hidden; text-overflow:ellipsis; white-space:nowrap;">
                        ${(r.sample_tokens || []).slice(0, 6).map(t => `"${t.trim()}"`).join(', ')}
                    </td>
                    <td style="padding:6px;">
                        <div style="height:12px; width:${(s * 200).toFixed(0)}px;
                            border-radius:3px; background:linear-gradient(90deg,#1e3a5f,${c});"></div>
                    </td>
                </tr>`;
            }).join('')}
        </table>
        <p class="hint">Click a row to run a full live search for that pattern.</p>
    `;

    // Bar chart
    Plotly.newPlot('pattern-bar-chart', [{
        x: data.results.map(r => r.pattern),
        y: data.results.map(r => r.similarity || 0),
        type: 'bar',
        marker: {
            color: data.results.map(r => r.similarity || 0),
            colorscale: [[0, '#da3633'], [0.5, '#d29922'], [1, '#22c55e']],
        },
        hovertext: data.results.map(r =>
            `${r.pattern}: sim=${(r.similarity||0).toFixed(4)}, ${r.n_matched||'?'} tokens`),
        hoverinfo: 'text',
    }], {
        paper_bgcolor: '#161b22', plot_bgcolor: '#0d1117', font: { color: '#c9d1d9' },
        title: `Pattern fit — "${filter}" tokens`,
        margin: { l: 50, r: 20, t: 50, b: 80 },
        xaxis: { color: '#8b949e' }, yaxis: { color: '#8b949e', title: 'Similarity' },
    }, { responsive: true });
}

function quickSearch(patternName) {
    document.getElementById('pattern-select').value = patternName;
    searchPatternLive();
}


// ─── Dark layout helper ──────────────────────────────────

function darkLayout(title) {
    return {
        paper_bgcolor: '#161b22', plot_bgcolor: '#0d1117',
        font: { color: '#c9d1d9' },
        scene: {
            bgcolor: '#0d1117',
            xaxis: { gridcolor: '#21262d', color: '#6b7280', showbackground: false },
            yaxis: { gridcolor: '#21262d', color: '#6b7280', showbackground: false },
            zaxis: { gridcolor: '#21262d', color: '#6b7280', showbackground: false },
        },
        title: { text: title, font: { size: 13 } },
        margin: { l: 0, r: 0, t: 50, b: 0 },
        legend: {
            bgcolor: 'rgba(22,27,34,0.85)', bordercolor: '#30363d',
            font: { color: '#c9d1d9', size: 10 },
        },
    };
}


// ─── Init ────────────────────────────────────────────────

setTimeout(() => { previewPattern(); }, 400);
