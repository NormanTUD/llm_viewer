/**
 * main.js — Core application logic
 * Handles: tab switching, model loading, text analysis, tokenization display
 */

let APP_STATE = {
    model: 'gpt2',
    text: '',
    tokens: [],
    tokenIds: [],
    traceKey: null,
    nLayers: 0,
    nHeads: 0,
    dModel: 0,
};

// ─── Tab Switching ────────────────────────────────────────

document.querySelectorAll('.tab-btn').forEach(btn => {
    btn.addEventListener('click', () => {
        document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
        document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
        btn.classList.add('active');
        document.getElementById(btn.dataset.tab).classList.add('active');
    });
});

// ─── Model Selection ──────────────────────────────────────

const modelSelect = document.getElementById('model-select');
const modelInfo = document.getElementById('model-info');

async function loadModelConfig() {
    const model = modelSelect.value;
    APP_STATE.model = model;
    modelInfo.textContent = 'Loading...';
    try {
        const resp = await fetch(`/api/model_config/${model}`);
        const config = await resp.json();
        APP_STATE.nLayers = config.n_layers;
        APP_STATE.nHeads = config.n_heads;
        APP_STATE.dModel = config.d_model;
        modelInfo.textContent = `${config.n_layers}L / ${config.n_heads}H / d=${config.d_model}`;
        updateLayerControls();
    } catch (e) {
        modelInfo.textContent = 'Error loading model';
    }
}

modelSelect.addEventListener('change', loadModelConfig);

function updateLayerControls() {
    // Update layer slider
    const slider = document.getElementById('layer-slider');
    if (slider) {
        slider.max = APP_STATE.nLayers;
        slider.value = 0;
    }

    // Update layer selects
    ['attn-layer-select', 'slicer-layer-select'].forEach(id => {
        const sel = document.getElementById(id);
        if (sel) {
            sel.innerHTML = '';
            for (let i = 0; i <= APP_STATE.nLayers; i++) {
                const opt = document.createElement('option');
                opt.value = i;
                opt.textContent = i === 0 ? 'Input (layer 0)' : `Layer ${i}`;
                sel.appendChild(opt);
            }
        }
    });

    // Update head select
    const headSel = document.getElementById('attn-head-select');
    if (headSel) {
        headSel.innerHTML = '<option value="avg">Average (all heads)</option>';
        for (let i = 0; i < APP_STATE.nHeads; i++) {
            const opt = document.createElement('option');
            opt.value = i;
            opt.textContent = `Head ${i}`;
            headSel.appendChild(opt);
        }
    }
}

// ─── Analyze Button ───────────────────────────────────────

const btnAnalyze = document.getElementById('btn-analyze');
const textInput = document.getElementById('text-input');
const tokenDisplay = document.getElementById('token-display');

btnAnalyze.addEventListener('click', runAnalysis);
textInput.addEventListener('keydown', e => { if (e.key === 'Enter') runAnalysis(); });

async function runAnalysis() {
    const text = textInput.value.trim();
    if (!text) return;

    APP_STATE.text = text;
    btnAnalyze.innerHTML = '🔍 Analyzing... <span class="loading"></span>';
    btnAnalyze.disabled = true;

    try {
        // Step 1: Tokenize
        const tokResp = await fetch('/api/tokenize', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({ text, model: APP_STATE.model }),
        });
        const tokData = await tokResp.json();
        APP_STATE.tokens = tokData.tokens;
        APP_STATE.tokenIds = tokData.token_ids;
        displayTokens(tokData);

        // Step 2: Full trace
        const traceResp = await fetch('/api/trace', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({ text, model: APP_STATE.model }),
        });
        const traceData = await traceResp.json();
        APP_STATE.traceKey = traceData.trace_key;
        APP_STATE.nLayers = traceData.n_layers;
        APP_STATE.nHeads = traceData.n_heads;
        APP_STATE.dModel = traceData.d_model;

        updateLayerControls();
        populateTokenSelects();

        // Auto-show embeddings
        showEmbeddings();

    } catch (e) {
        console.error('Analysis failed:', e);
        alert('Analysis failed. Make sure the model is downloaded.');
    }

    btnAnalyze.innerHTML = '🔍 Analyze';
    btnAnalyze.disabled = false;
}

function displayTokens(tokData) {
    tokenDisplay.innerHTML = '';
    tokData.tokens.forEach((tok, i) => {
        const chip = document.createElement('span');
        chip.className = 'token-chip';
        chip.innerHTML = `${escapeHtml(tok)}<span class="token-id">${tokData.token_ids[i]}</span>`;
        chip.title = `Token ${i}: "${tok}" (ID: ${tokData.token_ids[i]})`;
        chip.addEventListener('click', () => selectToken(i));
        tokenDisplay.appendChild(chip);
    });
}

function populateTokenSelects() {
    const sel = document.getElementById('traj-token-select');
    if (sel) {
        sel.innerHTML = '';
        APP_STATE.tokens.forEach((tok, i) => {
            const opt = document.createElement('option');
            opt.value = i;
            opt.textContent = `[${i}] ${tok}`;
            sel.appendChild(opt);
        });
    }
}

function selectToken(idx) {
    console.log('Selected token:', idx, APP_STATE.tokens[idx]);
    // Highlight
    document.querySelectorAll('.token-chip').forEach((chip, i) => {
        chip.style.borderColor = i === idx ? '#58a6ff' : '#374151';
        chip.style.background = i === idx ? '#1e3a5f' : '#1f2937';
    });
}

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

// ─── Helper: API call ─────────────────────────────────────

async function apiPost(url, data) {
    const resp = await fetch(url, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(data),
    });
    return resp.json();
}

// ─── Initialize ───────────────────────────────────────────

loadModelConfig();
