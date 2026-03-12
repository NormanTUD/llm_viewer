# model_wrapper.py — Multi-model support with automatic download
import torch
import torch.nn as nn
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from collections import defaultdict

# ────────────────────────────────────────────────────────────
# Registry of models that will be auto-downloaded on first use
# ────────────────────────────────────────────────────────────
MODEL_REGISTRY = {
    # GPT-2 family
    'gpt2':            {'hf': 'gpt2',               'family': 'gpt2', 'label': 'GPT-2 (124 M)'},
    'gpt2-medium':     {'hf': 'gpt2-medium',        'family': 'gpt2', 'label': 'GPT-2 Medium (355 M)'},
    'gpt2-large':      {'hf': 'gpt2-large',         'family': 'gpt2', 'label': 'GPT-2 Large (774 M)'},
    'gpt2-xl':         {'hf': 'gpt2-xl',            'family': 'gpt2', 'label': 'GPT-2 XL (1.5 B)'},
    # GPT-Neo
    'gpt-neo-125M':    {'hf': 'EleutherAI/gpt-neo-125M',  'family': 'gptneo',  'label': 'GPT-Neo 125 M'},
    'gpt-neo-1.3B':    {'hf': 'EleutherAI/gpt-neo-1.3B',  'family': 'gptneo',  'label': 'GPT-Neo 1.3 B'},
    'gpt-neo-2.7B':    {'hf': 'EleutherAI/gpt-neo-2.7B',  'family': 'gptneo',  'label': 'GPT-Neo 2.7 B'},
    # GPT-J
    'gpt-j-6B':        {'hf': 'EleutherAI/gpt-j-6B',      'family': 'gptj',    'label': 'GPT-J 6 B'},
    # Pythia (GPT-NeoX)
    'pythia-70m':      {'hf': 'EleutherAI/pythia-70m',     'family': 'neox',    'label': 'Pythia 70 M'},
    'pythia-160m':     {'hf': 'EleutherAI/pythia-160m',    'family': 'neox',    'label': 'Pythia 160 M'},
    'pythia-410m':     {'hf': 'EleutherAI/pythia-410m',    'family': 'neox',    'label': 'Pythia 410 M'},
    'pythia-1b':       {'hf': 'EleutherAI/pythia-1b',      'family': 'neox',    'label': 'Pythia 1 B'},
    'pythia-1.4b':     {'hf': 'EleutherAI/pythia-1.4b',    'family': 'neox',    'label': 'Pythia 1.4 B'},
    # OPT
    'opt-125m':        {'hf': 'facebook/opt-125m',         'family': 'opt',     'label': 'OPT 125 M'},
    'opt-350m':        {'hf': 'facebook/opt-350m',         'family': 'opt',     'label': 'OPT 350 M'},
    'opt-1.3b':        {'hf': 'facebook/opt-1.3b',         'family': 'opt',     'label': 'OPT 1.3 B'},
}


class UnifiedModelWrapper:
    """
    Architecture-agnostic wrapper that intercepts activations
    for GPT-2, GPT-Neo, GPT-J, Pythia, and OPT families.
    Models are downloaded automatically from HuggingFace Hub.
    """

    def __init__(self, model_key='gpt2'):
        info = MODEL_REGISTRY.get(model_key)
        if info is None:
            # Allow raw HuggingFace IDs as fallback
            hf_name = model_key
            family = 'auto'
        else:
            hf_name = info['hf']
            family = info['family']

        print(f"[*] Downloading / loading model: {hf_name}  (family={family})")
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.tokenizer = AutoTokenizer.from_pretrained(hf_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            hf_name, torch_dtype=torch.float32
        ).to(self.device)
        self.model.eval()

        self.model_key = model_key
        self.hf_name = hf_name
        self.family = family

        # ── Detect architecture specifics ──
        self._detect_arch()

        self._activations = {}
        self._hooks = []
        print(f"[✓] Loaded: {self.n_layers} layers, d_model={self.d_model}, "
              f"device={self.device}")

    # ── Architecture detection ──────────────────────────────

    def _detect_arch(self):
        cfg = self.model.config
        # n_layer / n_embd
        self.n_layers = getattr(cfg, 'n_layer', None) or \
                        getattr(cfg, 'num_hidden_layers', None) or \
                        getattr(cfg, 'num_layers', None)
        self.d_model = getattr(cfg, 'n_embd', None) or \
                       getattr(cfg, 'hidden_size', None)
        self.n_heads = getattr(cfg, 'n_head', None) or \
                       getattr(cfg, 'num_attention_heads', None)
        self.vocab_size = cfg.vocab_size

        # Determine how to reach transformer blocks
        if hasattr(self.model, 'transformer') and hasattr(self.model.transformer, 'h'):
            self._blocks = self.model.transformer.h          # GPT-2 / Neo / J
            self._block_type = 'gpt2'
        elif hasattr(self.model, 'gpt_neox') and hasattr(self.model.gpt_neox, 'layers'):
            self._blocks = self.model.gpt_neox.layers        # Pythia / NeoX
            self._block_type = 'neox'
        elif hasattr(self.model, 'model') and hasattr(self.model.model, 'decoder'):
            self._blocks = self.model.model.decoder.layers   # OPT
            self._block_type = 'opt'
        else:
            raise RuntimeError(f"Unsupported architecture: {type(self.model)}")

    def _get_mlp_module(self, block):
        if self._block_type in ('gpt2', 'neox'):
            return block.mlp
        elif self._block_type == 'opt':
            # OPT has fc1+fc2 without a wrapping .mlp, so hook fc2 (output)
            return block.fc2
        return block.mlp

    def _get_attn_module(self, block):
        if self._block_type == 'gpt2':
            return block.attn
        elif self._block_type == 'neox':
            return block.attention
        elif self._block_type == 'opt':
            return block.self_attn
        return block.attn

    # ── Public info ─────────────────────────────────────────

    @property
    def layers(self):
        return self._blocks

    def get_model_info(self):
        total_params = sum(p.numel() for p in self.model.parameters())
        return {
            'model_name': self.hf_name,
            'model_key': self.model_key,
            'family': self.family,
            'n_layers': self.n_layers,
            'd_model': self.d_model,
            'n_heads': self.n_heads,
            'vocab_size': self.vocab_size,
            'total_params': total_params,
            'total_params_human': f"{total_params/1e6:.1f}M",
            'device': str(self.device),
        }

    # ── Hook machinery ──────────────────────────────────────

    def _make_hook(self, layer_idx, component='mlp'):
        def hook_fn(module, inp, out):
            tensor = out[0] if isinstance(out, tuple) else out
            self._activations[f"layer_{layer_idx}_{component}"] = tensor.detach().cpu()
        return hook_fn

    def register_hooks(self, layer_indices=None, components=('mlp', 'attn')):
        self.remove_hooks()
        if layer_indices is None:
            layer_indices = range(self.n_layers)
        for idx in layer_indices:
            block = self._blocks[idx]
            if 'mlp' in components:
                h = self._get_mlp_module(block).register_forward_hook(
                    self._make_hook(idx, 'mlp'))
                self._hooks.append(h)
            if 'attn' in components:
                h = self._get_attn_module(block).register_forward_hook(
                    self._make_hook(idx, 'attn'))
                self._hooks.append(h)

    def remove_hooks(self):
        for h in self._hooks:
            h.remove()
        self._hooks.clear()
        self._activations.clear()

    # ── Forward with activations ────────────────────────────

    def forward_with_activations(self, text, layer_indices=None):
        self.register_hooks(layer_indices)
        tokens = self.tokenizer.encode(text, return_tensors='pt').to(self.device)
        token_strings = [self.tokenizer.decode([t]) for t in tokens[0]]

        with torch.no_grad():
            outputs = self.model(tokens, output_attentions=True)

        logits = outputs.logits
        attentions = [a.detach().cpu().numpy() for a in outputs.attentions]
        activations = dict(self._activations)

        last_logits = logits[0, -1, :]
        top_k = torch.topk(last_logits, 10)
        probs = torch.softmax(last_logits, dim=0)
        predictions = [
            {'token': self.tokenizer.decode([idx]), 'prob': float(probs[idx])}
            for idx in top_k.indices
        ]
        self.remove_hooks()

        return {
            'tokens': token_strings,
            'activations': activations,
            'attentions': attentions,
            'predictions': predictions,
            'logits_shape': list(logits.shape),
        }

    def get_activations(self, text, layer):
        """Return last-token MLP activation for *layer*."""
        result = self.forward_with_activations(text, layer_indices=[layer])
        key = f"layer_{layer}_mlp"
        act = result['activations'].get(key)
        if act is None:
            return None
        # act may be [1, seq, d] or [1, d] depending on architecture
        if act.dim() == 3:
            return act[0, -1, :]
        elif act.dim() == 2:
            return act[0, :]
        return act

    # ── All-layer activations for 3-D visualisation ─────────

    def get_all_layer_activations(self, text):
        """
        Returns dict  layer_idx → [d_model] tensor  (last token MLP).
        Used for the 3-D layer trajectory plot.
        """
        self.register_hooks(components=('mlp',))
        tokens = self.tokenizer.encode(text, return_tensors='pt').to(self.device)
        with torch.no_grad():
            self.model(tokens)
        result = {}
        for idx in range(self.n_layers):
            key = f"layer_{idx}_mlp"
            act = self._activations.get(key)
            if act is not None:
                if act.dim() == 3:
                    result[idx] = act[0, -1, :]
                elif act.dim() == 2:
                    result[idx] = act[0, :]
                else:
                    result[idx] = act
        self.remove_hooks()
        return result

    # ── Causal tracing ──────────────────────────────────────

    def causal_trace(self, prompt, subject_range=None):
        tokens = self.tokenizer.encode(prompt, return_tensors='pt').to(self.device)

        # Clean run
        self.register_hooks()
        with torch.no_grad():
            clean_out = self.model(tokens)
        clean_logits = clean_out.logits[0, -1, :]
        clean_pred = torch.argmax(clean_logits).item()
        clean_prob = float(torch.softmax(clean_logits, dim=0)[clean_pred])
        clean_activations = dict(self._activations)
        self.remove_hooks()

        causal_effects = {}
        for layer_idx in range(self.n_layers):
            effect = self._measure_layer_effect(
                tokens, layer_idx, clean_activations, clean_pred)
            causal_effects[layer_idx] = effect

        return {
            'prompt': prompt,
            'predicted_token': self.tokenizer.decode([clean_pred]),
            'predicted_prob': clean_prob,
            'causal_effects': causal_effects,
            'n_layers': self.n_layers,
        }

    def _measure_layer_effect(self, tokens, layer_idx, clean_acts, target_token):
        noise_scale = 3.0 * (clean_acts.get(
            'layer_0_mlp', torch.zeros(1)).std().item() + 1e-6)

        def patch_hook(module, inp, out):
            key = f"layer_{layer_idx}_mlp"
            if key in clean_acts:
                clean = clean_acts[key].to(self.device)
                if isinstance(out, tuple):
                    return (clean,) + out[1:]
                return clean
            return out

        def make_noise_hook(scale):
            def hook(module, inp, out):
                if isinstance(out, tuple):
                    return (out[0] + torch.randn_like(out[0]) * scale,) + out[1:]
                return out + torch.randn_like(out) * scale
            return hook

        noise_hooks = []
        for idx in range(self.n_layers):
            block = self._blocks[idx]
            mlp_mod = self._get_mlp_module(block)
            if idx == layer_idx:
                h = mlp_mod.register_forward_hook(patch_hook)
            else:
                h = mlp_mod.register_forward_hook(make_noise_hook(noise_scale))
            noise_hooks.append(h)

        with torch.no_grad():
            patched_out = self.model(tokens)

        for h in noise_hooks:
            h.remove()

        patched_logits = patched_out.logits[0, -1, :]
        return float(torch.softmax(patched_logits, dim=0)[target_token])

    # ── Weight stats ────────────────────────────────────────

    def get_weight_stats(self, layer_idx):
        block = self._blocks[layer_idx]
        stats = {}
        for name, param in block.named_parameters():
            stats[name] = {
                'shape': list(param.shape),
                'mean': float(param.mean()),
                'std': float(param.std()),
                'min': float(param.min()),
                'max': float(param.max()),
                'norm': float(param.norm()),
            }
        return stats


# Backward-compatible alias
GPT2Wrapper = UnifiedModelWrapper
