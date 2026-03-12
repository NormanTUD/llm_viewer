"""
Level 1: Model loading — the most fundamental building block.
Loads HuggingFace transformer models that are freely downloadable.
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Registry of freely available models
AVAILABLE_MODELS = {
    "gpt2":        {"name": "gpt2",         "desc": "GPT-2 Small (117M params, 12 layers)"},
    "gpt2-medium": {"name": "gpt2-medium",  "desc": "GPT-2 Medium (345M params, 24 layers)"},
    "gpt2-large":  {"name": "gpt2-large",   "desc": "GPT-2 Large (774M params, 36 layers)"},
    "gpt2-xl":     {"name": "gpt2-xl",      "desc": "GPT-2 XL (1.5B params, 48 layers)"},
    "distilgpt2":  {"name": "distilgpt2",   "desc": "DistilGPT-2 (82M params, 6 layers)"},
}

# In-memory cache so we don't reload
_model_cache = {}
_tokenizer_cache = {}


def list_available_models():
    """Return dict of model_key -> description."""
    return {k: v["desc"] for k, v in AVAILABLE_MODELS.items()}


def load_tokenizer(model_key: str):
    """Load and cache a tokenizer."""
    if model_key not in _tokenizer_cache:
        hf_name = AVAILABLE_MODELS[model_key]["name"]
        _tokenizer_cache[model_key] = AutoTokenizer.from_pretrained(hf_name)
    return _tokenizer_cache[model_key]


def load_model(model_key: str):
    """Load and cache a model in eval mode, on CPU (or GPU if available)."""
    if model_key not in _model_cache:
        hf_name = AVAILABLE_MODELS[model_key]["name"]
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = AutoModelForCausalLM.from_pretrained(
            hf_name,
            output_hidden_states=True,
            output_attentions=True,
        ).to(device).eval()
        _model_cache[model_key] = model
    return _model_cache[model_key]


def get_device(model_key: str):
    model = load_model(model_key)
    return next(model.parameters()).device


def get_model_config(model_key: str):
    """Return useful config info: n_layers, n_heads, d_model, vocab_size."""
    model = load_model(model_key)
    cfg = model.config
    return {
        "n_layers": cfg.n_layer,
        "n_heads": cfg.n_head,
        "d_model": cfg.n_embd,
        "vocab_size": cfg.vocab_size,
    }
