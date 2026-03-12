"""
Level 1: Tokenization utilities — breaking text into tokens
and mapping them back to readable strings.
"""
from typing import List, Dict
from utils.model_loader import load_tokenizer


def tokenize_text(text: str, model_key: str) -> Dict:
    """
    Tokenize text and return full info:
    - token_ids: list of int
    - tokens: list of str (decoded individual tokens)
    - token_spans: character-level spans in original text
    """
    tokenizer = load_tokenizer(model_key)
    encoding = tokenizer(text, return_offsets_mapping=False, return_tensors="pt")
    token_ids = encoding["input_ids"][0].tolist()

    # Decode each token individually
    tokens = [tokenizer.decode([tid]) for tid in token_ids]

    return {
        "token_ids": token_ids,
        "tokens": tokens,
        "num_tokens": len(token_ids),
        "input_ids_tensor": encoding["input_ids"],
    }


def decode_token_id(token_id: int, model_key: str) -> str:
    """Decode a single token ID to its string representation."""
    tokenizer = load_tokenizer(model_key)
    return tokenizer.decode([token_id])


def get_vocab_tokens(model_key: str, start: int = 0, end: int = 1000) -> List[Dict]:
    """Get a slice of the vocabulary: id -> token string."""
    tokenizer = load_tokenizer(model_key)
    vocab_size = tokenizer.vocab_size
    end = min(end, vocab_size)
    return [
        {"id": i, "token": tokenizer.decode([i])}
        for i in range(start, end)
    ]


def find_token_id(text: str, model_key: str) -> List[int]:
    """Find the token ID(s) for a given text string."""
    tokenizer = load_tokenizer(model_key)
    return tokenizer.encode(text, add_special_tokens=False)
