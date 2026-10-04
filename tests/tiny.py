"""Tiny model + tokenizer for tests (CPU, no download)."""
from __future__ import annotations

import string

from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import PreTrainedTokenizerFast

from lorasub.modeling import build_tiny_model

TINY_CFG = dict(vocab_size=160, hidden_size=48, intermediate_size=96, num_hidden_layers=2,
                num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=512,
                bos_token_id=1, eos_token_id=2, pad_token_id=0, tie_word_embeddings=False)


def tiny_tokenizer() -> PreTrainedTokenizerFast:
    """Character-level tokenizer over printable ASCII (+ special tokens)."""
    chars = [c for c in string.printable if c not in "\x0b\x0c"]
    vocab = {"<pad>": 0, "<bos>": 1, "<eos>": 2, "<unk>": 3}
    for c in chars:
        vocab[c] = len(vocab)
    tok = Tokenizer(models.WordLevel(vocab=vocab, unk_token="<unk>"))
    tok.pre_tokenizer = pre_tokenizers.Split(pattern="", behavior="isolated")
    fast = PreTrainedTokenizerFast(tokenizer_object=tok, bos_token="<bos>", eos_token="<eos>",
                                   pad_token="<pad>", unk_token="<unk>")
    fast.padding_side = "right"
    assert len(fast) <= TINY_CFG["vocab_size"]
    return fast


def tiny_model(seed: int = 0):
    import torch

    torch.manual_seed(seed)
    return build_tiny_model(TINY_CFG)
