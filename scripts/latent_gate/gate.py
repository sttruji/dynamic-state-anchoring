"""Shared pieces of the latent-entry gate: an additive attention bias for GPT-2, the gate head and gold labels.

The bias rides on GPT-2's eager attention: every block adds `extra_bias` (broadcast to [batch, heads, queries, keys]) to its
attention logits before the softmax. A soft gate puts log g_j on the keys of latent entry j; a window puts a large negative
value on the entries it hides.
"""
from __future__ import annotations

import re

import torch
from torch import nn
from transformers.models.gpt2 import modeling_gpt2

HIDE = -1e4  # an additive bias that removes a key from the softmax in FP32 and BF16


def install_bias(model) -> None:
    """Patch GPT-2 attention once so each block adds its `extra_bias` to the attention logits."""
    attention = modeling_gpt2.GPT2Attention
    if not getattr(attention, "_latent_gate_patched", False):
        original = attention._attn

        def _attn(self, query, key, value, attention_mask=None, head_mask=None):
            extra = getattr(self, "extra_bias", None)
            if extra is not None:
                extra = extra.to(query.dtype)
                attention_mask = extra if attention_mask is None else attention_mask + extra
            scale = getattr(self, "query_scale", None)
            if scale is not None:  # multiplies every logit of these queries (log-n / SSMax-style sharpening)
                query = query * scale
            return original(self, query, key, value, attention_mask, head_mask)

        attention._attn = _attn
        attention._latent_gate_patched = True
    assert model.config._attn_implementation == "eager", "the bias needs eager attention"
    set_bias(model, None)


def set_bias(model, bias, scale=None) -> None:
    for block in model.transformer.h:
        block.attn.extra_bias = bias
        block.attn.query_scale = scale


class Gate(nn.Module):
    """Scores a latent entry from the final hidden state at its position; log-sigmoid of the score is its bias."""

    def __init__(self, width: int = 768, hidden: int = 128, start: float = 3.0):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(width, hidden), nn.GELU(), nn.Linear(hidden, 1))
        nn.init.zeros_(self.net[2].weight)
        nn.init.constant_(self.net[2].bias, start)  # g = sigmoid(3) = 0.95 everywhere, so training starts near no gate

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.net(hidden).squeeze(-1)


def top_visible(scores: torch.Tensor, budget: int) -> torch.Tensor:
    """The entries a hard gate keeps: the `budget` highest scores along the last axis, a tie going to the older entry.

    Inference evicts the lowest score whenever more than `budget` entries exist, a tie evicting the newest. Scores are fixed
    per entry, so what survives that stream of evictions is exactly this set, at every step.
    """
    order = scores.argsort(dim=-1, descending=True, stable=True)[..., :budget]
    return torch.zeros_like(scores, dtype=torch.bool).scatter(-1, order, True)


def numeric(text: str):
    try:
        return float(text.replace(",", "").strip())
    except ValueError:
        return None


def gold_results(steps) -> list[float]:
    """Intermediate results of a GSM8K-Aug rationale, from `<<a*b=c>>` steps (a list, or one string)."""
    text = " ".join(steps) if isinstance(steps, (list, tuple)) else (steps or "")
    return [x for x in (numeric(m) for m in re.findall(r"=\s*([-\d.,]+)\s*>>", text)) if x is not None]


def decode_labels(tokenizer, logits: torch.Tensor, results: list[list[float]]) -> torch.Tensor:
    """1.0 where a row's logit-lens top-1 token is one of that row's gold intermediate results."""
    values = [numeric(tokenizer.decode([int(i)])) for i in logits.argmax(-1)]
    return torch.tensor([float(v is not None and v in r) for v, r in zip(values, results)], device=logits.device)
