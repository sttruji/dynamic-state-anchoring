"""Idea F: KK-LSE, a key-to-key bias that counts each distinct latent entry once.

In every attention head, latent key j gets the logit bias

    b_j = -alpha * log sum_l exp(beta * (cos(k_j, k_l) - 1))

summed over the latent keys l in the cache (j included). The sum is minus the log-normalizer of a key-to-key attention, so
one attention pass sets the weights of the other. m exact copies each get -alpha * log m, so at alpha = 1 a cluster weighs
what one entry does; a distinct key gets about 0. Question and answer keys get no bias. alpha is per layer and head and
starts at zero, so the model starts as the checkpoint; beta is fixed.

The latent keys are the `count` cache entries from index `start`, which callers set before each forward: the question
width, and the number of latent steps. The patched attention is GPT-2's eager `_attn`, which every path here uses.

Plan: docs/brainstorm/INNER_LOOP_QUICK_REFERENCE.md.
"""
from __future__ import annotations

import torch
from torch import nn


def multiplicity_bias(keys, alpha, beta: float):
    """-alpha * log of each key's soft count among `keys` [batch, heads, n, dim], for alpha [heads]: [batch, heads, n]."""
    unit = nn.functional.normalize(keys.float(), dim=-1)
    kernel = torch.exp(beta * (unit @ unit.transpose(-1, -2) - 1))
    return -alpha[None, :, None] * torch.log(kernel.sum(-1))


class KKBias(nn.Module):
    """KK-LSE on every layer of a GPT-2: one alpha per layer and head, added to each eager attention's logits."""

    def __init__(self, model, beta: float = 14.0):
        super().__init__()
        config = model.config
        self.beta = beta
        self.alpha = nn.Parameter(torch.zeros(config.n_layer, config.n_head, device=model.transformer.wte.weight.device))
        self.start, self.count, self.active = None, 0, True
        self._patched = []
        for layer, block in enumerate(model.transformer.h):
            original = block.attn._attn
            block.attn._attn = self._wrap(layer, original)
            self._patched.append((block.attn, original))

    def _wrap(self, layer, original):
        def attend(query, key, value, attention_mask=None, head_mask=None):
            bias = self.bias(layer, key)
            if bias is not None:
                attention_mask = bias if attention_mask is None else attention_mask + bias
            return original(query, key, value, attention_mask, head_mask)
        return attend

    def bias(self, layer: int, key):
        """The additive mask [batch, heads, 1, keys] for one layer, or None when fewer than two latents are cached."""
        if not self.active or self.start is None:
            return None
        end = min(self.start + self.count, key.shape[-2])
        if end - self.start < 2:  # one entry's count is exactly 1: no bias
            return None
        b = multiplicity_bias(key[..., self.start:end, :], self.alpha[layer], self.beta)
        return nn.functional.pad(b, (self.start, key.shape[-2] - end))[:, :, None, :]

    def detach_hooks(self) -> None:
        for attn, original in self._patched:
            attn._attn = original
        self._patched = []


def save_kk(kk: KKBias, path) -> None:
    torch.save({"beta": kk.beta, "alpha": kk.alpha.detach().cpu()}, path)


def load_kk(model, path) -> KKBias:
    saved = torch.load(path, map_location="cpu", weights_only=True)
    kk = KKBias(model, beta=saved["beta"])
    with torch.no_grad():
        kk.alpha.copy_(saved["alpha"])
    return kk
