"""LoRA adapters for GPT-2's Conv1D projections, following CODI's GPT-2 recipe.

CODI (Shen et al., 2025) trains GPT-2 with peft LoRA on ``c_attn``, ``c_proj``
and ``c_fc`` (rank 128, alpha 32, dropout 0.1) and freezes the rest of the
backbone. Here the one other trainable backbone weight is the ``<SA>`` row of
the tied embedding, which is new and which the explicit teacher must predict.
Saved checkpoints fold the adapters into ordinary GPT-2 weights, so inference
code loads them like any other backbone.
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn


TARGETS = (("attn", "c_attn"), ("attn", "c_proj"), ("mlp", "c_fc"), ("mlp", "c_proj"))


class LoRAConv1D(nn.Module):
    """``base(x) + scale * B(A(dropout(x)))`` with the base projection frozen."""

    def __init__(self, base: nn.Module, rank: int, alpha: float, dropout: float):
        super().__init__()
        fan_in, fan_out = base.weight.shape  # GPT-2 Conv1D stores [input, output]
        options = {"device": base.weight.device, "dtype": base.weight.dtype}
        self.base = base.requires_grad_(False)
        self.lora_a = nn.Parameter(torch.empty(rank, fan_in, **options))
        self.lora_b = nn.Parameter(torch.zeros(fan_out, rank, **options))
        # peft's default initialization; B = 0 starts exactly at the base model.
        nn.init.kaiming_uniform_(self.lora_a, a=math.sqrt(5))
        self.scale = alpha / rank
        self.dropout = nn.Dropout(dropout)

    def delta(self, x, outputs=slice(None)):
        """Adapter output, optionally for a slice of the projection's output columns."""
        return F.linear(F.linear(self.dropout(x), self.lora_a), self.lora_b[outputs]) * self.scale

    def forward(self, x):
        return self.base(x) + self.delta(x)

    def merged_weight(self):
        return (self.base.weight + self.scale * (self.lora_b @ self.lora_a).T).contiguous()


def apply_lora(model, rank: int, alpha: float, dropout: float):
    """Freeze an AnchoredLoop's GPT-2 backbone except new adapters and the marker row."""
    backbone = model.backbone
    if backbone.kind != "gpt2":
        raise ValueError("LoRA adapters currently support GPT-2 only")
    if rank < 1 or alpha <= 0 or not 0 <= dropout < 1:
        raise ValueError("LoRA needs a positive rank and alpha and dropout in [0, 1)")
    if model.config.marker_token_id is None:
        raise ValueError("LoRA training keeps the marker row trainable; register the marker first")
    backbone.requires_grad_(False)
    for block in backbone.model.transformer.h:
        for parent, name in TARGETS:
            owner = getattr(block, parent)
            setattr(owner, name, LoRAConv1D(getattr(owner, name), rank, alpha, dropout))
    embedding = backbone.embeddings.weight  # tied to the output layer
    embedding.requires_grad_(True)
    keep = torch.zeros(embedding.shape[0], 1, dtype=embedding.dtype)
    keep[model.config.marker_token_id] = 1
    # Only the <SA> row learns. The optimizer must not decay this weight, or the
    # frozen rows would drift; see the trainer's parameter groups.
    embedding.register_hook(lambda grad: grad * keep.to(grad.device))
    return model


def has_lora(module: nn.Module) -> bool:
    return any(isinstance(child, LoRAConv1D) for child in module.modules())


def merged_state_dict(module: nn.Module) -> dict:
    """``module.state_dict()`` with every adapter folded into a plain Conv1D weight."""
    state = module.state_dict()
    for name, child in module.named_modules():
        if isinstance(child, LoRAConv1D):
            for key in ("lora_a", "lora_b", "base.weight", "base.bias"):
                del state[f"{name}.{key}"]
            state[f"{name}.weight"] = child.merged_weight().detach()
            state[f"{name}.bias"] = child.base.bias.detach()
    return state
