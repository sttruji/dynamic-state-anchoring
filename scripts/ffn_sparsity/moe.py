"""D1-R's mixture-of-experts FFN (results/ffn_sparsity/d1r/PLAN.md), simulated with masks on SPARSE-0's hook.

Every layer's FFN is split into shared neurons that are always on and 12 routed experts, of which 4 are on per position.
Dropped neurons stand in at their representative values mu per phase, as in SPARSE-0, so the constant they add tracks the
trainable c_proj. The arms:
    moe     a frozen 2-layer MLP router per layer picks the experts from the FFN input
    hash    a fixed random table picks them: by the token id at Q and A, by the step index at T (<bot> is step 0)
    prune   a fixed set of as many neurons, the top by Taylor importance (SPARSE-0's S50 at 50%)
    None    nothing dropped (the dense model)
"""
from __future__ import annotations

import torch
from torch import nn

from sparsity import FFNMasks, active_params, ranked

EXPERTS, ACTIVE, HIDDEN = 12, 4, 128


def make_router(width: int, experts: int = EXPERTS, hidden: int = HIDDEN) -> nn.Module:
    return nn.Sequential(nn.Linear(width, hidden), nn.ReLU(), nn.Linear(hidden, experts))


def router_params(width: int = 768, layers: int = 12, experts: int = EXPERTS, hidden: int = HIDDEN) -> int:
    return layers * (width * hidden + hidden + hidden * experts + experts)


def top_experts(scores: torch.Tensor, k: int = ACTIVE) -> torch.Tensor:
    """[..., experts] bool: the k highest scores, scored in fp32, ties to the lower index."""
    order = ranked(scores.float())[..., :k]
    return torch.zeros(scores.shape, dtype=torch.bool, device=scores.device).scatter_(-1, order, True)


def hash_table(layers: int, keys: int, experts: int = EXPERTS, active: int = ACTIVE, seed: int = 11) -> torch.Tensor:
    """[layers, keys, experts] bool: a fixed random choice of `active` experts per layer and key."""
    generator = torch.Generator().manual_seed(seed)
    order = torch.rand(layers, keys, experts, generator=generator).argsort(-1)[..., :active]
    return torch.zeros(layers, keys, experts, dtype=torch.bool).scatter_(-1, order, True)


def step_key(vocab: int, step: int) -> int:
    """The hash key of thinking position `step`: <bot> is 0, latent step j is j. Token ids key Q and A."""
    return vocab + step


def expert_norms(centered: torch.Tensor, weight: torch.Tensor, members: list[torch.Tensor]) -> torch.Tensor:
    """[..., experts]: ||W_out,e^T (a_e - mu_e)|| per routed expert, from centered activations [..., inner]."""
    return torch.stack([(centered[..., m] @ weight[m]).norm(dim=-1) for m in members], -1)


def moe_active_params(arm: str | None, width: int = 768, inner: int = 3072, layers: int = 12) -> int:
    """Active non-embedding parameters per token. MoE adds its routers; the hash table and the masks hold no weights."""
    if arm is None:
        return active_params(1.0, width, inner, layers)
    return active_params(0.5, width, inner, layers) + (router_params(width, layers) if arm == "moe" else 0)


class MoEMasks(FFNMasks):
    """FFNMasks with D1-R's selections. arm is None or one of "moe", "hash", "prune"; keys holds the current call's hash
    keys [batch, positions]. With audit set to a dict, the moe and hash arms also count how often their experts are the
    oracle's (the 4 with the largest contribution norms about the static means), per layer and phase."""

    def __init__(self, model, conversion: dict | None = None):
        super().__init__(model)
        self.keys, self.audit = None, None
        if conversion is not None:
            self.load(conversion)

    def load(self, conversion: dict) -> None:
        device = self.norms[0].device
        self.shared = conversion["shared"].to(device)
        self.owner = conversion["owner"].to(device)
        self.prune = conversion["prune"].to(device)
        self.table = conversion["hash"].to(device)
        self.mu = {name: conversion["mu"][name].to(device) for name in ("moe", "hash", "prune")}
        self.experts = int(self.owner.max()) + 1
        self.members = [[torch.nonzero(row == e).flatten() for e in range(self.experts)] for row in self.owner]
        width = self.blocks[0].c_fc.weight.shape[0]
        with torch.random.fork_rng(devices=[]):  # building the routers must not move the training run's random state
            self.routers = [make_router(width, self.experts) for _ in self.blocks]
        for router, state in zip(self.routers, conversion["routers"]):
            router.load_state_dict(state)
            router.to(device).eval().requires_grad_(False)

    def _forward(self, layer: int):
        def forward(mlp, hidden_states):
            a = mlp.act(mlp.c_fc(hidden_states))
            if self.arm is not None:
                mu = self.mu[self.arm][layer].to(a.dtype)[self.phase.to(a.device)]
                a = torch.where(self.keep(layer, hidden_states, a), a, mu)
            return mlp.dropout(mlp.c_proj(a))
        return forward

    @torch.no_grad()
    def chosen(self, layer: int, x: torch.Tensor) -> torch.Tensor:
        """[batch, positions, experts] bool: the experts the arm turns on."""
        if self.arm == "moe":
            with torch.autocast(x.device.type, enabled=False):
                return top_experts(self.routers[layer](x.float()))
        return self.table[layer][self.keys.to(self.table.device)]

    @torch.no_grad()
    def keep(self, layer: int, x: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        if self.arm == "prune":
            return self.prune[layer].expand(a.shape)
        chosen = self.chosen(layer, x)
        if self.audit is not None:
            self.count(layer, a, chosen)
        return chosen[..., self.owner[layer].clamp(min=0)] | self.shared[layer]

    def count(self, layer: int, a: torch.Tensor, chosen: torch.Tensor) -> None:
        centered = a.float() - self.mu["prune"][layer][self.phase.to(a.device)]
        best = top_experts(expert_norms(centered, self.blocks[layer].c_proj.weight.float(), self.members[layer]))
        hits = (best & chosen).sum(-1).float() / best.sum(-1)  # [batch, positions]
        for p, phase in enumerate("QTA"):
            here = (self.phase == p).to(a.device)
            if here.any():
                total, n = self.audit.get((layer, phase), (0.0, 0))
                self.audit[(layer, phase)] = (total + float(hits[:, here].sum()), n + int(hits[:, here].numel()))
