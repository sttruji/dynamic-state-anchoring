"""SPARSE-0's FFN masks (results/ffn_sparsity/PLAN.md): a hook on every GPT-2 block's MLP that keeps a set of neurons per
position and replaces each dropped neuron's activation by its representative value mu (G-MoEfication).

Every position carries a phase, set per model call: Q (question tokens), T (<bot> and the latent steps) and A (<eot> and
the answer tokens). An arm sparsifies only the positions of its phases and leaves the others dense.
"""
from __future__ import annotations

import types

import torch

PHASES = "QTA"
Q, T, A = range(3)
FRACTIONS = (0.5, 0.25)
# name: (selection, fraction kept, phases sparsified); ID runs the hook with nothing dropped.
ARMS = {"ID": None,
        "O50": ("oracle", 0.5, "QTA"), "O25": ("oracle", 0.25, "QTA"),
        "S50": ("static", 0.5, "QTA"), "S25": ("static", 0.25, "QTA"),
        "O25-T": ("oracle", 0.25, "T"), "O25-Q": ("oracle", 0.25, "Q")}


def neurons(fraction: float, inner: int) -> int:
    return int(round(fraction * inner))


def ranked(score: torch.Tensor) -> torch.Tensor:
    """Indices along the last dimension, highest score first, ties to the lower index."""
    return torch.sort(score, dim=-1, descending=True, stable=True).indices


def oracle_keep(a: torch.Tensor, norms: torch.Tensor, k: int) -> torch.Tensor:
    """[..., inner] bool: per position, the k neurons with the largest |a_i| * ||w_out,i||, scored in fp32."""
    order = ranked(a.float().abs() * norms.float())[..., :k]
    return torch.zeros(a.shape, dtype=torch.bool, device=a.device).scatter_(-1, order, True)


def phase_tags(question: int = 0, latent: int = 0, answer: int = 0, device=None) -> torch.Tensor:
    """The phase of each position of one model call: question tokens, then T positions, then A positions."""
    return torch.tensor([Q] * question + [T] * latent + [A] * answer, dtype=torch.long, device=device)


class FFNMasks:
    """Installed on model.transformer.h[*].mlp. With arm None the hook computes exactly what GPT2MLP computes.

    arm:    None, or (selection, fraction, phases) from ARMS
    phase:  [positions] phase tags of the current call, shared by every row of the batch
    mu:     (selection, fraction) -> [layers, phases, inner] representative values
    static: fraction -> [layers, inner] bool, the fixed kept set
    watch:  None, or a callable (layer, a, phase) that sees every post-GELU activation [batch, positions, inner]
    """

    def __init__(self, model):
        self.blocks = [block.mlp for block in model.transformer.h]
        self.norms = [mlp.c_proj.weight.detach().float().norm(dim=1) for mlp in self.blocks]  # Conv1D: [inner, width]
        self.inner = self.blocks[0].c_fc.nf
        self.arm, self.phase, self.mu, self.static, self.watch = None, None, {}, {}, None
        for layer, mlp in enumerate(self.blocks):
            mlp.forward = types.MethodType(self._forward(layer), mlp)

    def _forward(self, layer: int):
        def forward(mlp, hidden_states):
            a = mlp.act(mlp.c_fc(hidden_states))
            if self.watch is not None:
                self.watch(layer, a, self.phase)
            if self.arm is not None:
                a = self.masked(layer, a)
            return mlp.dropout(mlp.c_proj(a))
        return forward

    def keep(self, layer: int, a: torch.Tensor) -> torch.Tensor:
        """[batch, positions, inner] bool: the neurons the arm keeps; positions outside its phases keep all."""
        selection, fraction, phases = self.arm
        if selection == "oracle":
            keep = oracle_keep(a, self.norms[layer], neurons(fraction, self.inner))
        else:
            keep = self.static[fraction][layer].to(a.device).expand(a.shape)
        sparse = torch.isin(self.phase, torch.tensor([PHASES.index(p) for p in phases], device=self.phase.device))
        return keep | ~sparse.to(a.device)[None, :, None]

    def masked(self, layer: int, a: torch.Tensor) -> torch.Tensor:
        selection, fraction, _ = self.arm
        mu = self.mu[(selection, fraction)][layer].to(a.device, a.dtype)[self.phase.to(a.device)]  # [positions, inner]
        return torch.where(self.keep(layer, a), a, mu)


def active_params(fraction: float, width: int = 768, inner: int = 3072, layers: int = 12) -> int:
    """Active non-embedding parameters per token with `fraction` of each FFN's neurons kept: attention, layer norms and
    the kept neurons' c_fc columns and c_proj rows. The dropped neurons' constant sum(mu_i * w_out,i) is one vector per
    layer and phase, folded into c_proj's bias, so it adds nothing per token."""
    kept = neurons(fraction, inner)
    attention = width * 3 * width + 3 * width + width * width + width
    ffn = width * kept + kept + kept * width + width
    return layers * (attention + 4 * width + ffn) + 2 * width
