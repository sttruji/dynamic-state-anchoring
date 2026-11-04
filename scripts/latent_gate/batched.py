"""Batched latent loops for CODI-style models (GPT-2 or LLaMA) under the latent-entry policies of the held-out/scale study.

Plan: results/latent_gate/heldout_scale/PLAN.md. Rows are left-padded and positions are explicit, so latent step t of a row
sits at position bot + t whatever the padding, as in the batch-1 `stage1_eval.py`. Eviction masks instead of deleting: an
evicted entry gets HIDE on its keys, which takes it out of every softmax exactly as deleting it would (exp(-1e4) is 0 at
FP32), while the cache keeps one column per latent step so a batch's rows stay aligned. Latent entry j of a row sits at
cache column Q + j - 1, Q being the padded question length (<bot> is column Q - 1).

The latent bias reaches GPT-2 through the patched eager attention in gate.py, and LLaMA through a 4D additive attention
mask, which transformers 4.46 adds to the logits unchanged.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from gate import HIDE, install_bias, set_bias, top_visible

POLICIES = ("plain", "keep_first", "window", "random", "logn", "dedup_soft", "shift", "copy_last", "copy_pair", "gate",
            "gate_soft")
EVICTING = ("keep_first", "window", "random", "gate")
COPYING = ("copy_last", "copy_pair")


@dataclass
class Loop:
    """What the loops need from a loaded CODI model."""
    model: torch.nn.Module
    projection: torch.nn.Module
    tokenizer: object
    bot_id: int
    eot_id: int
    pad_id: int
    eos_id: int
    vocab_size: int
    llama: bool
    max_new_tokens: int = 256
    gate: torch.nn.Module | None = None

    @classmethod
    def of(cls, codi, gate=None) -> "Loop":
        llama = getattr(codi, "family", "gpt2") == "llama"
        if not llama:
            install_bias(codi.model)
        tokenizer = codi.tokenizer
        return cls(codi.model, codi.projection, tokenizer, codi.bot_id, codi.eot_id,
                   tokenizer.pad_token_id if llama else tokenizer.eos_token_id, tokenizer.eos_token_id, codi.vocab_size,
                   llama, codi.max_new_tokens, gate)

    def encode(self, question: str) -> list[int]:
        return self.tokenizer.encode(question.strip().replace("  ", " ")) + [self.bot_id]

    def decode(self, tokens: list[int]) -> str:
        return self.tokenizer.decode([t for t in tokens if t < len(self.tokenizer)], skip_special_tokens=True)


def forward(loop: Loop, *, ids=None, embeds=None, cache=None, mask, positions, bias=None, hidden=False, attentions=False):
    """One forward over new tokens. `mask` [batch, keys] is 1 on real keys (past and new); `bias` [batch, keys] or None."""
    device = mask.device
    if loop.llama:
        width = (ids if ids is not None else embeds).shape[1]
        if width > 1:  # the question prefill: transformers builds the causal and padding mask
            assert bias is None, "no latent bias before the latents exist"
            attention = mask
        else:
            attention = (1.0 - mask[:, None, None, :].float()) * torch.finfo(torch.float32).min
            if bias is not None:
                attention = attention + bias[:, None, None, :].to(device)
        return loop.model(input_ids=ids, inputs_embeds=embeds, past_key_values=cache, attention_mask=attention,
                          position_ids=positions, use_cache=True, output_hidden_states=hidden,
                          output_attentions=attentions)
    set_bias(loop.model, None if bias is None else bias[:, None, None, :])
    try:
        return loop.model(input_ids=ids, inputs_embeds=embeds, past_key_values=cache, attention_mask=mask,
                          position_ids=positions, use_cache=True, output_hidden_states=hidden,
                          output_attentions=attentions)
    finally:
        set_bias(loop.model, None)


@dataclass
class Asked:
    """A batch's encoded questions: the cache after <bot>, the first latent and each row's <bot> position."""
    cache: tuple
    first: torch.Tensor
    mask: torch.Tensor
    bot: torch.Tensor


@torch.no_grad()
def ask(loop: Loop, questions: list[list[int]], device) -> Asked:
    width = max(len(q) for q in questions)
    ids = torch.tensor([[loop.pad_id] * (width - len(q)) + q for q in questions], device=device)
    mask = torch.tensor([[0] * (width - len(q)) + [1] * len(q) for q in questions], device=device)
    positions = (mask.cumsum(1) - 1).clamp(min=0)
    out = forward(loop, ids=ids, mask=mask, positions=positions, hidden=True)
    assert isinstance(out.past_key_values, tuple), "branching policies off one prefill relies on an immutable cache"
    return Asked(out.past_key_values, loop.projection(out.hidden_states[-1][:, -1:, :]), mask, positions[:, -1])


def latent_bias(policy: str, budget: int, visible: torch.Tensor, fed: list, log_gates: list) -> torch.Tensor | None:
    """Additive bias [batch, n] on the n latent entries, as stage1_eval's `weights`, with HIDE on evicted entries."""
    batch, n = visible.shape
    bias = torch.zeros(batch, n, device=visible.device)
    if policy in ("gate", "gate_soft") and log_gates:
        bias = torch.stack(log_gates, 1).float()
    elif n > budget and policy == "logn":
        bias = torch.full((batch, n), -math.log(n / budget), device=visible.device)
    elif n > budget and policy == "dedup_soft":
        vectors = F.normalize(torch.stack(fed, 1), dim=-1)
        bias = -((vectors @ vectors.transpose(1, 2)) >= 0.95).sum(2).float().log()
    bias = torch.where(visible, bias, torch.full_like(bias, HIDE))
    return bias if bool((bias != 0).any()) else None


def shares(attentions, question: int, n: int, budget: int) -> list[dict]:
    """Per row, the newest query's attention share (mean over heads) on latents 1..budget and on later latents, by layer."""
    a = torch.stack([layer[:, :, -1, :] for layer in attentions]).mean(2).float()  # [layers, batch, keys]
    early = a[:, :, question:question + min(n, budget)].sum(-1)
    later = a[:, :, question + budget:question + n].sum(-1)
    return [{"latent_1_6": [round(float(x), 5) for x in early[:, r]], "latent_later": [round(float(x), 5) for x in later[:, r]]}
            for r in range(a.shape[1])]


@torch.no_grad()
def readout(loop: Loop, cache, asked: Asked, n: int, bias: torch.Tensor | None, start: torch.Tensor, attention: bool,
            budget: int):
    """Greedy answer after <eot> for every row; the latent bias stays on the n latent entries throughout."""
    batch, question = asked.mask.shape
    device = asked.mask.device
    embed = loop.model.get_input_embeddings()
    step = embed(torch.full((batch, 1), loop.eot_id, device=device))
    tokens, done, att = [[] for _ in range(batch)], torch.zeros(batch, dtype=torch.bool, device=device), None
    for offset in range(loop.max_new_tokens):
        keys = question + n + 1 + offset
        mask = torch.cat([asked.mask, torch.ones(batch, keys - question, dtype=asked.mask.dtype, device=device)], 1)
        full = None if bias is None else torch.cat([torch.zeros(batch, question, device=device), bias,
                                                    torch.zeros(batch, 1 + offset, device=device)], 1)
        out = forward(loop, embeds=step, cache=cache, mask=mask, positions=(start + offset)[:, None], bias=full,
                      attentions=attention and offset == 0)
        if attention and offset == 0:
            att = shares(out.attentions, question, n, budget)
        cache = out.past_key_values
        token = out.logits[:, -1, :loop.vocab_size - 1].argmax(-1)
        for row in (~done).nonzero().flatten().tolist():
            tokens[row].append(int(token[row]))
        done |= token == loop.eos_id
        if bool(done.all()):
            break
        step = embed(token[:, None])
    return tokens, att


@torch.no_grad()
def run(loop: Loop, policy: str, asked: Asked, depths: list[int], budget: int, seeds: list[int],
        attention: bool = False) -> list[dict]:
    """One latent loop per row under `policy`, read out at the depths the policy applies to; one dict per row."""
    assert policy in POLICIES, policy
    batch, question = asked.mask.shape
    device = asked.mask.device
    shift_to = max(depths)
    deepest = budget if policy == "shift" else max(depths)
    cache, latent = asked.cache, asked.first
    visible = torch.zeros(batch, 0, dtype=torch.bool, device=device)
    fed, log_gates, results = [], [], [{} for _ in range(batch)]
    generators = [torch.Generator().manual_seed(seed) for seed in seeds]
    for step in range(1, deepest + 1):
        n = step - 1
        if policy in COPYING and step > budget:  # an exact copy of entry `budget` (or of the last pair, alternating)
            source = budget if policy == "copy_last" else (budget - 1 if (step - budget) % 2 == 1 else budget)
            column = question + source - 1
            cache = tuple((torch.cat([k, k[:, :, column:column + 1]], 2), torch.cat([v, v[:, :, column:column + 1]], 2))
                          for k, v in cache)
            fed.append(fed[source - 1])
        else:
            bias = latent_bias(policy, budget, visible, fed, log_gates)
            mask = torch.cat([asked.mask, torch.ones(batch, n + 1, dtype=asked.mask.dtype, device=device)], 1)
            full = None if bias is None else torch.cat([torch.zeros(batch, question, device=device), bias,
                                                        torch.zeros(batch, 1, device=device)], 1)
            out = forward(loop, embeds=latent, cache=cache, mask=mask, positions=(asked.bot + step)[:, None], bias=full,
                          hidden=True)
            cache, hidden = out.past_key_values, out.hidden_states[-1][:, -1, :]
            fed.append(latent[:, 0])
            if policy in ("gate", "gate_soft"):
                log_gates.append(F.logsigmoid(loop.gate(hidden.float())))
            latent = loop.projection(hidden[:, None, :])
        visible = torch.cat([visible, torch.ones(batch, 1, dtype=torch.bool, device=device)], 1)
        if policy in EVICTING and step > budget:
            if policy == "keep_first":
                visible[:, budget:] = False
            elif policy == "window":
                visible[:, :step - budget] = False
            elif policy == "gate":
                visible = top_visible(torch.stack(log_gates, 1), budget)
            else:  # random: stage1_eval's draw, a per-question generator over the entries still kept
                for row in range(batch):
                    kept = visible[row].nonzero().flatten()
                    drop = int(torch.randint(len(kept), (1,), generator=generators[row]))
                    visible[row, kept[drop]] = False
        if step in depths and (policy == "plain" or (policy == "shift" and step == budget) or
                               (step > budget and policy != "shift")):
            n = step
            bias = latent_bias(policy, budget, visible, fed, log_gates) if policy not in COPYING else None
            start = asked.bot + (shift_to if policy == "shift" else step) + 1
            tokens, att = readout(loop, cache, asked, n, bias, start, attention, budget)
            repeats = None
            if policy == "dedup_soft":  # entries whose fed latent is a near-copy (cosine >= 0.95) of an earlier one
                vectors = F.normalize(torch.stack(fed, 1), dim=-1)
                earlier = torch.ones(n, n, dtype=torch.bool, device=device).tril(-1)
                repeats = (((vectors @ vectors.transpose(1, 2)) >= 0.95) & earlier).any(2)
            for row in range(batch):
                cell = {"tokens": tokens[row], "kept": [j + 1 for j in visible[row].nonzero().flatten().tolist()]}
                if repeats is not None:
                    cell["repeats"] = [j + 1 for j in repeats[row].nonzero().flatten().tolist()]
                if att is not None:
                    cell["attn"] = att[row]
                results[row][str(step)] = cell
    return results
