"""Step-level selection among sampled latent steps, on CODI-style GPT-2 checkpoints.

A candidate at latent step s is one execution of that step: its cache entries and the next latent. Candidate 0 runs with
dropout off and is the model's own step. The others rerun the step with MC dropout at rate p. A selector that keeps
candidate 0 at every step reproduces the checkpoint exactly.

SIM-CoT's step decoder reads each candidate's next latent back into a `<<a op b=c>>` step. That gives features that need
no training: does the step parse, is its arithmetic right, are its operands grounded in the question or earlier steps,
and how confident is the decoder. `choose` is the rule fixed before Stage 0 ran.

Plan: docs/brainstorm/STEP_SELECTION.md.
"""
from __future__ import annotations

import contextlib
import re
import sys
from pathlib import Path

import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "inner_loop"))
from looped import arithmetic, latent_step, parse_step  # noqa: E402

NUMBER = re.compile(r"\d+(?:\.\d+)?")
CONSTANTS = frozenset({1.0, 2.0, 3.0, 4.0, 7.0, 10.0, 12.0, 24.0, 30.0, 60.0, 100.0, 1000.0})


@contextlib.contextmanager
def mc_dropout(modules, p: float):
    """Every nn.Dropout inside `modules` at rate p and switched on; rates and modes are restored on exit."""
    drops = [m for module in modules for m in module.modules() if isinstance(m, nn.Dropout)]
    saved = [(m, m.p, m.training) for m in drops]
    try:
        for m in drops:
            m.p = p
            m.train()
        yield
    finally:
        for m, rate, training in saved:
            m.p = rate
            m.train(training)


def expand(cache, rows: int):
    """A batch-1 cache (GPT-2's tuple of (key, value) per layer) copied into `rows` rows."""
    return tuple((k.expand(rows, -1, -1, -1).contiguous(), v.expand(rows, -1, -1, -1).contiguous()) for k, v in cache)


def pick(cache, row: int):
    return tuple((k[row:row + 1], v[row:row + 1]) for k, v in cache)


def step(model, projection, latent, cache, position: int):
    """One latent step for every row at the same position: the next latents [rows, 1, width] and the cache."""
    hidden, cache = latent_step(model, None, latent, cache,
                                torch.full((latent.shape[0], 1), position, device=latent.device))
    return projection(hidden), cache


def sampled(model, projection, latent, cache, position: int, p: float):
    """`step` with MC dropout at rate p in the model and the projection, each row drawing its own masks."""
    with mc_dropout((model, projection), p):
        return step(model, projection, latent, cache, position)


def sample_step(model, projection, latent, cache, position: int, rows: int, p: float):
    """`rows` dropout reruns of one latent step from a shared batch-1 state: their next latents and caches."""
    return sampled(model, projection, latent.expand(rows, -1, -1), expand(cache, rows), position, p)


@torch.no_grad()
def readout(model, codi, cache, position: int, rows: int, max_tokens: int | None = None) -> list[list[int]]:
    """Greedy answers after <eot> for `rows` chains at the same position, as CODIGPT2 reads them, for at most
    `max_tokens` tokens and never past CODI's own limit: token lists."""
    embed = model.get_input_embeddings()
    device = cache[0][0].device
    current = embed(torch.full((rows, 1), codi.eot_id, device=device))
    tokens = [[] for _ in range(rows)]
    done = torch.zeros(rows, dtype=torch.bool, device=device)
    for offset in range(min(max_tokens or codi.max_new_tokens, codi.max_new_tokens)):
        out = model(inputs_embeds=current, past_key_values=cache, use_cache=True,
                    position_ids=torch.full((rows, 1), position + offset, device=device))
        cache = out.past_key_values
        token = out.logits[:, -1, :codi.vocab_size - 1].argmax(-1)
        for i in (~done).nonzero().flatten().tolist():
            tokens[i].append(int(token[i]))
        done |= token == codi.tokenizer.eos_token_id
        if bool(done.all()):
            break
        current = embed(token[:, None])
    return tokens


@torch.no_grad()
def read(decoder, embed, latents, eos: int, max_tokens: int = 32):
    """Greedy decodes of latents [n, width], each read on its own as SIM-CoT's decoder was trained: the token lists
    (end-of-text dropped) and each decode's mean token log-probability, end-of-text included."""
    n = len(latents)
    current, cache = latents[:, None, :], None
    done = torch.zeros(n, dtype=torch.bool, device=latents.device)
    total = torch.zeros(n, device=latents.device)
    count = torch.zeros(n, device=latents.device)
    out = [[] for _ in range(n)]
    for _ in range(max_tokens):
        result = decoder(inputs_embeds=current, past_key_values=cache, use_cache=True)
        cache = result.past_key_values
        logp = result.logits[:, -1].float().log_softmax(-1)
        token = logp.argmax(-1)
        live = ~done
        total += torch.where(live, logp.gather(1, token[:, None])[:, 0], torch.zeros_like(total))
        count += live.float()
        for i in live.nonzero().flatten().tolist():
            out[i].append(int(token[i]))
        done |= token == eos
        if bool(done.all()):
            break
        current = embed(token[:, None])
    return [[t for t in seq if t != eos] for seq in out], (total / count.clamp(min=1)).tolist()


def numbers(text: str) -> set[float]:
    return {float(x) for x in NUMBER.findall((text or "").replace(",", ""))}


def as_number(text):
    try:
        return float(str(text).replace(",", ""))
    except ValueError:
        return None


def features(text: str, logprob: float, known) -> dict:
    """The decoder features of one candidate: does its decoded step parse, is its arithmetic right, are its operands all
    in `known` (question numbers, earlier results and CONSTANTS), the decoder's confidence, and the step's result."""
    parsed = parse_step(text)
    if parsed is None:
        return {"text": text, "logprob": logprob, "parsed": False, "valid": False, "grounded": False, "result": None}
    expression, result = parsed
    value, claimed = arithmetic(expression), as_number(result)
    operands = numbers(expression)
    return {"text": text, "logprob": logprob, "parsed": True,
            "valid": value is not None and claimed is not None and abs(value - claimed) < 1e-6,
            "grounded": bool(operands) and all(any(abs(x - k) < 1e-6 for k in known) for x in operands),
            "result": claimed}


def rank(f: dict) -> tuple[int, int]:
    return int(f["valid"]), int(f["grounded"])


def choose(candidates: list[dict]) -> int:
    """The fixed rule: the best (valid arithmetic, grounded operands) rank wins; candidate 0 on a tie for it, else the
    most confident decode among the candidates with that rank."""
    best = max(rank(f) for f in candidates)
    if rank(candidates[0]) == best:
        return 0
    return max((i for i, f in enumerate(candidates) if rank(f) == best), key=lambda i: candidates[i]["logprob"])
