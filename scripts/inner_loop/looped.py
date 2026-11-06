"""Idea A: an inner loop over a middle block of GPT-2 layers inside each CODI latent step.

One latent step runs layers [0, first) once, the block [first, last) k times at the same position, and layers [last, n)
once; then ln_f and CODI's projection make the next latent, as before. Pass 1 of the block is plain CODI. Pass r >= 2 is

    x_r = x_{r-1} + g * (B(x_{r-1} + e @ W) - x_{r-1})

with e the block's input, g a per-channel gate, W an input re-injection and B the block with a LoRA switched on. g, W and
the LoRA's second factor start at zero, so every k reproduces the checkpoint until training moves them. The block keeps the
keys and values of pass 1 (kv="first"), so a step still adds exactly one cache entry per layer; kv="last" keeps the final
pass's instead (not an identity at the start).

The SIM-CoT GPT-2 checkpoint also ships the step decoder SIM-CoT trained to read each latent back into its `<<a op b=c>>`
step; it gives the loop a per-step loss and serves as the probe of what a latent holds.

Plan: docs/brainstorm/INNER_LOOP_QUICK_REFERENCE.md.
"""
from __future__ import annotations

import ast
import hashlib
import math
import operator
import re

import torch
from torch import nn

TARGETS = ("attn.c_attn", "attn.c_proj", "mlp.c_fc", "mlp.c_proj")
# The checkpoint fetch_simcot.py pins; here its `decoder.*` tensors are kept instead of dropped.
SIMCOT_REPO, SIMCOT_REVISION = "internlm/SIM_COT-GPT2-CODI", "8685b42870510ba33f7caae30f7a24a2020d2e50"
SIMCOT_SHA256 = "9eaa159155d1db8e8fa3024b959b1e7640cbd5f78b738a3759b2aa61b80b409d"
STEP_STARTS, STEP_END = (16791, 9959), 4211  # GPT-2 tokens "<<", " <<" and ">>", as in SIM-CoT's get_steps


class LoopParams(nn.Module):
    """The new parameters of the inner loop: gate, input re-injection and a LoRA on the looped blocks' projections."""

    def __init__(self, model, first: int = 4, last: int = 8, rank: int = 32, alpha: float = 32.0,
                 lora_on_first: bool = False):
        super().__init__()
        width = model.config.n_embd
        self.first, self.last, self.scale = first, last, alpha / rank
        self.lora_on_first = lora_on_first  # the parameter-matched control: the same LoRA on pass 1, no loop
        self.gate = nn.Parameter(torch.zeros(width))
        self.inject = nn.Parameter(torch.zeros(width, width))
        self.lora_a, self.lora_b = nn.ParameterList(), nn.ParameterList()
        self.active = False
        self._hooks = []
        for i in range(first, last):
            for name in TARGETS:
                conv = model.transformer.h[i].get_submodule(name)
                fan_in, fan_out = conv.weight.shape  # GPT-2's Conv1D stores [in, out]
                a = nn.Parameter(torch.randn(fan_in, rank, device=conv.weight.device) / math.sqrt(fan_in))
                b = nn.Parameter(torch.zeros(rank, fan_out, device=conv.weight.device))
                self.lora_a.append(a)
                self.lora_b.append(b)
                self._hooks.append(conv.register_forward_hook(self._hook(a, b)))
        self.to(model.transformer.wte.weight.device)

    def _hook(self, a, b):
        def add_lora(module, inputs, output):
            if self.active:
                return output + (inputs[0] @ a @ b) * self.scale
        return add_lora

    def detach_hooks(self) -> None:
        for hook in self._hooks:
            hook.remove()
        self._hooks = []

    def fill(self, gate: float) -> "LoopParams":
        """A training-free loop: a constant gate, no re-injection, no LoRA (a damped re-application of the block)."""
        with torch.no_grad():
            self.gate.fill_(gate)
        return self

    def config(self) -> dict:
        return {"first": self.first, "last": self.last, "rank": self.lora_a[0].shape[1] if len(self.lora_a) else 0,
                "alpha": self.scale * (self.lora_a[0].shape[1] if len(self.lora_a) else 0),
                "lora_on_first": self.lora_on_first}


def save_loop(loop: LoopParams, path) -> None:
    torch.save({"config": loop.config(), "state": {k: v.detach().cpu() for k, v in loop.state_dict().items()}}, path)


def load_loop(model, path) -> LoopParams:
    saved = torch.load(path, map_location="cpu", weights_only=True)
    loop = LoopParams(model, **saved["config"])
    loop.load_state_dict(saved["state"])
    return loop


def latent_step(model, loop: LoopParams | None, embeds, cache, position, k: int = 1, mask=None, kv: str = "first"):
    """One latent step from input embeddings [batch, 1, width] at `position` [batch, 1].

    `cache` is GPT-2's legacy tuple of (key, value) per layer and `mask` a [batch, keys + 1] 1/0 mask or None. With no loop,
    or k = 1 and the LoRA off, this is exactly model(inputs_embeds=embeds, past_key_values=cache, ...). Returns the final
    hidden state after ln_f [batch, 1, width] and the new cache.
    """
    t = model.transformer
    hidden = t.drop(embeds + t.wpe(position))
    extended = None
    if mask is not None:  # as GPT2Model builds it for eager attention
        extended = (1.0 - mask[:, None, None, :].to(model.dtype)) * torch.finfo(model.dtype).min
    presents = [None] * len(t.h)

    def run(i, h):
        out = t.h[i](h, layer_past=cache[i], attention_mask=extended, use_cache=True)
        return out[0], out[1]

    first, last = (loop.first, loop.last) if loop is not None else (len(t.h), len(t.h))
    for i in range(first):
        hidden, presents[i] = run(i, hidden)
    if loop is not None:
        entry = hidden
        loop.active = loop.lora_on_first
        for i in range(first, last):
            hidden, presents[i] = run(i, hidden)
        loop.active = True
        for _ in range(2, k + 1):
            y = hidden + entry @ loop.inject
            for i in range(first, last):
                y, present = run(i, y)
                if kv == "last":
                    presents[i] = present
            hidden = hidden + loop.gate * (y - hidden)
        loop.active = False
    for i in range(last, len(t.h)):
        hidden, presents[i] = run(i, hidden)
    return t.ln_f(hidden), tuple(presents)


# ---------------------------------------------------------------- SIM-CoT's step decoder

def load_step_decoder(device, revision: str = SIMCOT_REVISION):
    """SIM-CoT's GPT-2 step decoder from the pinned checkpoint: a full GPT-2 with an untied head, FP32, frozen."""
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file
    from transformers import GPT2Config, GPT2LMHeadModel
    from dsa.bench.models.gpt2_vanilla import GPT2_REVISION

    path = hf_hub_download(SIMCOT_REPO, "model-00001-of-00001.safetensors", revision=revision)
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 24), b""):
            digest.update(chunk)
    assert digest.hexdigest() == SIMCOT_SHA256, f"unexpected checkpoint {digest.hexdigest()}"
    state = {k.removeprefix("decoder."): v.float() for k, v in load_file(path).items() if k.startswith("decoder.")}
    config = GPT2Config.from_pretrained("gpt2", revision=GPT2_REVISION)
    config.vocab_size = state["lm_head.weight"].shape[0]
    config.tie_word_embeddings = False
    config._attn_implementation = "eager"
    decoder = GPT2LMHeadModel(config)
    missing, unexpected = decoder.load_state_dict(state, strict=False)
    if unexpected or [k for k in missing if not k.endswith((".attn.bias", ".attn.masked_bias"))]:
        raise ValueError(f"step decoder does not match GPT-2: missing={missing} unexpected={unexpected}")
    decoder.to(device).eval()
    for p in decoder.parameters():
        p.requires_grad_(False)
    return decoder


def step_targets(tokenizer, steps, slots: int) -> list[list[int] | None]:
    """SIM-CoT's decoder targets for one problem: each `<<...>>` step of the joined rationale plus end-of-text, in `slots`
    slots; steps past the last slot merge into it, and slots past the last step get None (no loss)."""
    text = " ".join(steps) if isinstance(steps, (list, tuple)) else (steps or "")
    ids, found, i = tokenizer.encode(text), [], 0
    while i < len(ids):
        if ids[i] in STEP_STARTS and STEP_END in ids[i + 1:]:
            end = ids.index(STEP_END, i + 1)
            found.append(ids[i:end + 1])
            i = end + 1
        else:
            i += 1
    if len(found) > slots:
        found = found[:slots - 1] + [sum(found[slots - 1:], [])]
    eos = tokenizer.eos_token_id
    return [s + [eos] for s in found] + [None] * (slots - len(found))


def step_loss(decoder, embed, latents, targets, pad: int):
    """Mean cross-entropy of the decoder reading each latent back into its step (teacher forcing).

    latents: [batch, slots, width]; targets: per row, per slot, a token list or None. Returns None if nothing to score."""
    rows = [(b, s, t) for b, row in enumerate(targets) for s, t in enumerate(row) if t is not None]
    if not rows:
        return None
    length = max(len(t) for _, _, t in rows)
    tokens = torch.tensor([t + [pad] * (length - len(t)) for _, _, t in rows], device=latents.device)
    known = torch.tensor([[1] * len(t) + [0] * (length - len(t)) for _, _, t in rows], device=latents.device)
    first = latents[[b for b, _, _ in rows], [s for _, s, _ in rows]][:, None, :]
    inputs = torch.cat([first, embed(tokens[:, :-1])], 1)
    mask = torch.cat([torch.ones_like(known[:, :1]), known[:, :-1]], 1)
    logits = decoder(inputs_embeds=inputs, attention_mask=mask).logits.float()
    labels = tokens.masked_fill(known == 0, -100)  # the pad id may lie outside the decoder's vocabulary
    return nn.functional.cross_entropy(logits.reshape(-1, logits.shape[-1]), labels.reshape(-1), ignore_index=-100)


@torch.no_grad()
def decode_steps(decoder, embed, latents, eos: int, max_tokens: int = 32) -> list[list[int]]:
    """Greedy decodes of latents [n, width], each read on its own as SIM-CoT's decoder was trained."""
    step, cache, done = latents[:, None, :], None, torch.zeros(len(latents), dtype=torch.bool, device=latents.device)
    out = [[] for _ in range(len(latents))]
    for _ in range(max_tokens):
        result = decoder(inputs_embeds=step, past_key_values=cache, use_cache=True)
        cache, token = result.past_key_values, result.logits[:, -1].argmax(-1)
        for i in (~done).nonzero().flatten().tolist():
            out[i].append(int(token[i]))
        done |= token == eos
        if bool(done.all()):
            break
        step = embed(token[:, None])
    return [[t for t in seq if t != eos] for seq in out]


_OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv}


def arithmetic(expression: str):
    """The value of a +-*/ expression, or None."""
    try:
        tree = ast.parse(expression.replace(",", ""), mode="eval")
    except SyntaxError:
        return None

    def value(node):
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return float(node.value)
        if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
            return _OPS[type(node.op)](value(node.left), value(node.right))
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
            return -value(node.operand)
        raise ValueError(expression)

    try:
        return value(tree.body)
    except (ValueError, ZeroDivisionError, OverflowError):
        return None


STEP = re.compile(r"<<([^<>=]*)=([^<>=]*)>>")


def parse_step(text: str):
    """(expression, result) of the first `<<a op b=c>>` in a decode, or None."""
    match = STEP.search(text)
    return (match.group(1).strip(), match.group(2).strip()) if match else None
