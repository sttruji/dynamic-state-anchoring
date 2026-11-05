"""Part C of results/latent_gate/heldout_scale/PLAN.md (Amendment 1): fine-tune a CODI-style LLaMA to run more latent
steps, with or without the fixed hard gate. It is Stage 1's recipe (stage1_train.py), ported to LLaMA:

    ft6           6 latent steps, full cache: the control for the extra fine-tuning
    ft12          12 steps, full cache
    gate12_fixed  12 steps; each query sees only the `budget` highest-scoring latent entries before it, with log g on
                  them. The gate runs at FP32 outside autocast and a tie is kept for the older entry, as at inference.

A full fine-tune: FP32 weights, BF16 autocast, fused AdamW. A 1B model does not take Stage 1's batch of 32 at once, so
each update accumulates micro-batches whose losses are normalised by the whole batch, giving the batch-32 gradient. The
latent bias rides on a 4D additive attention mask, as in batched.py; for the teacher-forced answer the same mask also
carries the causal and padding masks. LLaMA needs use_cache=True to take a tuple cache, so the answer pass returns one.

    python scripts/latent_gate/partc_train.py --llama simcot-llama1b --arm ft12 --output <dir> --device cuda --bf16
    python scripts/latent_gate/partc_train.py ... --smoke    # two updates, peak memory, nothing saved
"""
from __future__ import annotations

import argparse
import contextlib
import json
import os
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch
import torch.nn.functional as F

from batched import forward
from gate import HIDE, Gate, decode_labels, gold_results, top_visible

ARMS = ("ft6", "ft12", "gate12_fixed")


def key_bias(arm: str, budget: int, question: int, step: int, scores: list) -> torch.Tensor | None:
    """[batch, question + step]: the bias on the keys a query at latent step `step` sees (depth + 1 for the answer)."""
    if arm != "gate12_fixed" or not scores:
        return None
    s = torch.stack(scores[:step - 1], 1)  # [batch, entries]
    logs = F.logsigmoid(s.float())
    if s.shape[1] > budget:  # scores are fixed per entry, so a hidden entry never returns
        logs = torch.where(top_visible(s, budget), logs, torch.full_like(logs, HIDE))
    zeros = torch.zeros(s.shape[0], question, device=s.device)
    return torch.cat([zeros, logs, torch.zeros(s.shape[0], 1, device=s.device)], 1)


def answer_mask(mask: torch.Tensor, bias: torch.Tensor | None, answer: int) -> torch.Tensor:
    """[batch, 1, answer, keys]: the additive mask for `answer` new tokens at the end of `mask` [batch, keys], with
    padding, the causal mask and `bias` [batch, keys - answer] on the earlier keys."""
    keys = mask.shape[1]
    causal = torch.ones(answer, keys, dtype=torch.bool, device=mask.device).tril(keys - answer)
    out = torch.where(causal[None] & mask[:, None, :].bool(), 0.0, torch.finfo(torch.float32).min)
    if bias is not None:
        out = out + F.pad(bias, (0, answer))[:, None, :]
    return out[:, None]


def train_step(loop, gate, batch, depth: int, arm: str, budget: int, rows: int, tokens: int,
               budget_weight: float = 0.1, label_spread: int = 1, autocast=contextlib.nullcontext):
    """One micro-batch of (question, answer, gold results). Its loss is its share of the whole batch's: CE is summed over
    answer tokens and divided by `tokens`, the budget penalty summed over rows and divided by `rows`."""
    model = loop.model
    device = next(model.parameters()).device
    width, gated = len(batch), arm.startswith("gate")
    q_len, a_len = max(len(q) for q, _, _ in batch), max(len(a) for _, a, _ in batch)
    q_ids = torch.tensor([[loop.pad_id] * (q_len - len(q)) + q for q, _, _ in batch], device=device)
    q_mask = torch.tensor([[0] * (q_len - len(q)) + [1] * len(q) for q, _, _ in batch], device=device)
    a_ids = torch.tensor([a + [loop.pad_id] * (a_len - len(a)) for _, a, _ in batch], device=device)
    a_mask = torch.tensor([[1] * len(a) + [0] * (a_len - len(a)) for _, a, _ in batch], device=device)
    results = [r for _, _, r in batch]
    positions = (q_mask.cumsum(1) - 1).clamp(min=0)
    with autocast():
        out = model(input_ids=q_ids, attention_mask=q_mask, position_ids=positions, use_cache=True,
                    output_hidden_states=True, num_logits_to_keep=1)
        cache, latent = out.past_key_values, loop.projection(out.hidden_states[-1][:, -1:, :])
        start, mask, scores, labels = positions[:, -1] + 1, q_mask, [], []
        for step in range(1, depth + 1):
            mask = torch.cat([mask, torch.ones(width, 1, dtype=mask.dtype, device=device)], 1)
            out = forward(loop, embeds=latent, cache=cache, mask=mask, positions=(start + step - 1)[:, None],
                          bias=key_bias(arm, budget, q_len, step, scores), hidden=True)
            cache, hidden = out.past_key_values, out.hidden_states[-1][:, -1, :]
            if gated:
                with torch.autocast(device.type, enabled=False):
                    scores.append(gate(hidden.detach().float()))
                if loop.tokenizer is not None:
                    vocab = len(loop.tokenizer)
                    labels.append(decode_labels(loop.tokenizer, out.logits[:, -1, :vocab].detach(), results))
            latent = loop.projection(hidden[:, None, :])
        mask = torch.cat([mask, a_mask], 1)
        bias = key_bias(arm, budget, q_len, depth + 1, scores)
        attention = answer_mask(mask, None if bias is None else bias[:, :-1], a_len)
        out = model(inputs_embeds=model.get_input_embeddings()(a_ids), past_key_values=cache, attention_mask=attention,
                    position_ids=start[:, None] + depth + torch.arange(a_len, device=device)[None], use_cache=True)
        target, keep = a_ids[:, 1:], a_mask[:, 1:].float()
        ce = F.cross_entropy(out.logits[:, :-1].float().reshape(-1, out.logits.shape[-1]), target.reshape(-1),
                             reduction="none").view(width, -1)
        ce = (ce * keep).sum()
        loss = ce / tokens
        stats = {"ce": float(ce) / tokens}
        if gated:
            g = torch.sigmoid(torch.stack(scores, 1))
            over = F.relu(g.sum(1) - budget).sum()
            loss = loss + budget_weight * over / rows
            stats.update(gate_sum=float(g.sum()) / rows, over=float(over) / rows)
            if labels:
                y = torch.stack(labels, 1)
                for k in range(1, label_spread + 1):
                    y = torch.maximum(y, F.pad(y[:, k:], (0, k)))
                stats["label_rate"] = float(y.sum()) / (rows * y.shape[1])
    stats["loss"] = float(loss)
    return loss, stats, out.logits


def accumulate(loop, gate, batch, depth: int, arm: str, budget: int, micro: int, **options) -> dict:
    """Forward and backward over `batch` in micro-batches of `micro`; returns the whole batch's statistics."""
    rows, tokens = len(batch), sum(len(a) - 1 for _, a, _ in batch)
    totals = {}
    for first in range(0, rows, micro):
        loss, stats, _ = train_step(loop, gate, batch[first:first + micro], depth, arm, budget, rows, tokens, **options)
        loss.backward()
        for key, value in stats.items():
            totals[key] = totals.get(key, 0.0) + value
    return totals


def main() -> None:
    from batched import Loop
    from dsa.bench.models.codi_llama import CHECKPOINTS, CODILlama

    parser = argparse.ArgumentParser()
    parser.add_argument("--llama", choices=sorted(CHECKPOINTS), required=True)
    parser.add_argument("--arm", required=True, choices=ARMS)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--train-file", type=Path,
                        default=Path(os.environ.get("DSA_DATA_DIR", ROOT / "data")) / "gsm_train.json")
    parser.add_argument("--records", type=int, default=32_768)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--micro-batch", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--gate-lr", type=float, default=1e-3)
    parser.add_argument("--warmup", type=float, default=0.03)
    parser.add_argument("--budget", type=int, default=6)
    parser.add_argument("--budget-weight", type=float, default=0.1)
    parser.add_argument("--label-spread", type=int, default=1)
    parser.add_argument("--max-question-tokens", type=int, default=256)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--save-every", type=int, default=256)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--smoke", action="store_true", help="two updates, then report peak memory and time; no files")
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    codi = CODILlama(checkpoint=args.llama, device=args.device)
    gate = Gate(codi.model.config.hidden_size).to(codi.device)
    loop = Loop.of(codi, gate)
    model, projection, tokenizer = codi.model, codi.projection, codi.tokenizer
    model.train()
    projection.train()
    gated = args.arm.startswith("gate")

    # The same seeded sample and order for every arm, as stage1_train (lengths counted in LLaMA tokens).
    source = json.loads(args.train_file.read_text())
    examples = []
    for i in random.Random(args.seed).sample(range(len(source)), len(source)):
        row = source[i]
        question = loop.encode(row["question"])
        if len(question) > args.max_question_tokens or not str(row["answer"]).strip():
            continue
        answer = ([codi.eot_id] + tokenizer.encode(f"The answer is: {str(row['answer']).strip()}", add_special_tokens=False)
                  + [tokenizer.eos_token_id])
        examples.append((question, answer, gold_results(row.get("steps", []))))
        if len(examples) == args.records:
            break
    del source
    updates = len(examples) // args.batch
    depth = 6 if args.arm == "ft6" else 12

    groups = [{"params": list(model.parameters()) + list(projection.parameters()), "lr": args.lr}]
    if gated:
        groups.append({"params": list(gate.parameters()), "lr": args.gate_lr, "weight_decay": 0.0})
    optimizer = torch.optim.AdamW(groups, weight_decay=0.01, fused=codi.device.type == "cuda")
    warm = max(1, int(args.warmup * updates))
    schedule = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda u: min((u + 1) / warm, max(0.0, (updates - u) / max(1, updates - warm))))
    autocast = (lambda: torch.autocast("cuda", dtype=torch.bfloat16)) if args.bf16 else contextlib.nullcontext
    options = {"budget_weight": args.budget_weight, "label_spread": args.label_spread, "autocast": autocast}
    params = [p for g in groups for p in g["params"]]

    def one_update(update):
        stats = accumulate(loop, gate, examples[update * args.batch:(update + 1) * args.batch], depth, args.arm,
                           args.budget, args.micro_batch, **options)
        norm = torch.nn.utils.clip_grad_norm_(params, 2.0)
        optimizer.step()
        schedule.step()
        optimizer.zero_grad(set_to_none=True)
        return stats, norm

    if args.smoke:
        torch.cuda.reset_peak_memory_stats()
        began = time.time()
        for update in range(2):
            stats, _ = one_update(update)
        peak = torch.cuda.max_memory_allocated() / 2 ** 30
        total = torch.cuda.get_device_properties(0).total_memory / 2 ** 30
        print(f"smoke {args.arm}, depth {depth}, micro-batch {args.micro_batch}: {stats} — peak {peak:.1f} of {total:.0f} GiB, "
              f"{(time.time() - began) / 2:.1f}s per update, about {(time.time() - began) / 2 * updates / 60:.0f} min for "
              f"{updates} updates", flush=True)
        return

    args.output.mkdir(parents=True, exist_ok=True)
    resume = args.output / "resume.pt"
    done = 0
    if resume.exists():
        saved = torch.load(resume, map_location="cpu", weights_only=False)
        model.load_state_dict(saved["model"])
        projection.load_state_dict(saved["projection"])
        gate.load_state_dict(saved["gate"])
        optimizer.load_state_dict(saved["optimizer"])
        schedule.load_state_dict(saved["schedule"])
        done = saved["done"]
        del saved
        print(f"resuming at update {done}/{updates}", flush=True)
    (args.output / "args.json").write_text(json.dumps(vars(args) | {"updates": updates, "examples": len(examples),
                                                                    "depth": depth}, default=str, indent=1))
    began = time.time()
    with (args.output / "metrics.jsonl").open("a") as log:
        for update in range(done, updates):
            stats, norm = one_update(update)
            if update % args.log_every == 0 or update == updates - 1:
                line = {"update": update, "depth": depth, "grad_norm": round(float(norm), 3), "lr": schedule.get_last_lr()[0],
                        "seconds": round(time.time() - began, 1), "peak_gib": round(torch.cuda.max_memory_allocated() / 2 ** 30, 1)
                        if torch.cuda.is_available() else None} | {k: round(v, 5) for k, v in stats.items()}
                log.write(json.dumps(line) + "\n")
                log.flush()
                print(json.dumps(line), flush=True)
            if (update + 1) % args.save_every == 0 and update + 1 < updates:
                torch.save({"model": model.state_dict(), "projection": projection.state_dict(), "gate": gate.state_dict(),
                            "optimizer": optimizer.state_dict(), "schedule": schedule.state_dict(), "done": update + 1},
                           resume.with_suffix(".tmp"))
                resume.with_suffix(".tmp").replace(resume)
    folder = args.output / "model"
    folder.mkdir(parents=True, exist_ok=True)
    state = {f"codi.base_model.model.{k}": v.detach().float().cpu() for k, v in model.state_dict().items()
             if k != "lm_head.weight"}  # tied to the embeddings
    state.update({f"prj.{k}": v.detach().float().cpu() for k, v in projection.state_dict().items()})
    torch.save(state, folder / "pytorch_model.bin")
    if gated:
        torch.save({"state": gate.state_dict(), "width": model.config.hidden_size, "budget": args.budget}, folder / "gate.pt")
    resume.unlink(missing_ok=True)
    print(f"{args.arm}: {updates} updates on {len(examples)} examples in {time.time() - began:.0f}s -> {folder}", flush=True)


if __name__ == "__main__":
    main()
