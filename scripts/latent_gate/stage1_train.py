"""Stage 1: fine-tune a CODI-style GPT-2 to run more latent steps, with or without a gate over its latent entries.

Plan: results/latent_gate/stage1/PLAN.md. Every arm starts from the same checkpoint and sees the same examples in the same
order; only the latent depth and what each query may attend to differ.

    ft6           6 latent steps, full cache: a control for the extra fine-tuning
    ft6x2         6 steps on twice the examples: roughly the compute of a 12-step arm
    ft12          12 steps, full cache
    ftrand        a depth from 6 to 12 drawn per batch, full cache (Kuo et al.)
    window12      12 steps; every query sees only the 6 latent entries before it
    rand12        12 steps; above 6 entries, a random one is hidden for good at each step (training under eviction,
                  with no learned choice)
    gate12        12 steps; log g_j on every key of latent entry j, with a budget penalty on the sum of g
    gate12_hard   gate12, but each query sees only the `budget` highest-scoring latent entries before it (log g on
                  those), so training matches hard eviction at inference; same loss as gate12
    gate12_label  gate12 plus a loss pulling g_j towards the entry's intermediate-result label
    gate12_fixed  gate12_hard with its tie bug fixed (results/latent_gate/heldout_scale/PLAN.md, E3): gate12_hard keeps
                  every entry tied with the last kept score, and its BF16 scores tie often; here the gate runs at FP32
                  outside autocast and exactly `budget` entries stay visible, a tie kept for the older one, as at inference

    python scripts/latent_gate/stage1_train.py --arm gate12_label --checkpoint <dir> --output <dir> --device cuda --bf16
"""
from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
import random
import sys
import time
from pathlib import Path

ARMS = ("ft6", "ft6x2", "ft12", "ftrand", "window12", "rand12", "gate12", "gate12_hard", "gate12_label", "gate12_fixed")
ROOT = Path(__file__).resolve().parents[2]
parser = argparse.ArgumentParser()
parser.add_argument("--arm", required=True, choices=ARMS)
parser.add_argument("--checkpoint", help="a directory holding pytorch_model.bin; the published CODI weights if omitted")
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--train-file", type=Path,
                    default=Path(os.environ.get("DSA_DATA_DIR", ROOT / "data")) / "gsm_train.json")
parser.add_argument("--records", type=int, default=32_768)
parser.add_argument("--batch", type=int, default=32)
parser.add_argument("--lr", type=float, default=2e-5)
parser.add_argument("--gate-lr", type=float, default=1e-3)
parser.add_argument("--warmup", type=float, default=0.03)
parser.add_argument("--budget", type=int, default=6)
parser.add_argument("--budget-weight", type=float, default=0.1)
parser.add_argument("--label-weight", type=float, default=1.0)
parser.add_argument("--label-spread", type=int, default=1,
                    help="an entry is also labelled 1 when one of the next k entries decodes to a gold result; with 1, "
                         "CODI's operator-phase entry before each result is kept with it")
parser.add_argument("--gate-train", choices=("soft", "hard"), default="soft",
                    help="hard: during training too, entries outside the top `budget` scores are hidden (gate12_hard sets it)")
parser.add_argument("--max-question-tokens", type=int, default=256)
parser.add_argument("--seed", type=int, default=7)
parser.add_argument("--device", default="cuda")
parser.add_argument("--bf16", action="store_true")
parser.add_argument("--save-every", type=int, default=200)
parser.add_argument("--log-every", type=int, default=10)
args = parser.parse_args()
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch
import torch.nn.functional as F
from dsa.bench.models.codi import CODIGPT2
from gate import HIDE, Gate, decode_labels, gold_results, install_bias, set_bias, top_visible

gated = args.arm.startswith("gate")
fixed = args.arm == "gate12_fixed"
hard_gate = args.gate_train == "hard" or args.arm in ("gate12_hard", "gate12_fixed")
random.seed(args.seed)
torch.manual_seed(args.seed)
codi = CODIGPT2(checkpoint=args.checkpoint, device=args.device)
model, projection, tokenizer, device = codi.model, codi.projection, codi.tokenizer, codi.device
install_bias(model)
model.train()
projection.train()
embed = model.get_input_embeddings()
gate = Gate(model.config.n_embd).to(device)

# The same seeded sample and order for every arm.
source = json.loads(args.train_file.read_text())
examples = []
for i in random.Random(args.seed).sample(range(len(source)), len(source)):
    row = source[i]
    question = tokenizer.encode(row["question"].strip().replace("  ", " ")) + [codi.bot_id]
    if len(question) > args.max_question_tokens or not str(row["answer"]).strip():
        continue
    answer = [codi.eot_id] + tokenizer.encode(f"The answer is: {str(row['answer']).strip()}") + [tokenizer.eos_token_id]
    examples.append((question, answer, gold_results(row.get("steps", []))))
    if len(examples) == args.records * (2 if args.arm == "ft6x2" else 1):
        break
del source  # the full split is about 385k problems; only the sample is kept
updates = len(examples) // args.batch
depth_rng = random.Random(args.seed + 1)
depths = [6 if args.arm.startswith("ft6") else depth_rng.randint(6, 12) if args.arm == "ftrand" else 12
          for _ in range(updates)]

groups = [{"params": list(model.parameters()) + list(projection.parameters()), "lr": args.lr}]
if gated:
    groups.append({"params": list(gate.parameters()), "lr": args.gate_lr, "weight_decay": 0.0})
optimizer = torch.optim.AdamW(groups, weight_decay=0.01)
warm = max(1, int(args.warmup * updates))
schedule = torch.optim.lr_scheduler.LambdaLR(
    optimizer, lambda u: min((u + 1) / warm, max(0.0, (updates - u) / max(1, updates - warm))))


def key_bias(keys: int, question: int, step: int, scores: list, hidden_mask: torch.Tensor) -> torch.Tensor | None:
    """Bias on keys for queries at latent step `step` (depth + 1 for the readout); latent entry j sits at question + j - 1."""
    if not gated and args.arm not in ("window12", "rand12"):
        return None
    width = scores[0].shape[0] if scores else hidden_mask.shape[0]
    bias = torch.zeros(width, keys, device=device)
    seen = range(1, step)  # entries before this query; the query's own key is never biased
    if args.arm == "window12":
        for j in seen:
            if j < step - args.budget:
                bias[:, question + j - 1] = HIDE
        return bias[:, None, None, :]
    if args.arm == "rand12":
        bias[:, question:question + step - 1] = torch.where(hidden_mask[:, 1:step], HIDE, 0.0)
        return bias[:, None, None, :]
    if not scores:
        return None
    s = torch.stack(scores[:step - 1], 1)  # [batch, entries]
    logs = F.logsigmoid(s.float())
    if hard_gate and s.shape[1] > args.budget:  # scores are fixed per entry, so a hidden entry never returns
        if fixed:
            logs = torch.where(top_visible(s, args.budget), logs, torch.full_like(logs, HIDE))
        else:  # as run in Stage 1: every entry tied with the last kept score stays visible
            cut = s.topk(args.budget, dim=1).values[:, -1:]
            logs = torch.where(s >= cut, logs, torch.full_like(logs, HIDE))
    bias = torch.cat([bias[:, :question], logs, bias[:, question + s.shape[1]:]], 1)
    return bias[:, None, None, :]


def hide_one(hidden_mask: torch.Tensor, step: int, generator: torch.Generator) -> None:
    """rand12: once entries 1..step-1 hold more than `budget` visible ones, hide one at random in each row, for good."""
    visible = ~hidden_mask[:, 1:step]
    for row in range(hidden_mask.shape[0]):
        options = visible[row].nonzero().flatten()
        if len(options) > args.budget:
            hidden_mask[row, 1 + int(options[int(torch.randint(len(options), (1,), generator=generator))])] = True


def train_step(batch, depth, update):
    width = len(batch)
    hidden_mask = torch.zeros(width, depth + 2, dtype=torch.bool, device=device)  # column j: entry j hidden (rand12)
    generator = torch.Generator().manual_seed(args.seed * 100_003 + update)
    q_len = max(len(q) for q, _, _ in batch)
    a_len = max(len(a) for _, a, _ in batch)
    pad = tokenizer.eos_token_id
    q_ids = torch.tensor([[pad] * (q_len - len(q)) + q for q, _, _ in batch], device=device)
    q_mask = torch.tensor([[0] * (q_len - len(q)) + [1] * len(q) for q, _, _ in batch], device=device)
    a_ids = torch.tensor([a + [pad] * (a_len - len(a)) for _, a, _ in batch], device=device)
    a_mask = torch.tensor([[1] * len(a) + [0] * (a_len - len(a)) for _, a, _ in batch], device=device)
    results = [r for _, _, r in batch]
    positions = (q_mask.cumsum(1) - 1).clamp(min=0)

    set_bias(model, None)
    out = model(input_ids=q_ids, attention_mask=q_mask, position_ids=positions, use_cache=True, output_hidden_states=True)
    cache, latent = out.past_key_values, projection(out.hidden_states[-1][:, -1:, :])
    start, mask, scores, labels = positions[:, -1] + 1, q_mask, [], []
    for step in range(1, depth + 1):
        mask = torch.cat([mask, torch.ones(width, 1, dtype=mask.dtype, device=device)], 1)
        if args.arm == "rand12":
            hide_one(hidden_mask, step, generator)
        set_bias(model, key_bias(mask.shape[1], q_len, step, scores, hidden_mask))
        out = model(inputs_embeds=latent, past_key_values=cache, attention_mask=mask,
                    position_ids=(start + step - 1)[:, None], use_cache=True, output_hidden_states=True)
        cache, hidden = out.past_key_values, out.hidden_states[-1][:, -1, :]
        if gated:
            with torch.autocast("cuda", enabled=False) if fixed else contextlib.nullcontext():
                scores.append(gate(hidden.detach().float()))
            labels.append(decode_labels(tokenizer, out.logits[:, -1, :].detach(), results))
        latent = projection(hidden[:, None, :])
    mask = torch.cat([mask, a_mask], 1)
    if args.arm == "rand12":
        hide_one(hidden_mask, depth + 1, generator)
    set_bias(model, key_bias(mask.shape[1], q_len, depth + 1, scores, hidden_mask))
    out = model(inputs_embeds=embed(a_ids), past_key_values=cache, attention_mask=mask,
                position_ids=start[:, None] + depth + torch.arange(a_len, device=device)[None], use_cache=False)
    set_bias(model, None)
    target, keep = a_ids[:, 1:], a_mask[:, 1:].float()
    ce = F.cross_entropy(out.logits[:, :-1].float().reshape(-1, out.logits.shape[-1]), target.reshape(-1),
                         reduction="none").view(width, -1)
    ce = (ce * keep).sum() / keep.sum()
    stats = {"ce": float(ce)}
    loss = ce
    if gated:
        s, y = torch.stack(scores, 1), torch.stack(labels, 1)
        for k in range(1, args.label_spread + 1):
            y = torch.maximum(y, F.pad(y[:, k:], (0, k)))
        g = torch.sigmoid(s)
        over = F.relu(g.sum(1) - args.budget).mean()
        loss = loss + args.budget_weight * over
        stats.update(gate_sum=float(g.sum(1).mean()), over=float(over), label_rate=float(y.mean()))
        if args.arm == "gate12_label":
            bce = F.binary_cross_entropy_with_logits(s, y)
            loss = loss + args.label_weight * bce
            stats["bce"] = float(bce)
    return loss, stats


def save_model(folder: Path) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    state = {f"codi.base_model.model.{k}": v.detach().cpu() for k, v in model.state_dict().items()
             if not k.endswith((".attn.bias", ".attn.masked_bias"))}
    state.update({f"prj.{k}": v.detach().cpu() for k, v in projection.state_dict().items()})
    torch.save(state, folder / "pytorch_model.bin")
    if gated:
        torch.save({"state": gate.state_dict(), "width": model.config.n_embd, "budget": args.budget}, folder / "gate.pt")


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
    print(f"resuming at update {done}/{updates}", flush=True)
(args.output / "args.json").write_text(json.dumps(vars(args) | {"updates": updates, "examples": len(examples)}, default=str, indent=1))
start_time = time.time()
with (args.output / "metrics.jsonl").open("a") as log:
    for update in range(done, updates):
        batch = examples[update * args.batch:(update + 1) * args.batch]
        with torch.autocast("cuda", dtype=torch.bfloat16) if args.bf16 else contextlib.nullcontext():
            loss, stats = train_step(batch, depths[update], update)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_([p for g in groups for p in g["params"]], 2.0)
        optimizer.step()
        schedule.step()
        if update % args.log_every == 0 or update == updates - 1:
            line = {"update": update, "depth": depths[update], "loss": round(float(loss), 5), "grad_norm": round(float(norm), 3),
                    "lr": schedule.get_last_lr()[0], "seconds": round(time.time() - start_time, 1)} | {k: round(v, 5) for k, v in stats.items()}
            log.write(json.dumps(line) + "\n")
            log.flush()
            print(json.dumps(line), flush=True)
        if (update + 1) % args.save_every == 0 and update + 1 < updates:
            torch.save({"model": model.state_dict(), "projection": projection.state_dict(), "gate": gate.state_dict(),
                        "optimizer": optimizer.state_dict(), "schedule": schedule.state_dict(), "done": update + 1},
                       resume.with_suffix(".tmp"))
            resume.with_suffix(".tmp").replace(resume)
save_model(args.output / "model")
resume.unlink(missing_ok=True)
print(f"{args.arm}: {updates} updates on {len(examples)} examples in {time.time() - start_time:.0f}s -> {args.output / 'model'}", flush=True)
