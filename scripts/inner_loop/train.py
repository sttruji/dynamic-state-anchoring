"""Stage 1 of idea A: fine-tune SIM-CoT GPT-2 with an inner loop over layers 4-7 in each latent step, against controls.

Every arm starts from the same checkpoint, sees the same examples in the same order with the same seed, and trains on the
same objective: answer cross-entropy plus SIM-CoT's step loss, the frozen step decoder reading each latent back into its
`<<a op b=c>>` step (docs/brainstorm/INNER_LOOP_QUICK_REFERENCE.md).

    ft6       6 latent steps, no loop: the control for the fine-tune and the new loss
    ft12      12 latent steps, no loop: the more-steps arm, matched to the loop at k=4 in layer evaluations
    loop      6 steps, the block looped k times, k drawn per batch (1 with --p-k1, else uniform on 2..--k-max)
    loop_k1   6 steps, the loop's LoRA on pass 1 and no loop: rules out "more parameters"

    python scripts/inner_loop/train.py --arm loop --checkpoint <simcot dir> --output <dir> --device cuda --bf16
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

ARMS = ("ft6", "ft12", "loop", "loop_k1")
ROOT = Path(__file__).resolve().parents[2]
parser = argparse.ArgumentParser()
parser.add_argument("--arm", required=True, choices=ARMS)
parser.add_argument("--checkpoint", required=True, help="a directory holding pytorch_model.bin (fetch_simcot.py)")
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--train-file", type=Path,
                    default=Path(os.environ.get("DSA_DATA_DIR", ROOT / "data")) / "gsm_train.json")
parser.add_argument("--records", type=int, default=98_304)
parser.add_argument("--batch", type=int, default=32)
parser.add_argument("--lr", type=float, default=2e-5)
parser.add_argument("--loop-lr", type=float, default=1e-3, help="for the new loop parameters (untested)")
parser.add_argument("--warmup", type=float, default=0.03)
parser.add_argument("--step-weight", type=float, default=1.0, help="weight of the step-decoder loss")
parser.add_argument("--k-max", type=int, default=4)
parser.add_argument("--p-k1", type=float, default=0.25)
parser.add_argument("--first", type=int, default=4)
parser.add_argument("--last", type=int, default=8)
parser.add_argument("--rank", type=int, default=32)
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
from looped import LoopParams, latent_step, load_step_decoder, save_loop, step_loss, step_targets

random.seed(args.seed)
torch.manual_seed(args.seed)
codi = CODIGPT2(checkpoint=args.checkpoint, device=args.device)
model, projection, tokenizer, device = codi.model, codi.projection, codi.tokenizer, codi.device
model.train()
projection.train()
embed = model.get_input_embeddings()
decoder = load_step_decoder(device)
depth = 12 if args.arm == "ft12" else 6
loop = None
if args.arm.startswith("loop"):
    loop = LoopParams(model, first=args.first, last=args.last, rank=args.rank, lora_on_first=args.arm == "loop_k1")

# The same seeded sample and order for every arm.
source = json.loads(args.train_file.read_text())
examples = []
for i in random.Random(args.seed).sample(range(len(source)), len(source)):
    row = source[i]
    question = tokenizer.encode(row["question"].strip().replace("  ", " ")) + [codi.bot_id]
    if len(question) > args.max_question_tokens or not str(row["answer"]).strip():
        continue
    answer = [codi.eot_id] + tokenizer.encode(f"The answer is: {str(row['answer']).strip()}") + [tokenizer.eos_token_id]
    examples.append((question, answer, step_targets(tokenizer, row.get("steps", []), depth + 1)))
    if len(examples) == args.records:
        break
del source
updates = len(examples) // args.batch
k_rng = random.Random(args.seed + 1)
ks = [(1 if k_rng.random() < args.p_k1 else k_rng.randint(2, args.k_max)) if args.arm == "loop" else 1
      for _ in range(updates)]

groups = [{"params": list(model.parameters()) + list(projection.parameters()), "lr": args.lr}]
if loop is not None:
    groups.append({"params": list(loop.parameters()), "lr": args.loop_lr, "weight_decay": 0.0})
optimizer = torch.optim.AdamW(groups, weight_decay=0.01)
warm = max(1, int(args.warmup * updates))
schedule = torch.optim.lr_scheduler.LambdaLR(
    optimizer, lambda u: min((u + 1) / warm, max(0.0, (updates - u) / max(1, updates - warm))))


def train_step(batch, k):
    width = len(batch)
    q_len = max(len(q) for q, _, _ in batch)
    a_len = max(len(a) for _, a, _ in batch)
    pad = tokenizer.eos_token_id
    q_ids = torch.tensor([[pad] * (q_len - len(q)) + q for q, _, _ in batch], device=device)
    q_mask = torch.tensor([[0] * (q_len - len(q)) + [1] * len(q) for q, _, _ in batch], device=device)
    a_ids = torch.tensor([a + [pad] * (a_len - len(a)) for _, a, _ in batch], device=device)
    a_mask = torch.tensor([[1] * len(a) + [0] * (a_len - len(a)) for _, a, _ in batch], device=device)
    positions = (q_mask.cumsum(1) - 1).clamp(min=0)

    out = model(input_ids=q_ids, attention_mask=q_mask, position_ids=positions, use_cache=True, output_hidden_states=True)
    cache, latent = out.past_key_values, projection(out.hidden_states[-1][:, -1:, :])
    start, mask, latents = positions[:, -1] + 1, q_mask, [latent]
    for step in range(1, depth + 1):
        mask = torch.cat([mask, torch.ones(width, 1, dtype=mask.dtype, device=device)], 1)
        hidden, cache = latent_step(model, loop, latent, cache, (start + step - 1)[:, None], k=k, mask=mask)
        latent = projection(hidden)
        latents.append(latent)
    mask = torch.cat([mask, a_mask], 1)
    out = model(inputs_embeds=embed(a_ids), past_key_values=cache, attention_mask=mask,
                position_ids=start[:, None] + depth + torch.arange(a_len, device=device)[None], use_cache=False)
    target, keep = a_ids[:, 1:], a_mask[:, 1:].float()
    ce = F.cross_entropy(out.logits[:, :-1].float().reshape(-1, out.logits.shape[-1]), target.reshape(-1),
                         reduction="none").view(width, -1)
    ce = (ce * keep).sum() / keep.sum()
    steps = step_loss(decoder, embed, torch.cat(latents, 1), [t for _, _, t in batch], pad=pad)
    loss = ce if steps is None else ce + args.step_weight * steps
    stats = {"ce": float(ce), "step": float(steps) if steps is not None else None}
    if loop is not None:
        stats.update(gate=float(loop.gate.abs().mean()), inject=float(loop.inject.abs().mean()))
    return loss, stats


def save_model(folder: Path) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    state = {f"codi.base_model.model.{k}": v.detach().cpu() for k, v in model.state_dict().items()
             if not k.endswith((".attn.bias", ".attn.masked_bias"))}
    state.update({f"prj.{k}": v.detach().cpu() for k, v in projection.state_dict().items()})
    torch.save(state, folder / "pytorch_model.bin")
    if loop is not None:
        save_loop(loop, folder / "loop.pt")


args.output.mkdir(parents=True, exist_ok=True)
resume = args.output / "resume.pt"
done = 0
if resume.exists():
    saved = torch.load(resume, map_location="cpu", weights_only=False)
    model.load_state_dict(saved["model"])
    projection.load_state_dict(saved["projection"])
    if loop is not None:
        loop.load_state_dict(saved["loop"])
    optimizer.load_state_dict(saved["optimizer"])
    schedule.load_state_dict(saved["schedule"])
    done = saved["done"]
    print(f"resuming at update {done}/{updates}", flush=True)
(args.output / "args.json").write_text(json.dumps(vars(args) | {"updates": updates, "examples": len(examples), "depth": depth},
                                                  default=str, indent=1))
start_time = time.time()
with (args.output / "metrics.jsonl").open("a") as log:
    for update in range(done, updates):
        batch = examples[update * args.batch:(update + 1) * args.batch]
        with torch.autocast("cuda", dtype=torch.bfloat16) if args.bf16 else contextlib.nullcontext():
            loss, stats = train_step(batch, ks[update])
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_([p for g in groups for p in g["params"]], 2.0)
        optimizer.step()
        schedule.step()
        if update % args.log_every == 0 or update == updates - 1:
            line = {"update": update, "k": ks[update], "loss": round(float(loss), 5), "grad_norm": round(float(norm), 3),
                    "lr": schedule.get_last_lr()[0], "seconds": round(time.time() - start_time, 1)}
            line |= {k: (round(v, 5) if v is not None else None) for k, v in stats.items()}
            log.write(json.dumps(line) + "\n")
            log.flush()
            print(json.dumps(line), flush=True)
        if (update + 1) % args.save_every == 0 and update + 1 < updates:
            torch.save({"model": model.state_dict(), "projection": projection.state_dict(),
                        "loop": loop.state_dict() if loop is not None else None,
                        "optimizer": optimizer.state_dict(), "schedule": schedule.state_dict(), "done": update + 1},
                       resume.with_suffix(".tmp"))
            resume.with_suffix(".tmp").replace(resume)
save_model(args.output / "model")
resume.unlink(missing_ok=True)
print(f"{args.arm}: {updates} updates on {len(examples)} examples in {time.time() - start_time:.0f}s -> {args.output / 'model'}",
      flush=True)
