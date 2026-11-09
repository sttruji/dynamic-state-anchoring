"""D1-R's fine-tune (results/ffn_sparsity/d1r/PLAN.md): Stage 1b's ft6_step recipe from scripts/inner_loop/train.py,
unchanged, on SIM-CoT GPT-2 with every FFN masked as the arm says (moe.py). The router, the hash table, the masks and the
stand-in values stay fixed; every weight of the model and the projection trains.

    dense               no mask. With --stop-after 51 --check <Stage 1b ft6_step metrics.jsonl>, the check that this code
                        path reproduces Stage 1b's losses
    moe / hash / prune  the converted FFN from --conversion

    python scripts/ffn_sparsity/train_moe.py --arm moe --conversion <conversion.pt> --checkpoint <simcot dir> \
        --output <dir> --seed 7 --device cuda --bf16
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

ROOT = Path(__file__).resolve().parents[2]
parser = argparse.ArgumentParser()
parser.add_argument("--arm", required=True, choices=("dense", "moe", "hash", "prune"))
parser.add_argument("--conversion", type=Path, help="conversion.pt from convert.py, for every arm but dense")
parser.add_argument("--checkpoint", required=True, help="a directory holding pytorch_model.bin (fetch_simcot.py)")
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--train-file", type=Path,
                    default=Path(os.environ.get("DSA_DATA_DIR", ROOT / "data")) / "gsm_train.json")
parser.add_argument("--records", type=int, default=98_304)
parser.add_argument("--batch", type=int, default=128)
parser.add_argument("--lr", type=float, help="default 2e-5 at batch 32, scaled by sqrt(batch / 32)")
parser.add_argument("--warmup", type=float, default=0.03)
parser.add_argument("--step-weight", type=float, default=1.0, help="weight of the step-decoder loss")
parser.add_argument("--max-question-tokens", type=int, default=256)
parser.add_argument("--seed", type=int, default=7)
parser.add_argument("--device", default="cuda")
parser.add_argument("--bf16", action="store_true")
parser.add_argument("--save-every", type=int, default=200)
parser.add_argument("--log-every", type=int, default=10)
parser.add_argument("--stop-after", type=int, help="stop after this many updates, without saving the model")
parser.add_argument("--check", type=Path, help="a metrics.jsonl whose logged losses this run must reproduce")
parser.add_argument("--tolerance", type=float, default=0.02)
args = parser.parse_args()
assert (args.arm == "dense") != (args.conversion is not None), "every arm but dense needs --conversion"
# The rate was set at batch 32 (Stage 1); Adam's square-root rule carries it to other batch sizes.
args.lr = 2e-5 * math.sqrt(args.batch / 32) if args.lr is None else args.lr
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts/inner_loop"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch
import torch.nn.functional as F
from dsa.bench.models.codi import CODIGPT2
from looped import latent_step, load_step_decoder, step_loss, step_targets
from moe import MoEMasks, step_key
from sparsity import phase_tags

# The same order of random draws as scripts/inner_loop/train.py, so the dense arm reproduces ft6_step.
random.seed(args.seed)
torch.manual_seed(args.seed)
codi = CODIGPT2(checkpoint=args.checkpoint, device=args.device)
model, projection, tokenizer, device = codi.model, codi.projection, codi.tokenizer, codi.device
model.train()
projection.train()
embed = model.get_input_embeddings()
decoder = load_step_decoder(device)
depth = 6
conversion = torch.load(args.conversion, map_location="cpu", weights_only=True) if args.conversion else None
masks = MoEMasks(model, conversion)
masks.arm = None if args.arm == "dense" else args.arm
vocab = codi.vocab_size

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

groups = [{"params": list(model.parameters()) + list(projection.parameters()), "lr": args.lr}]
optimizer = torch.optim.AdamW(groups, weight_decay=0.01)
warm = max(1, int(args.warmup * updates))
schedule = torch.optim.lr_scheduler.LambdaLR(
    optimizer, lambda u: min((u + 1) / warm, max(0.0, (updates - u) / max(1, updates - warm))))


def tag(phase, keys):
    masks.phase, masks.keys = phase, keys


def train_step(batch):
    width = len(batch)
    q_len = max(len(q) for q, _, _ in batch)
    a_len = max(len(a) for _, a, _ in batch)
    pad = tokenizer.eos_token_id
    q_ids = torch.tensor([[pad] * (q_len - len(q)) + q for q, _, _ in batch], device=device)
    q_mask = torch.tensor([[0] * (q_len - len(q)) + [1] * len(q) for q, _, _ in batch], device=device)
    a_ids = torch.tensor([a + [pad] * (a_len - len(a)) for _, a, _ in batch], device=device)
    a_mask = torch.tensor([[1] * len(a) + [0] * (a_len - len(a)) for _, a, _ in batch], device=device)
    positions = (q_mask.cumsum(1) - 1).clamp(min=0)

    tag(phase_tags(q_len - 1, 1, device=device),
        torch.cat([q_ids[:, :-1], torch.full((width, 1), step_key(vocab, 0), device=device)], 1))
    out = model(input_ids=q_ids, attention_mask=q_mask, position_ids=positions, use_cache=True, output_hidden_states=True)
    cache, latent = out.past_key_values, projection(out.hidden_states[-1][:, -1:, :])
    start, mask, latents = positions[:, -1] + 1, q_mask, [latent]
    for step in range(1, depth + 1):
        mask = torch.cat([mask, torch.ones(width, 1, dtype=mask.dtype, device=device)], 1)
        tag(phase_tags(latent=1, device=device), torch.full((width, 1), step_key(vocab, step), device=device))
        hidden, cache = latent_step(model, None, latent, cache, (start + step - 1)[:, None], k=1, mask=mask)
        latent = projection(hidden)
        latents.append(latent)
    mask = torch.cat([mask, a_mask], 1)
    tag(phase_tags(answer=a_len, device=device), a_ids)
    out = model(inputs_embeds=embed(a_ids), past_key_values=cache, attention_mask=mask,
                position_ids=start[:, None] + depth + torch.arange(a_len, device=device)[None], use_cache=False)
    target, keep = a_ids[:, 1:], a_mask[:, 1:].float()
    ce = F.cross_entropy(out.logits[:, :-1].float().reshape(-1, out.logits.shape[-1]), target.reshape(-1),
                         reduction="none").view(width, -1)
    ce = (ce * keep).sum() / keep.sum()
    targets = [t for _, _, t in batch]
    if args.step_weight:
        steps = step_loss(decoder, embed, torch.cat(latents, 1), targets, pad=pad)
    else:
        with torch.no_grad():
            steps = step_loss(decoder, embed, torch.cat(latents, 1).detach(), targets, pad=pad)
    loss = ce + args.step_weight * steps if args.step_weight and steps is not None else ce
    return loss, {"ce": float(ce), "step": float(steps) if steps is not None else None}


def save_model(folder: Path) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    state = {f"codi.base_model.model.{k}": v.detach().cpu() for k, v in model.state_dict().items()
             if not k.endswith((".attn.bias", ".attn.masked_bias"))}
    state.update({f"prj.{k}": v.detach().cpu() for k, v in projection.state_dict().items()})
    torch.save(state, folder / "pytorch_model.bin")


args.output.mkdir(parents=True, exist_ok=True)
resume = args.output / "resume.pt"
done = 0
if resume.exists():
    saved = torch.load(resume, map_location="cpu", weights_only=False)
    model.load_state_dict(saved["model"])
    projection.load_state_dict(saved["projection"])
    optimizer.load_state_dict(saved["optimizer"])
    schedule.load_state_dict(saved["schedule"])
    done = saved["done"]
    print(f"resuming at update {done}/{updates}", flush=True)
(args.output / "args.json").write_text(json.dumps(vars(args) | {"updates": updates, "examples": len(examples), "depth": depth},
                                                  default=str, indent=1))
last = min(updates, args.stop_after or updates)
start_time = time.time()
with (args.output / "metrics.jsonl").open("a") as log:
    for update in range(done, last):
        batch = examples[update * args.batch:(update + 1) * args.batch]
        with torch.autocast("cuda", dtype=torch.bfloat16) if args.bf16 else contextlib.nullcontext():
            loss, stats = train_step(batch)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_([p for g in groups for p in g["params"]], 2.0)
        optimizer.step()
        schedule.step()
        if update % args.log_every == 0 or update == updates - 1:
            line = {"update": update, "loss": round(float(loss), 5), "grad_norm": round(float(norm), 3),
                    "lr": schedule.get_last_lr()[0], "seconds": round(time.time() - start_time, 1)}
            line |= {k: (round(v, 5) if v is not None else None) for k, v in stats.items()}
            log.write(json.dumps(line) + "\n")
            log.flush()
            print(json.dumps(line), flush=True)
        if (update + 1) % args.save_every == 0 and update + 1 < last:
            torch.save({"model": model.state_dict(), "projection": projection.state_dict(),
                        "optimizer": optimizer.state_dict(), "schedule": schedule.state_dict(), "done": update + 1},
                       resume.with_suffix(".tmp"))
            resume.with_suffix(".tmp").replace(resume)
if args.check:
    ours = {r["update"]: r["loss"] for r in map(json.loads, (args.output / "metrics.jsonl").read_text().splitlines())}
    theirs = {r["update"]: r["loss"] for r in map(json.loads, args.check.read_text().splitlines())}
    shared = sorted(set(ours) & set(theirs))
    gaps = {u: abs(ours[u] - theirs[u]) for u in shared}
    result = {"updates": shared, "gaps": gaps, "max_gap": max(gaps.values()) if gaps else None,
              "tolerance": args.tolerance, "passed": bool(gaps) and max(gaps.values()) <= args.tolerance}
    (args.output / "check.json").write_text(json.dumps(result, indent=1) + "\n")
    print(f"check against {args.check}: max gap {result['max_gap']} over updates {shared}: "
          f"{'passed' if result['passed'] else 'FAILED'}", flush=True)
    if not result["passed"]:
        raise SystemExit("the dense code path does not reproduce the reference losses")
if last == updates:
    save_model(args.output / "model")
    resume.unlink(missing_ok=True)
print(f"{args.arm}: {last} of {updates} updates on {len(examples)} examples in {time.time() - start_time:.0f}s", flush=True)
