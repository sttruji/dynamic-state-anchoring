"""Stage 1 evaluation: a checkpoint at several latent depths, with the cache unbounded, held to a budget, or re-weighted.

Policies, scored at every depth past the budget (`plain` also at the budget depth itself):
    plain       the full cache
    window      keep the most recent `budget` latent entries; evict the rest
    random      evict a random latent entry once there are more than `budget` (seeded by the question index)
    logn        no eviction; with n > budget latent entries, each gets -log(n / budget), so the latent block keeps the
                attention mass `budget` entries would get (additive, block-level)
    logscale    no eviction; every logit of a query with K keys is multiplied by log K / log K6, K6 being its key count
                had the loop stopped at `budget` (multiplicative, entropy-invariant / SSMax-style sharpening)
    dedup_soft  no eviction; each latent entry gets -log(size of its cluster), the entries whose fed latents have cosine
                >= 0.95 with it, which exactly cancels the boost from exact copies
    gate        evict the entry with the lowest gate score once there are more than `budget`; log g on the kept ones
    gate_soft   no eviction, log g on every latent entry: the condition the gate was trained in
The re-weighting policies act only once there are more than `budget` latent entries, so a run that stops at the budget is
the plain run.

    python scripts/latent_gate/stage1_eval.py --checkpoint <run>/model --dataset gsm8k_valid --output <file.jsonl>
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", help="a directory holding pytorch_model.bin (and gate.pt for gated arms)")
parser.add_argument("--dataset", required=True)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--depths", type=int, nargs="+", default=[6, 12, 24])
parser.add_argument("--budget", type=int, default=6)
parser.add_argument("--policies", nargs="+", help="default: all that apply to the checkpoint")
parser.add_argument("--device", default="cuda")
parser.add_argument("--data-dir", type=Path, default=None, help="default: $DSA_DATA_DIR or <repo>/data")
parser.add_argument("--limit", type=int)
parser.add_argument("--shard", type=int, default=0)
parser.add_argument("--shards", type=int, default=1)
args = parser.parse_args()
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import math
import os
import torch
import torch.nn.functional as F
from dsa.bench import data as bench_data
from dsa.bench.models.codi import CODIGPT2
from dsa.bench.scoring import score
from gate import Gate, gold_results, install_bias, set_bias

EVICTING = ("window", "random", "gate")
data_dir = args.data_dir or Path(os.environ.get("DSA_DATA_DIR", ROOT / "data"))
rows, _ = bench_data.load_eval_rows(args.dataset, data_dir)
answer_type = bench_data.EVAL_DATASETS[args.dataset]["answer_type"]
codi = CODIGPT2(checkpoint=args.checkpoint, device=args.device)
model, projection, device = codi.model, codi.projection, codi.device
install_bias(model)
embed = model.get_input_embeddings()
gate_file = Path(args.checkpoint) / "gate.pt" if args.checkpoint else None
gate = None
if gate_file and gate_file.exists():
    saved = torch.load(gate_file, map_location="cpu", weights_only=True)
    gate = Gate(saved["width"]).to(device).eval()
    gate.load_state_dict(saved["state"])
policies = tuple(args.policies or ("plain", "window", "random", "logn", "logscale", "dedup_soft")
                 + (("gate", "gate_soft") if gate is not None else ()))
assert gate is not None or not any(p.startswith("gate") for p in policies), "gate policies need a gate.pt"
deepest = max(args.depths)


def weights(policy, fed, log_gates):
    """Additive bias on each current latent entry (a list of 0-d tensors), or None."""
    count = len(fed)
    if policy.startswith("gate"):
        return log_gates or None
    if count <= args.budget:
        return None
    if policy == "logn":
        return [torch.tensor(-math.log(count / args.budget), device=device)] * count
    if policy == "dedup_soft":
        vectors = F.normalize(torch.stack(fed), dim=-1)
        sizes = ((vectors @ vectors.T) >= 0.95).sum(1).float()
        return list(-sizes.log())
    return None


def attend(policy, keys, latents, fed, log_gates):
    """Set the bias and query scale for the next forward, whose queries see `keys` keys."""
    shown = weights(policy, fed, log_gates)
    bias = None
    if shown:
        bias = torch.zeros(1, keys, device=device)
        bias[0, question + 1:question + 1 + len(shown)] = torch.stack(shown)
        bias = bias[:, None, None, :]
    scale = None
    if policy == "logscale" and latents > args.budget:
        scale = math.log(keys) / math.log(keys - (latents - args.budget))
    set_bias(model, bias, scale)


def readout(cache, row, position, policy, fed, log_gates):
    step = embed(torch.tensor([[codi.eot_id]], device=device))
    tokens = []
    for offset in range(codi.max_new_tokens):
        attend(policy, cache[0][0].shape[2] + 1, len(fed), fed, log_gates)
        output = model(inputs_embeds=step, past_key_values=cache, use_cache=True,
                       position_ids=torch.tensor([[position + offset]], device=device))
        cache = output.past_key_values
        token = int(output.logits[0, -1, :codi.vocab_size - 1].argmax())
        tokens.append(token)
        if token == codi.tokenizer.eos_token_id:
            break
        step = embed(torch.tensor([[token]], device=device))
    set_bias(model, None)
    text = codi.tokenizer.decode([t for t in tokens if t < len(codi.tokenizer)], skip_special_tokens=True)
    scored = score(text, row["answer"], row.get("answer_type", answer_type))
    return {"text": text, "prediction": scored["prediction"], "correct": bool(scored["correct"])}


def run(policy, row, asked, first, index):
    """One latent loop under `policy`, with readouts at every requested depth."""
    cache, latent, entries, fed, log_gates, out = asked, first, [], [], [], {}
    generator = torch.Generator().manual_seed(index)
    for step in range(1, deepest + 1):
        attend(policy, cache[0][0].shape[2] + 1, len(fed), fed, log_gates)
        output = model(inputs_embeds=latent, past_key_values=cache, use_cache=True, output_hidden_states=True,
                       position_ids=torch.tensor([[question + step]], device=device))
        set_bias(model, None)
        cache, hidden = output.past_key_values, output.hidden_states[-1][:, -1, :]
        entries.append(step)
        fed.append(latent[0, 0])
        if policy.startswith("gate"):
            log_gates.append(F.logsigmoid(gate(hidden.float()))[0])
        latent = projection(hidden[:, None, :])
        if policy in EVICTING and len(entries) > args.budget:
            if policy == "window":
                drop = 0
            elif policy == "random":
                drop = int(torch.randint(len(entries), (1,), generator=generator))
            else:  # lowest gate; a tie evicts the newest, so an undecided gate falls back to keeping the first entries
                drop = len(log_gates) - 1 - int(torch.stack(log_gates).flip(0).argmin())
            index_kept = torch.tensor([i for i in range(cache[0][0].shape[2]) if i != question + 1 + drop], device=device)
            cache = tuple((k[:, :, index_kept], v[:, :, index_kept]) for k, v in cache)
            for kept in (entries, fed) + ((log_gates,) if log_gates else ()):
                del kept[drop]
        if step in args.depths and (policy == "plain" or step > args.budget):
            out[str(step)] = readout(cache, row, question + step + 1, policy, fed, log_gates) | {"kept": list(entries)}
    return out


indexed = list(enumerate(rows))[:args.limit] if args.limit else list(enumerate(rows))
indexed = indexed[args.shard::args.shards]
args.output.parent.mkdir(parents=True, exist_ok=True)
start = time.time()
with torch.no_grad(), args.output.open("w") as sink:
    for number, (index, row) in enumerate(indexed, 1):
        encoded = codi.tokenizer.encode(row["question"].strip().replace("  ", " ")) + [codi.bot_id]
        question = len(encoded) - 1  # <bot> sits at this cache index, latent step t at question + t
        output = model(input_ids=torch.tensor([encoded], device=device), use_cache=True, output_hidden_states=True)
        asked, first = output.past_key_values, projection(output.hidden_states[-1][:, -1:, :])
        record = {"id": row["id"], "gold": row["answer"], "rationale_steps": len(gold_results(row.get("rationale", "")))}
        for policy in policies:
            record[policy] = run(policy, row, asked, first, index)
        sink.write(json.dumps(record) + "\n")
        sink.flush()
        if number % 50 == 0:
            print(f"{args.dataset} shard {args.shard}: {number}/{len(indexed)} {time.time() - start:.0f}s", flush=True)

records = [json.loads(line) for line in args.output.read_text().splitlines()]
print(f"{args.dataset}, n={len(records)}, correct (4+ rationale steps in brackets):")
for policy in policies:
    cells = []
    for depth in args.depths:
        key = str(depth)
        if records and key in records[0][policy]:
            hits = [r[policy][key]["correct"] for r in records]
            long = [r[policy][key]["correct"] for r in records if r["rationale_steps"] >= 4]
            cells.append(f"{depth}: {sum(hits)} ({sum(long)}/{len(long)})")
    print(f"  {policy:10} " + "  ".join(cells))
print(f"finished in {time.time() - start:.0f}s", flush=True)
