"""D1-R's conversion (results/ffn_sparsity/d1r/PLAN.md): SIM-CoT GPT-2's FFNs as shared neurons plus 12 routed experts,
built from SPARSE-0's calibration file (Taylor importance and per-phase mean activations), pinned by --sha256.

1. FFN inputs: the dense model in latent mode on SPARSE-0's calibration records, answers teacher-forced, keeping every T
   and A position and a seeded quarter of the Q positions.
2. Experts, per layer: the top quarter of neurons by importance are shared (768 of 3,072). The rest are split into 12
   experts of equal size (192) by balanced k-means on their standardized activations over --features kept positions.
3. Routers, per layer: a 2-layer MLP (width -> 128 -> 12) regressing each expert's contribution norm
   ||W_out,e^T (a_e - mu_e)|| about the static means, fit on the records with index % 10 != 0 and scored on the rest by
   recall of the top 4.
4. Stand-in values per phase, for each arm's dropped neurons: the mean activation over the kept positions where the
   router (or the hash) leaves the neuron's expert off. Prune keeps the top half by importance with SPARSE-0's
   per-phase means, which makes it SPARSE-0's S50.

Writes --output (conversion.pt) and conversion.json beside it.

    python scripts/ffn_sparsity/convert.py --checkpoint <simcot dir> --calibration <SPARSE-0 calibration.pt> \
        --sha256 <hex> --output <conversion.pt> --device cuda
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", required=True, help="a directory holding pytorch_model.bin (fetch_simcot.py)")
parser.add_argument("--calibration", type=Path, required=True, help="SPARSE-0's calibration.pt")
parser.add_argument("--sha256", default=None, help="the calibration file's expected sha256")
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--train-file", type=Path,
                    default=Path(os.environ.get("DSA_DATA_DIR", ROOT / "data")) / "gsm_train.json")
parser.add_argument("--batch", type=int, default=64)
parser.add_argument("--depth", type=int, default=6)
parser.add_argument("--seed", type=int, default=11)
parser.add_argument("--features", type=int, default=8192)
parser.add_argument("--kmeans-iterations", type=int, default=25)
parser.add_argument("--router-steps", type=int, default=2000)
parser.add_argument("--router-batch", type=int, default=4096)
parser.add_argument("--router-lr", type=float, default=1e-3)
parser.add_argument("--device", default="cuda")
args = parser.parse_args()
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch
from dsa.bench.models.codi import CODIGPT2
from moe import ACTIVE, EXPERTS, expert_norms, hash_table, make_router, step_key, top_experts
from sparsity import PHASES, A, Q, T, ranked

began = time.time()
digest = hashlib.sha256(args.calibration.read_bytes()).hexdigest()
if args.sha256 and digest != args.sha256:
    raise SystemExit(f"{args.calibration} has sha256 {digest}, not {args.sha256}")
calibration = torch.load(args.calibration, map_location="cpu", weights_only=True)
codi = CODIGPT2(checkpoint=args.checkpoint, device=args.device)
model, projection, tokenizer, device = codi.model, codi.projection, codi.tokenizer, codi.device
model.eval()
projection.eval()
embed = model.get_input_embeddings()
blocks = [block.mlp for block in model.transformer.h]
layers, inner, vocab = len(blocks), blocks[0].c_fc.nf, codi.vocab_size
shared_size = inner // 4
expert_size = (inner - shared_size) // EXPERTS
assert shared_size + EXPERTS * expert_size == inner, "the routed neurons must split evenly into experts"

# 1. FFN inputs at the kept positions.
source = json.loads(args.train_file.read_text())
examples = []
for index in calibration["indices"]:
    row = source[index]
    question = tokenizer.encode(row["question"].strip().replace("  ", " ")) + [codi.bot_id]
    answer = [codi.eot_id] + tokenizer.encode(f"The answer is: {str(row['answer']).strip()}") + [tokenizer.eos_token_id]
    examples.append((index, question, answer))
del source
inputs = [[] for _ in blocks]
current = {}
hooks = [mlp.register_forward_pre_hook(lambda module, given, layer=layer: inputs[layer].append(
    given[0][current["keep"]].float())) for layer, mlp in enumerate(blocks)]
phases, keys, records = [], [], []
sampler = torch.Generator(device="cpu").manual_seed(args.seed)


def call(phase, valid, key, record, **model_inputs):
    """One model call, keeping its valid T and A positions and a random quarter of its valid Q positions."""
    keep = valid.bool().cpu() & ((phase.cpu() != Q)[None, :] | (torch.rand(valid.shape, generator=sampler) < 0.25))
    current["keep"] = keep.to(device)
    phases.append(phase.cpu()[None, :].expand(valid.shape)[keep])
    keys.append(key.cpu()[keep])
    records.append(record[:, None].expand(valid.shape)[keep])
    return model(**model_inputs)


with torch.no_grad():
    for start in range(0, len(examples), args.batch):
        batch = examples[start:start + args.batch]
        width, pad = len(batch), tokenizer.eos_token_id
        q_len, a_len = max(len(q) for _, q, _ in batch), max(len(a) for _, _, a in batch)
        q_ids = torch.tensor([[pad] * (q_len - len(q)) + q for _, q, _ in batch], device=device)
        q_mask = torch.tensor([[0] * (q_len - len(q)) + [1] * len(q) for _, q, _ in batch], device=device)
        a_ids = torch.tensor([a + [pad] * (a_len - len(a)) for _, _, a in batch], device=device)
        a_mask = torch.tensor([[1] * len(a) + [0] * (a_len - len(a)) for _, _, a in batch], device=device)
        record = torch.tensor([i for i, _, _ in batch])
        positions = (q_mask.cumsum(1) - 1).clamp(min=0)
        ones = torch.ones(width, 1, dtype=q_mask.dtype, device=device)
        q_keys = torch.cat([q_ids[:, :-1], torch.full((width, 1), step_key(vocab, 0), device=device)], 1)
        phase = torch.tensor([Q] * (q_len - 1) + [T])
        out = call(phase, q_mask, q_keys, record, input_ids=q_ids, attention_mask=q_mask, position_ids=positions,
                   use_cache=True, output_hidden_states=True)
        cache, latent = out.past_key_values, projection(out.hidden_states[-1][:, -1:, :])
        begin, mask = positions[:, -1] + 1, q_mask
        for step in range(1, args.depth + 1):
            mask = torch.cat([mask, ones], 1)
            out = call(torch.tensor([T]), ones, torch.full((width, 1), step_key(vocab, step)), record,
                       inputs_embeds=latent, past_key_values=cache, attention_mask=mask,
                       position_ids=(begin + step - 1)[:, None], use_cache=True, output_hidden_states=True)
            cache, latent = out.past_key_values, projection(out.hidden_states[-1][:, -1:, :])
        mask = torch.cat([mask, a_mask], 1)
        call(torch.tensor([A] * a_len), a_mask, a_ids, record, inputs_embeds=embed(a_ids), past_key_values=cache,
             attention_mask=mask, position_ids=begin[:, None] + args.depth + torch.arange(a_len, device=device)[None],
             use_cache=False)
for hook in hooks:
    hook.remove()
phases, keys, records = torch.cat(phases).to(device), torch.cat(keys).to(device), torch.cat(records)
held = (records % 10 == 0).to(device)
print(f"kept positions: {len(phases)} ({', '.join(f'{p} {int((phases == i).sum())}' for i, p in enumerate(PHASES))});"
      f" held out {int(held.sum())}", flush=True)


def balanced_kmeans(features: torch.Tensor, k: int, iterations: int, generator) -> torch.Tensor:
    """Cluster ids [rows] for k clusters of rows // k members each. Each pass assigns rows in order of their distance to
    the nearest centroid, each to its nearest centroid with room left; centroids are then the cluster means."""
    n, size = len(features), len(features) // k
    centroids = features[torch.randperm(n, generator=generator)[:k].to(features.device)].clone()
    assign = torch.zeros(n, dtype=torch.long)
    for _ in range(iterations):
        distance = torch.cdist(features, centroids).cpu()
        preference = torch.argsort(distance, dim=1, stable=True).tolist()
        room = [size] * k
        for row in torch.argsort(distance.min(1).values, stable=True).tolist():
            cluster = next(c for c in preference[row] if room[c])
            assign[row] = cluster
            room[cluster] -= 1
        centroids = torch.stack([features[(assign == c).to(features.device)].mean(0) for c in range(k)])
    return assign


def means_where_off(a: torch.Tensor, off: torch.Tensor, fallback: torch.Tensor) -> torch.Tensor:
    """[phases, inner]: the mean of a over the positions of each phase where off is set, else the fallback."""
    out = fallback.clone()
    for p in range(len(PHASES)):
        rows = phases == p
        count = off[rows].sum(0)
        total = (a[rows] * off[rows]).sum(0)
        out[p] = torch.where(count > 0, total / count.clamp(min=1), fallback[p])
    return out


table = hash_table(layers, vocab + args.depth + 1, seed=args.seed)
static = calibration["mu"]["static"].to(device)
conversion = {"shared": torch.zeros(layers, inner, dtype=torch.bool), "prune": torch.zeros(layers, inner, dtype=torch.bool),
              "owner": torch.full((layers, inner), -1, dtype=torch.long), "hash": table, "routers": [],
              "mu": {"moe": static.clone(), "hash": static.clone(), "prune": static.clone()}}
summary = {"calibration_sha256": digest, "positions": {p: int((phases == i).sum()) for i, p in enumerate(PHASES)},
           "held_out": int(held.sum()), "layers": {}}
generator = torch.Generator().manual_seed(args.seed)
for layer, mlp in enumerate(blocks):
    with torch.no_grad():
        x = torch.cat(inputs[layer])
        inputs[layer] = None
        a = torch.cat([mlp.act(mlp.c_fc(chunk)) for chunk in x.split(8192)])
        order = ranked(calibration["importance"][layer])
        conversion["shared"][layer, order[:shared_size]] = True
        conversion["prune"][layer, order[:shared_size + ACTIVE * expert_size]] = True
        routed = torch.sort(order[shared_size:]).values
        sample = torch.randperm(len(a), generator=generator)[:args.features].to(device)
        features = a[sample][:, routed.to(device)].T
        features = (features - features.mean(1, keepdim=True)) / features.std(1, keepdim=True).clamp(min=1e-6)
        conversion["owner"][layer, routed] = balanced_kmeans(features, EXPERTS, args.kmeans_iterations, generator)
        owner = conversion["owner"][layer].to(device)
        members = [torch.nonzero(owner == e).flatten() for e in range(EXPERTS)]
        weight = mlp.c_proj.weight.float()
        targets = torch.cat([expert_norms(chunk - static[layer][p], weight, members)
                             for chunk, p in zip(a.split(8192), phases.split(8192))])
    torch.manual_seed(args.seed + layer)
    router = make_router(x.shape[1]).to(device)
    optimizer = torch.optim.Adam(router.parameters(), lr=args.router_lr)
    train_rows = torch.nonzero(~held).flatten()
    for _ in range(args.router_steps):
        rows = train_rows[torch.randint(len(train_rows), (min(args.router_batch, len(train_rows)),),
                                        generator=generator).to(device)]
        loss = torch.nn.functional.mse_loss(router(x[rows]), targets[rows])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    router.eval()
    with torch.no_grad():
        scores = torch.cat([router(chunk) for chunk in x.split(8192)])
        chosen, best = top_experts(scores), top_experts(targets)
        hits = (chosen & best).sum(-1).float() / ACTIVE
        recall = {p: float(hits[held & (phases == i)].mean()) if (held & (phases == i)).any() else None
                  for i, p in enumerate(PHASES)}
        mse = {split: float(((scores - targets) ** 2)[rows].mean()) if rows.any() else None
               for split, rows in (("train", ~held), ("held", held))}
        for name, pick in (("moe", chosen), ("hash", table[layer].to(device)[keys])):
            off = ~(pick[:, owner.clamp(min=0)] | conversion["shared"][layer].to(device)[None])
            conversion["mu"][name][layer] = means_where_off(a, off, static[layer])
    conversion["routers"].append({k: v.detach().cpu() for k, v in router.state_dict().items()})
    summary["layers"][layer] = {"recall": recall, "mse": mse, "target_mean": float(targets.mean())}
    print(f"layer {layer}: router recall of the oracle's top {ACTIVE} on held-out positions "
          + ", ".join(f"{p} {v:.2f}" for p, v in recall.items() if v is not None) + f"; held-out mse {mse['held']}",
          flush=True)
    del x, a, features, targets, scores
conversion["mu"] = {name: value.cpu() for name, value in conversion["mu"].items()}
args.output.parent.mkdir(parents=True, exist_ok=True)
torch.save(conversion, args.output.with_suffix(".tmp"))
args.output.with_suffix(".tmp").replace(args.output)
summary["seconds"] = time.time() - began
(args.output.parent / "conversion.json").write_text(json.dumps(summary, indent=1) + "\n")
print(f"conversion: {layers} layers, {shared_size} shared + {EXPERTS} x {expert_size} routed, top {ACTIVE}; "
      f"{summary['seconds']:.0f}s -> {args.output}", flush=True)
