"""SPARSE-0: how much FFN width per token SIM-CoT GPT-2 needs, in which phase, and whether per-token selection beats a
fixed pruned mask (results/ffn_sparsity/PLAN.md). Evaluation only: nothing is trained.

Phase 1, calibration: on --calibration training records in latent mode with the answer teacher-forced, the dense model's
first-order Taylor importance of every FFN neuron and the representative values mu per phase. Saved to calibration.pt.
Phase 2, arms: greedy on the validation questions under each arm's mask (sparsity.ARMS), run as
dsa/bench/models/codi.py runs it, one evals/<arm>.jsonl per arm, resumable by arm. ID must match --reference, if given,
before any other arm runs; it also logs how much the oracle's neuron sets overlap between neighbouring positions.

Outputs in --output: calibration.pt, evals/<arm>.jsonl, overlap.json and status.json.

    python scripts/ffn_sparsity/run.py --checkpoint <simcot dir> --output <dir> --reference <seed1.jsonl> --device cuda
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", required=True, help="a directory holding pytorch_model.bin (fetch_simcot.py)")
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--train-file", type=Path,
                    default=Path(os.environ.get("DSA_DATA_DIR", ROOT / "data")) / "gsm_train.json")
parser.add_argument("--valid-file", type=Path, default=None, help="a JSON list in place of the locked gsm8k_valid")
parser.add_argument("--reference", type=Path, default=None, help="greedy predictions by id that ID must match")
parser.add_argument("--calibration", type=int, default=2048)
parser.add_argument("--batch", type=int, default=64)
parser.add_argument("--seed", type=int, default=11)
parser.add_argument("--depth", type=int, default=6)
parser.add_argument("--max-question-tokens", type=int, default=256)
parser.add_argument("--device", default="cuda")
args = parser.parse_args()
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch
import torch.nn.functional as F
from dsa.bench import data as bench_data
from dsa.bench.models.codi import CODIGPT2
from dsa.bench.scoring import score
from sparsity import ARMS, FRACTIONS, PHASES, A, T, FFNMasks, neurons, oracle_keep, phase_tags, ranked

codi = CODIGPT2(checkpoint=args.checkpoint, device=args.device)
model, projection, tokenizer, device = codi.model, codi.projection, codi.tokenizer, codi.device
model.eval()
projection.eval()
embed = model.get_input_embeddings()
masks = FFNMasks(model)
layers, inner = len(masks.blocks), masks.inner
out_dir = args.output
(out_dir / "evals").mkdir(parents=True, exist_ok=True)
status_path = out_dir / "status.json"
status = json.loads(status_path.read_text()) if status_path.exists() else {}


def save_status():
    status_path.write_text(json.dumps(status, indent=1) + "\n")


# Phase 1: calibration on training records, the dense model, the answer teacher-forced.
def calibration_examples():
    source = json.loads(args.train_file.read_text())
    examples = []
    for index in random.Random(args.seed).sample(range(len(source)), len(source)):
        row = source[index]
        question = tokenizer.encode(row["question"].strip().replace("  ", " ")) + [codi.bot_id]
        if len(question) > args.max_question_tokens or not str(row["answer"]).strip():
            continue
        answer = [codi.eot_id] + tokenizer.encode(f"The answer is: {str(row['answer']).strip()}") + [tokenizer.eos_token_id]
        examples.append((index, question, answer))
        if len(examples) == args.calibration:
            break
    return examples


def calibrate(batch, totals):
    """One batch: adds each example's |sum over its positions of a_i * dL/da_i| to the importance, and the activation sums
    and counts behind mu."""
    width, pad = len(batch), tokenizer.eos_token_id
    q_len, a_len = max(len(q) for _, q, _ in batch), max(len(a) for _, _, a in batch)
    q_ids = torch.tensor([[pad] * (q_len - len(q)) + q for _, q, _ in batch], device=device)
    q_mask = torch.tensor([[0] * (q_len - len(q)) + [1] * len(q) for _, q, _ in batch], device=device)
    a_ids = torch.tensor([a + [pad] * (a_len - len(a)) for _, _, a in batch], device=device)
    a_mask = torch.tensor([[1] * len(a) + [0] * (a_len - len(a)) for _, _, a in batch], device=device)
    positions = (q_mask.cumsum(1) - 1).clamp(min=0)
    calls, stored = [], []  # (phase, valid) per model call; (call, layer, a) per FFN
    masks.watch = lambda layer, a, phase: stored.append((len(calls) - 1, layer, a))

    def call(phase, valid, **inputs):
        calls.append((phase, valid))
        masks.phase = phase
        return model(**inputs)

    ones = torch.ones(width, 1, dtype=q_mask.dtype, device=device)
    out = call(phase_tags(q_len - 1, 1, device=device), q_mask, input_ids=q_ids, attention_mask=q_mask,
               position_ids=positions, use_cache=True, output_hidden_states=True)
    cache, latent = out.past_key_values, projection(out.hidden_states[-1][:, -1:, :])
    start, mask = positions[:, -1] + 1, q_mask
    for step in range(1, args.depth + 1):
        mask = torch.cat([mask, ones], 1)
        out = call(phase_tags(latent=1, device=device), ones, inputs_embeds=latent, past_key_values=cache,
                   attention_mask=mask, position_ids=(start + step - 1)[:, None], use_cache=True,
                   output_hidden_states=True)
        cache, latent = out.past_key_values, projection(out.hidden_states[-1][:, -1:, :])
    mask = torch.cat([mask, a_mask], 1)
    out = call(phase_tags(answer=a_len, device=device), a_mask, inputs_embeds=embed(a_ids), past_key_values=cache,
               attention_mask=mask, position_ids=start[:, None] + args.depth + torch.arange(a_len, device=device)[None],
               use_cache=False)
    masks.watch = None
    target, keep = a_ids[:, 1:], a_mask[:, 1:].float()
    ce = F.cross_entropy(out.logits[:, :-1].float().reshape(-1, out.logits.shape[-1]), target.reshape(-1),
                         reduction="none").view(width, -1)
    loss = ((ce * keep).sum(1) / keep.sum(1)).sum()  # each example's mean answer cross-entropy, summed
    # Unused: the last layer's FFN at positions whose final state nothing reads (question tokens, the last latent step).
    grads = torch.autograd.grad(loss, [a for _, _, a in stored], allow_unused=True)
    taylor = torch.zeros(layers, width, inner, dtype=torch.float64, device=device)
    for (number, layer, a), g in zip(stored, grads):
        phase, valid = calls[number]
        a, valid = a.detach().float(), valid.bool()
        if g is not None:
            taylor[layer] += ((a * g.float()) * valid[..., None]).sum(1).double()
        for p in range(len(PHASES)):
            here = valid & (phase == p)[None, :]
            if not here.any():
                continue
            totals["static"][0][layer, p] += (a * here[..., None]).sum((0, 1)).double()
            totals["static"][1][layer, p] += here.sum().double()
            for fraction in FRACTIONS:
                dropped = ~oracle_keep(a, masks.norms[layer], neurons(fraction, inner)) & here[..., None]
                totals[fraction][0][layer, p] += (a * dropped).sum((0, 1)).double()
                totals[fraction][1][layer, p] += dropped.sum((0, 1)).double()
    totals["importance"] += taylor.abs().sum(1)
    return float(loss) / width


calibration_path = out_dir / "calibration.pt"
if not calibration_path.exists():
    began = time.time()
    examples = calibration_examples()
    shape = (layers, len(PHASES), inner)
    zeros = lambda size: torch.zeros(size, dtype=torch.float64, device=device)  # noqa: E731
    totals = {"importance": zeros((layers, inner)), "static": (zeros(shape), zeros(shape)),
              **{fraction: (zeros(shape), zeros(shape)) for fraction in FRACTIONS}}
    losses = []
    with torch.enable_grad():
        for start in range(0, len(examples), args.batch):
            losses.append(calibrate(examples[start:start + args.batch], totals))
    static_sum, static_count = totals["static"]
    everywhere = static_sum / static_count.clamp(min=1)  # a statically dropped neuron is dropped at every position
    mu = {"static": everywhere.float().cpu()}
    for fraction in FRACTIONS:
        total, count = totals[fraction]
        mu[f"oracle-{fraction}"] = torch.where(count > 0, total / count.clamp(min=1), everywhere).float().cpu()
    importance = totals["importance"].cpu()
    calibration = {"importance": importance, "mu": mu, "indices": [i for i, _, _ in examples],
                   "positions": {p: int(static_count[0, i, 0]) for i, p in enumerate(PHASES)}}
    torch.save(calibration, calibration_path.with_suffix(".tmp"))
    calibration_path.with_suffix(".tmp").replace(calibration_path)  # the file marks phase 1 done
    status["calibration"] = {"records": len(examples), "positions": calibration["positions"],
                             "mean_answer_ce": sum(losses) / len(losses), "seconds": time.time() - began}
    save_status()
print("calibration:", json.dumps(status["calibration"]), flush=True)
calibration = torch.load(calibration_path, map_location="cpu", weights_only=True)
for fraction in FRACTIONS:
    order = ranked(calibration["importance"])[:, :neurons(fraction, inner)]
    masks.static[fraction] = torch.zeros(layers, inner, dtype=torch.bool).scatter_(1, order, True).to(device)
    masks.mu[("static", fraction)] = calibration["mu"]["static"].to(device)
    masks.mu[("oracle", fraction)] = calibration["mu"][f"oracle-{fraction}"].to(device)


# Phase 2: the arms, greedy on validation.
class Overlap:
    """In ID: the Jaccard overlap of the oracle's kept sets between consecutive question tokens (Q) and consecutive
    thinking positions (T, from <bot> to latent step 6), per fraction and layer."""

    def __init__(self):
        self.sums = {(f, p): [0.0] * layers for f in FRACTIONS for p in "QT"}
        self.counts = {(f, p): [0] * layers for f in FRACTIONS for p in "QT"}
        self.last = {}

    def add(self, fraction, phase, layer, before, after):
        both, either = (before & after).sum(-1).float(), (before | after).sum(-1).float()
        self.sums[(fraction, phase)][layer] += float((both / either).sum())
        self.counts[(fraction, phase)][layer] += len(both)

    def __call__(self, layer, a, phase):
        if int(phase[0]) == A:
            return
        for fraction in FRACTIONS:
            keep = oracle_keep(a[0], masks.norms[layer], neurons(fraction, inner))
            if int(phase[0]) == T:  # a latent step
                self.add(fraction, "T", layer, self.last[(fraction, layer)][None], keep)
            elif len(keep) > 2:  # the question pass: question tokens, then <bot>
                self.add(fraction, "Q", layer, keep[:-2], keep[1:-1])
            self.last[(fraction, layer)] = keep[-1]

    def summary(self):
        return {f"{f}": {p: [s / max(1, c) for s, c in zip(self.sums[(f, p)], self.counts[(f, p)])] for p in "QT"}
                for f in FRACTIONS}


@torch.no_grad()
def greedy(ids):
    """Greedy text and positions fed per phase, as dsa/bench/models/codi.py runs it."""
    masks.phase = phase_tags(len(ids) - 1, 1, device=device)
    out = model(input_ids=torch.tensor([ids], device=device), use_cache=True, output_hidden_states=True)
    cache, latent = out.past_key_values, projection(out.hidden_states[-1][:, -1:, :])
    masks.phase = phase_tags(latent=1, device=device)
    for _ in range(args.depth):
        out = model(inputs_embeds=latent, past_key_values=cache, use_cache=True, output_hidden_states=True)
        cache, latent = out.past_key_values, projection(out.hidden_states[-1][:, -1:, :])
    masks.phase = phase_tags(answer=1, device=device)
    step, tokens = embed(torch.tensor([[codi.eot_id]], device=device)), []
    for _ in range(codi.max_new_tokens):
        out = model(inputs_embeds=step, past_key_values=cache, use_cache=True)
        cache = out.past_key_values
        token = int(out.logits[0, -1, :codi.vocab_size - 1].argmax())
        tokens.append(token)
        if token == tokenizer.eos_token_id:
            break
        step = embed(torch.tensor([[token]], device=device))
    text = tokenizer.decode([t for t in tokens if t < len(tokenizer)], skip_special_tokens=True)
    return text, [len(ids) - 1, 1 + args.depth, len(tokens)]


if args.valid_file:
    rows = json.loads(args.valid_file.read_text())
else:
    rows, _ = bench_data.load_eval_rows("gsm8k_valid", Path(os.environ.get("DSA_DATA_DIR", ROOT / "data")))
for name, arm in ARMS.items():
    path = out_dir / f"evals/{name}.jsonl"
    if path.exists():
        continue
    began = time.time()
    overlap = Overlap() if arm is None else None
    masks.arm, masks.watch = arm, overlap
    records = []
    for row in rows:
        ids = tokenizer.encode(row["question"].strip().replace("  ", " ")) + [codi.bot_id]
        text, positions = greedy(ids)
        scored = score(text, str(row["answer"]), "numeric")
        records.append({"id": row["id"], "rationale_steps": len(re.findall(r"<<", row.get("rationale", ""))),
                        "text": text, "prediction": scored["prediction"], "correct": bool(scored["correct"]),
                        "positions": positions})
    masks.arm, masks.watch = None, None
    if arm is None:
        matched = None
        if args.reference:
            reference = {json.loads(x)["id"]: json.loads(x)["greedy"]["prediction"]
                         for x in args.reference.read_text().splitlines()}
            matched = sum(reference.get(r["id"]) == r["prediction"] for r in records)
            if matched != len(records):
                raise SystemExit(f"identity check failed: {matched} of {len(records)} ID predictions match "
                                 f"{args.reference}")
        status["identity"] = {"questions": len(records), "matched": matched}
        (out_dir / "overlap.json").write_text(json.dumps(overlap.summary(), indent=1) + "\n")
    status.setdefault("arms", {})[name] = {"right": sum(r["correct"] for r in records), "seconds": time.time() - began}
    save_status()
    path.write_text("".join(json.dumps(r) + "\n" for r in records))  # written last: marks the arm done
    print(f"{name}: {status['arms'][name]['right']} of {len(records)} in {status['arms'][name]['seconds']:.0f}s",
          flush=True)
status["model"] = {"width": model.config.n_embd, "inner": inner, "layers": layers}
save_status()
