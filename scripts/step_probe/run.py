"""Where long problems break: extraction and probes (results/step_probe/PLAN.md). Evaluation only.

Phase 1, validation: greedy on every validation question, run as dsa/bench/models/codi.py runs it, keeping the hidden
states of --layers at <bot>, latent steps 1..depth and <eot>. Predictions must match --reference, if given.
Phase 2, training: the same on up to --per-length pool questions per step count, round-robin across counts, in chunks
of --chunk, resumable and capped at --minutes.
Phase 3, probes: per target (step values c_1..c_6, then the gold answer), layer and position, a linear probe to 4 digits
trained on greedy-right training questions, cross-fitted over --folds. Readings go to readings.npz.

Outputs in --output: valid/records.jsonl, valid/features.npy, train/selection.json, train/chunk_*.npy and .jsonl,
readings.npz and status.json.

    python scripts/step_probe/run.py --checkpoint <simcot dir> --output <dir> --reference <seed1.jsonl> --device cuda
"""
from __future__ import annotations

import argparse
import hashlib
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
parser.add_argument("--reference", type=Path, default=None, help="greedy predictions by id that validation must match")
parser.add_argument("--per-length", type=int, default=1500)
parser.add_argument("--max-steps", type=int, default=6)
parser.add_argument("--minutes", type=float, default=12.0, help="cap on training extraction, over all sessions")
parser.add_argument("--chunk", type=int, default=500)
parser.add_argument("--layers", default="3,6,9,12")
parser.add_argument("--folds", type=int, default=5)
parser.add_argument("--probe-steps", type=int, default=300)
parser.add_argument("--probe-lr", type=float, default=0.01)
parser.add_argument("--weight-decay", type=float, default=0.001)
parser.add_argument("--seed", type=int, default=11)
parser.add_argument("--depth", type=int, default=6)
parser.add_argument("--max-question-tokens", type=int, default=256)
parser.add_argument("--device", default="cuda")
args = parser.parse_args()
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import torch.nn.functional as F
from dsa.bench import data as bench_data
from dsa.bench.models.codi import CODIGPT2
from dsa.bench.scoring import score

LAYERS = [int(x) for x in args.layers.split(",")]
TARGETS = args.max_steps + 1  # c_1..c_max, then the gold answer
DIGITS = 4
codi = CODIGPT2(checkpoint=args.checkpoint, device=args.device)
model, projection, tokenizer, device = codi.model, codi.projection, codi.tokenizer, codi.device
model.eval()
projection.eval()
embed = model.get_input_embeddings()
out_dir = args.output
(out_dir / "valid").mkdir(parents=True, exist_ok=True)
(out_dir / "train").mkdir(parents=True, exist_ok=True)
status_path = out_dir / "status.json"
status = json.loads(status_path.read_text()) if status_path.exists() else {}


def save_status():
    status_path.write_text(json.dumps(status, indent=1) + "\n")


def as_target(text):
    """An integer in [0, 9999], or None."""
    try:
        value = float(str(text).replace(",", ""))
    except ValueError:
        return None
    return int(value) if value == int(value) and 0 <= value <= 9999 else None


def gold_of(answer):
    try:
        value = float(str(answer).replace(",", ""))
    except ValueError:
        return None
    return int(value) if value >= 0 and value == int(value) else None


def steps_of(row) -> list[str]:
    return list(row["steps"]) if "steps" in row else re.findall(r"<<[^<>]*>>", row.get("rationale", ""))


def step_value(step: str):
    return as_target(step.strip().strip("<>").rsplit("=", 1)[-1]) if "=" in step else None


def question_numbers(text: str) -> set[float]:
    return {float(x.replace(",", "")) for x in re.findall(r"\d[\d,]*\.?\d*", text) if x.replace(",", "").strip(".")}


def encode(row):
    ids = tokenizer.encode(row["question"].strip().replace("  ", " ")) + [codi.bot_id]
    return ids if len(ids) <= args.max_question_tokens else None


def layers_at(hidden_states):
    """The kept layers' hidden states at the last position: [layers, width]."""
    return torch.stack([hidden_states[layer][0, -1] for layer in LAYERS])


@torch.no_grad()
def greedy(ids):
    """Greedy text and the hidden states [positions, layers, width], positions <bot>, latent steps, <eot>."""
    out = model(input_ids=torch.tensor([ids], device=device), use_cache=True, output_hidden_states=True)
    cache, states = out.past_key_values, [layers_at(out.hidden_states)]
    latent = projection(out.hidden_states[-1][:, -1:, :])
    for _ in range(args.depth):
        out = model(inputs_embeds=latent, past_key_values=cache, use_cache=True, output_hidden_states=True)
        cache = out.past_key_values
        states.append(layers_at(out.hidden_states))
        latent = projection(out.hidden_states[-1][:, -1:, :])
    step, tokens = embed(torch.tensor([[codi.eot_id]], device=device)), []
    for position in range(codi.max_new_tokens):
        out = model(inputs_embeds=step, past_key_values=cache, use_cache=True, output_hidden_states=position == 0)
        if position == 0:
            states.append(layers_at(out.hidden_states))
        cache = out.past_key_values
        token = int(out.logits[0, -1, :codi.vocab_size - 1].argmax())
        tokens.append(token)
        if token == tokenizer.eos_token_id:
            break
        step = embed(torch.tensor([[token]], device=device))
    text = tokenizer.decode([t for t in tokens if t < len(tokenizer)], skip_special_tokens=True)
    return text, torch.stack(states).to(torch.float16).cpu().numpy()


def record(row, ids, key):
    text, states = greedy(ids)
    steps = steps_of(row)
    scored = score(text, str(row["answer"]), "numeric")
    numbers = question_numbers(row["question"])
    values = [step_value(s) for s in steps]
    return {**key, "steps": len(steps), "values": values, "gold": gold_of(row["answer"]), "text": text,
            "prediction": scored["prediction"], "correct": bool(scored["correct"]),
            "in_question": [v is not None and float(v) in numbers for v in values]}, states


# Phase 1: validation, with the identity check.
valid_records = out_dir / "valid/records.jsonl"
if not valid_records.exists():
    start = time.time()
    if args.valid_file:
        rows = json.loads(args.valid_file.read_text())
    else:
        rows, _ = bench_data.load_eval_rows("gsm8k_valid", Path(os.environ.get("DSA_DATA_DIR", ROOT / "data")))
    records, features = [], []
    for row in rows:
        ids = tokenizer.encode(row["question"].strip().replace("  ", " ")) + [codi.bot_id]
        entry, states = record(row, ids, {"id": row["id"]})
        records.append(entry)
        features.append(states)
    matched = None
    if args.reference:
        reference = {json.loads(x)["id"]: json.loads(x)["greedy"]["prediction"]
                     for x in args.reference.read_text().splitlines()}
        matched = sum(reference.get(r["id"]) == r["prediction"] for r in records)
        if matched != len(records):
            raise SystemExit(f"identity check failed: {matched} of {len(records)} validation predictions match "
                             f"{args.reference}")
    np.save(out_dir / "valid/features.npy", np.stack(features))
    status["valid"] = {"questions": len(records), "right": sum(r["correct"] for r in records),
                       "identity_matched": matched, "seconds": time.time() - start}
    save_status()
    valid_records.write_text("".join(json.dumps(r) + "\n" for r in records))  # written last: marks phase 1 done
print("validation:", json.dumps(status["valid"]), flush=True)

# Phase 2: training, round-robin across step counts, in resumable chunks.
selection_path = out_dir / "train/selection.json"
source = json.loads(args.train_file.read_text())
if not selection_path.exists():
    buckets = {n: [] for n in range(1, args.max_steps + 1)}
    for index in random.Random(args.seed).sample(range(len(source)), len(source)):
        row = source[index]
        n = len(steps_of(row))
        if n not in buckets or len(buckets[n]) >= args.per_length:
            continue
        if gold_of(row["answer"]) is None or encode(row) is None:
            continue
        buckets[n].append(index)
        if all(len(b) >= args.per_length for b in buckets.values()):
            break
    order = [b[i] for i in range(max(len(b) for b in buckets.values())) for b in buckets.values() if i < len(b)]
    selection_path.write_text(json.dumps({"order": order, "per_length": {n: len(b) for n, b in buckets.items()}}))
selection = json.loads(selection_path.read_text())["order"]
train = status.setdefault("train", {"selected": len(selection), "done": 0, "seconds": 0.0, "capped": False})
for start in range(0, len(selection), args.chunk):
    chunk_records = out_dir / f"train/chunk_{start:05d}.jsonl"
    if chunk_records.exists():
        continue
    if train["seconds"] > 60 * args.minutes:
        train["capped"] = True
        break
    began, records, features = time.time(), [], []
    for index in selection[start:start + args.chunk]:
        entry, states = record(source[index], encode(source[index]), {"index": index})
        records.append(entry)
        features.append(states)
    np.save(out_dir / f"train/chunk_{start:05d}.npy", np.stack(features))
    chunk_records.write_text("".join(json.dumps(r) + "\n" for r in records))  # written last: marks the chunk done
    train["done"] += len(records)
    train["seconds"] += time.time() - began
    save_status()
    print(f"training: {train['done']} of {len(selection)} questions, {train['seconds']:.0f}s", flush=True)
save_status()

# Phase 3: probes.
chunks = sorted(out_dir.glob("train/chunk_*.jsonl"))
train_records = [json.loads(x) for path in chunks for x in path.read_text().splitlines()]
train_features = np.concatenate([np.load(path.with_suffix(".npy")) for path in chunks])
valid_rows = [json.loads(x) for x in valid_records.read_text().splitlines()]
valid_features = np.load(out_dir / "valid/features.npy")
positions, layers, width = train_features.shape[1:]


def targets(records) -> np.ndarray:
    """[questions, TARGETS]: c_1..c_max then the gold answer, -1 where invalid."""
    table = np.full((len(records), TARGETS), -1, dtype=np.int64)
    for i, r in enumerate(records):
        for j, value in enumerate(r["values"][:args.max_steps]):
            table[i, j] = -1 if value is None else value
        gold = as_target(r["gold"]) if r["gold"] is not None else None
        table[i, -1] = -1 if gold is None else gold
    return table


def digits(values: np.ndarray) -> torch.Tensor:
    return torch.tensor(np.stack([(values // 10 ** d) % 10 for d in range(DIGITS)], 1), device=device)


def fit(x, values):
    """Probes for every (position, layer) at once on x [rows, positions, layers, width]: (mean, std, weight, bias)."""
    x = torch.tensor(x, device=device, dtype=torch.float32).permute(1, 2, 0, 3).reshape(positions * layers, -1, width)
    mean, std = x.mean(1, keepdim=True), x.std(1, keepdim=True).clamp(min=1e-6)
    x = (x - mean) / std
    weight = torch.zeros(positions * layers, width, DIGITS * 10, device=device, requires_grad=True)
    bias = torch.zeros(positions * layers, 1, DIGITS * 10, device=device, requires_grad=True)
    optimizer = torch.optim.Adam([weight, bias], lr=args.probe_lr, weight_decay=args.weight_decay)
    labels = digits(values).repeat(positions * layers, 1)
    for _ in range(args.probe_steps):
        logits = (torch.bmm(x, weight) + bias).reshape(-1, 10)
        loss = F.cross_entropy(logits, labels.reshape(-1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return mean, std, weight.detach(), bias.detach()


@torch.no_grad()
def read(probe, x) -> np.ndarray:
    """Readings [rows, layers, positions] of x [rows, positions, layers, width]."""
    mean, std, weight, bias = probe
    x = torch.tensor(x, device=device, dtype=torch.float32).permute(1, 2, 0, 3).reshape(positions * layers, -1, width)
    logits = (torch.bmm((x - mean) / std, weight) + bias).reshape(positions * layers, x.shape[1], DIGITS, 10)
    predicted = logits.argmax(-1)
    value = sum(predicted[..., d] * 10 ** d for d in range(DIGITS))
    return value.reshape(positions, layers, -1).permute(2, 1, 0).cpu().numpy().astype(np.int32)


if not (out_dir / "readings.npz").exists():
    began = time.time()
    train_targets, valid_targets = targets(train_records), targets(valid_rows)
    right = np.array([r["correct"] for r in train_records])
    folds = np.array([int(hashlib.sha256(str(r["index"]).encode()).hexdigest(), 16) % args.folds
                      for r in train_records])
    train_readings = np.zeros((len(train_records), TARGETS, layers, positions), dtype=np.int32)
    valid_readings = np.zeros((len(valid_rows), TARGETS, layers, positions), dtype=np.int32)
    torch.manual_seed(args.seed)
    for t in range(TARGETS):
        usable = right & (train_targets[:, t] >= 0)
        if not usable.any():
            continue
        for fold in range(args.folds):
            rows = usable & (folds != fold)
            if rows.any():
                held = folds == fold
                train_readings[held, t] = read(fit(train_features[rows], train_targets[rows, t]),
                                               train_features[held])
        valid_readings[:, t] = read(fit(train_features[usable], train_targets[usable, t]), valid_features)
    np.savez_compressed(out_dir / "readings.npz", train=train_readings, valid=valid_readings,
                        train_targets=train_targets, valid_targets=valid_targets, folds=folds,
                        layers=np.array(LAYERS))
    status["probes"] = {"seconds": time.time() - began, "train_questions": len(train_records)}
    save_status()
print("probes:", json.dumps(status.get("probes")), flush=True)
