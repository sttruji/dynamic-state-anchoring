"""Stage 0 of step-level selection (docs/brainstorm/STEP_SELECTION.md): evaluation only, for one sampling seed.

For each question:
    greedy     the latent steps with dropout off, its seven latents read by SIM-CoT's step decoder; its answers must
               equal the checkpoint's own (report.py --reference)
    chains     --chains independent runs of every step with MC dropout at rate --p, each answer read with dropout off
    branches   for each step s, --candidates - 1 dropout reruns of step s from greedy's state, each finished with dropout
               off; each rerun's new latent is read by the decoder and given the fixed rule's features
    rule       the untrained selector: at each step candidate 0 and --candidates - 1 dropout reruns, one kept by the
               fixed rule (stepsel.choose), continued to an answer

    python scripts/step_select/eval.py --checkpoint <simcot dir> --dataset gsm8k_valid --output <file.jsonl> --seed 1
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", required=True, help="a directory holding pytorch_model.bin (fetch_simcot.py)")
parser.add_argument("--dataset", required=True)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--p", type=float, default=0.2, help="MC-dropout rate for sampled steps")
parser.add_argument("--seed", type=int, default=1)
parser.add_argument("--chains", type=int, default=18)
parser.add_argument("--candidates", type=int, default=4, help="per step, candidate 0 included")
parser.add_argument("--depth", type=int, default=6)
parser.add_argument("--decoder-tokens", type=int, default=32)
parser.add_argument("--answer-tokens", type=int, default=32,
                    help="answer length cap for sampled paths, so one chain without end-of-text cannot stall a batch")
parser.add_argument("--device", default="cuda")
parser.add_argument("--data-dir", type=Path, default=None, help="default: $DSA_DATA_DIR or <repo>/data")
parser.add_argument("--limit", type=int)
args = parser.parse_args()
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch
from dsa.bench import data as bench_data
from dsa.bench.models.codi import CODIGPT2
from dsa.bench.scoring import score
from stepsel import CONSTANTS, choose, features, numbers, pick, read, readout, sample_step, sampled, step
from looped import load_step_decoder

torch.manual_seed(args.seed)
data_dir = args.data_dir or Path(os.environ.get("DSA_DATA_DIR", ROOT / "data"))
rows, _ = bench_data.load_eval_rows(args.dataset, data_dir)
answer_type = bench_data.EVAL_DATASETS[args.dataset]["answer_type"]
codi = CODIGPT2(checkpoint=args.checkpoint, device=args.device)
model, projection, tokenizer, device = codi.model, codi.projection, codi.tokenizer, codi.device
model.eval()
projection.eval()
embed = model.get_input_embeddings()
decoder = load_step_decoder(device)
eos, depth, extra = tokenizer.eos_token_id, args.depth, args.candidates - 1


def answer(tokens, row) -> dict:
    text = tokenizer.decode([t for t in tokens if t < len(tokenizer)], skip_special_tokens=True)
    scored = score(text, row["answer"], row.get("answer_type", answer_type))
    return {"text": text, "prediction": scored["prediction"], "correct": bool(scored["correct"])}


def decoded(latents):
    """Decoder reads of latents [n, 1, width]: (text, mean log-probability) per latent."""
    ids, logprobs = read(decoder, embed, latents[:, 0, :], eos, args.decoder_tokens)
    return [(tokenizer.decode(i), lp) for i, lp in zip(ids, logprobs)]


def finish(latent, cache, question: int, first: int):
    """Steps first..depth with dropout off for every row, then the rows' answer tokens."""
    for s in range(first, depth + 1):
        latent, cache = step(model, projection, latent, cache, question + s)
    return readout(model, codi, cache, question + depth + 1, latent.shape[0], args.answer_tokens)


def majority(entries):
    """The most common prediction (first seen wins a tie) and whether it is correct."""
    votes = Counter(e["prediction"] for e in entries)
    top = max(votes.values())
    winner = next(e for e in entries if votes[e["prediction"]] == top)
    return winner["correct"]


indexed = rows[:args.limit] if args.limit else rows
args.output.parent.mkdir(parents=True, exist_ok=True)
start = time.time()
with torch.no_grad(), args.output.open("w") as sink:
    for number, row in enumerate(indexed, 1):
        encoded = tokenizer.encode(row["question"].strip().replace("  ", " ")) + [codi.bot_id]
        question = len(encoded) - 1  # <bot> sits at this cache index, latent step s at question + s
        out = model(input_ids=torch.tensor([encoded], device=device), use_cache=True, output_hidden_states=True)
        caches, latents = [out.past_key_values], [projection(out.hidden_states[-1][:, -1:, :])]
        base = numbers(row["question"]) | CONSTANTS

        # Greedy, keeping the state before every step. Slot s is the latent step s makes; slot 0 comes from <bot>.
        for s in range(1, depth + 1):
            latent, cache = step(model, projection, latents[-1], caches[-1], question + s)
            latents.append(latent)
            caches.append(cache)
        greedy = answer(readout(model, codi, caches[-1], question + depth + 1, 1)[0], row)
        known, greedy_steps = set(base), []
        for text, lp in decoded(torch.cat(latents, 0)):
            greedy_steps.append(features(text, lp, known))
            if greedy_steps[-1]["result"] is not None:
                known.add(greedy_steps[-1]["result"])

        # Independent chains: every step sampled, from the question's state.
        latent, cache = sample_step(model, projection, latents[0], caches[0], question + 1, args.chains, args.p)
        for s in range(2, depth + 1):
            latent, cache = sampled(model, projection, latent, cache, question + s, args.p)
        chains = [answer(t, row) for t in readout(model, codi, cache, question + depth + 1, args.chains,
                                                  args.answer_tokens)]

        # Branches: greedy except at one step, which is rerun with dropout.
        branches = []
        for s in range(1, depth + 1):
            latent, cache = sample_step(model, projection, latents[s - 1], caches[s - 1], question + s, extra, args.p)
            reads = decoded(latent)
            answers = finish(latent, cache, question, s + 1)
            prior = base | {f["result"] for f in greedy_steps[:s] if f["result"] is not None}
            branches.append([{**answer(t, row), **features(text, lp, prior)} for t, (text, lp) in zip(answers, reads)])

        # The untrained selector: candidate 0 and `extra` reruns at every step, one kept by the fixed rule.
        latent, cache, known, chosen = latents[0], caches[0], set(base), []
        if greedy_steps[0]["result"] is not None:
            known.add(greedy_steps[0]["result"])
        for s in range(1, depth + 1):
            own, own_cache = step(model, projection, latent, cache, question + s)
            rerun, rerun_cache = sample_step(model, projection, latent, cache, question + s, extra, args.p)
            candidates = torch.cat([own, rerun], 0)
            scored = [features(text, lp, known) for text, lp in decoded(candidates)]
            k = choose(scored)
            chosen.append(k)
            latent, cache = candidates[k:k + 1], own_cache if k == 0 else pick(rerun_cache, k - 1)
            if scored[k]["result"] is not None:
                known.add(scored[k]["result"])
        rule = {**answer(readout(model, codi, cache, question + depth + 1, 1, args.answer_tokens)[0], row),
                "chosen": chosen}

        record = {"id": row["id"], "gold": row["answer"],
                  "rationale_steps": len(re.findall(r"<<", row.get("rationale", ""))),
                  "greedy": {**greedy, "steps": greedy_steps},
                  "chains": [{"prediction": c["prediction"], "correct": c["correct"]} for c in chains],
                  "branches": branches, "rule": rule}
        sink.write(json.dumps(record) + "\n")
        sink.flush()
        if number % 50 == 0:
            print(f"{args.dataset} seed {args.seed}: {number}/{len(indexed)} {time.time() - start:.0f}s", flush=True)

records = [json.loads(line) for line in args.output.read_text().splitlines()]
n = len(records)
greedy_hits = sum(r["greedy"]["correct"] for r in records)
chain_cover = sum(r["greedy"]["correct"] or any(c["correct"] for c in r["chains"]) for r in records)
branch_cover = sum(r["greedy"]["correct"] or any(b["correct"] for bs in r["branches"] for b in bs) for r in records)
print(f"{args.dataset} seed {args.seed}, p={args.p}, n={n}: greedy {greedy_hits}; "
      f"greedy or any of {args.chains} chains {chain_cover}; greedy or any of {depth * extra} branches {branch_cover}; "
      f"majority of 4 chains {sum(majority(r['chains'][:4]) for r in records)}; "
      f"untrained selector {sum(r['rule']['correct'] for r in records)}; {time.time() - start:.0f}s", flush=True)
