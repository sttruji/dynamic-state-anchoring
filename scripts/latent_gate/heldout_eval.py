"""Held-out/scale study evaluation: a CODI-style model at several latent depths under the plan's policies, batched.

Plan: results/latent_gate/heldout_scale/PLAN.md; the loops are in batched.py. Policies (budget 6 where one applies):
    plain       every latent entry, read out at every depth
    keep_first  entries 1..6; later steps are still computed, but each new entry is evicted at once
    window      the 6 most recent entries
    random      above 6 entries, a random kept one is evicted (stage1_eval's per-question draw)
    logn        every entry gets -log(n / 6) once there are n > 6
    dedup_soft  every entry gets -log(size of its cluster of fed latents with cosine >= 0.95)
    shift       6 entries, the answer read out at the positions it would have after the deepest depth
    copy_last   6 genuine steps, then exact copies of entry 6's keys and values; copy_pair copies entries 5 and 6 in turn
    gate        the 6 highest-scoring entries (gate.pt); gate_soft: log g on every entry, no eviction

    python scripts/latent_gate/heldout_eval.py --llama codi-llama1b --datasets gsm8k_valid --output-dir <dir>
    python scripts/latent_gate/heldout_eval.py --checkpoint <dir> --datasets gsm8k_test gsm_hard multiarith --output-dir <dir>

Writes <output-dir>/<dataset>.jsonl in stage1_eval's record format (per policy and depth: text, prediction, correct, kept,
plus attention shares with --attention), resuming after the questions already written, and <output-dir>/run.json.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from dsa.bench.models.codi_llama import CHECKPOINTS

parser = argparse.ArgumentParser()
source = parser.add_mutually_exclusive_group()
source.add_argument("--llama", choices=sorted(CHECKPOINTS), help="a published CODI-style LLaMA checkpoint")
source.add_argument("--checkpoint", help="a GPT-2 CODI-format directory (pytorch_model.bin, gate.pt for gated arms); "
                                         "the published CODI GPT-2 if neither is given")
parser.add_argument("--datasets", nargs="+", required=True)
parser.add_argument("--output-dir", type=Path, required=True)
parser.add_argument("--policies", nargs="+", default=["plain", "keep_first", "dedup_soft", "logn", "shift"])
parser.add_argument("--depths", type=int, nargs="+", default=[6, 12, 24])
parser.add_argument("--budget", type=int, default=6)
parser.add_argument("--batch", type=int, default=32)
parser.add_argument("--limit", type=int)
parser.add_argument("--attention", action="store_true", help="record the answer's attention shares on the latents")
parser.add_argument("--max-new-tokens", type=int, default=256)
parser.add_argument("--device", default="cuda")
parser.add_argument("--data-dir", type=Path, default=None, help="default: $DSA_DATA_DIR or <repo>/data")
args = parser.parse_args()

import torch
from batched import POLICIES, Loop, ask, run
from dsa.bench import data as bench_data
from dsa.bench.scoring import score
from gate import Gate, gold_results

unknown = set(args.policies) - set(POLICIES)
assert not unknown, f"unknown policies {unknown}"
if args.llama:
    from dsa.bench.models.codi_llama import CODILlama

    codi = CODILlama(checkpoint=args.llama, max_new_tokens=args.max_new_tokens, device=args.device)
else:
    from dsa.bench.models.codi import CODIGPT2

    codi = CODIGPT2(checkpoint=args.checkpoint, max_new_tokens=args.max_new_tokens, device=args.device)
gate = None
gate_file = Path(args.checkpoint) / "gate.pt" if args.checkpoint else None
if gate_file and gate_file.exists():
    saved = torch.load(gate_file, map_location="cpu", weights_only=True)
    gate = Gate(saved["width"]).to(codi.device).eval()
    gate.load_state_dict(saved["state"])
assert gate is not None or not any(p.startswith("gate") for p in args.policies), "gate policies need a gate.pt"
loop = Loop.of(codi, gate)
data_dir = args.data_dir or Path(os.environ.get("DSA_DATA_DIR", ROOT / "data"))
args.output_dir.mkdir(parents=True, exist_ok=True)
(args.output_dir / "run.json").write_text(json.dumps(
    {"model": codi.identity(), "gate": str(gate_file) if gate else None,
     "args": {k: v for k, v in vars(args).items()}}, default=str, indent=1))

for dataset in args.datasets:
    rows, _ = bench_data.load_eval_rows(dataset, data_dir)
    answer_type = bench_data.EVAL_DATASETS[dataset]["answer_type"]
    indexed = list(enumerate(rows))[:args.limit] if args.limit else list(enumerate(rows))
    target = args.output_dir / f"{dataset}.jsonl"
    done = {json.loads(line)["id"] for line in target.read_text().splitlines()} if target.exists() else set()
    todo = [(i, row) for i, row in indexed if row["id"] not in done]
    start = time.time()
    with torch.no_grad(), target.open("a") as sink:
        for first in range(0, len(todo), args.batch):
            chunk = todo[first:first + args.batch]
            asked = ask(loop, [loop.encode(row["question"]) for _, row in chunk], codi.device)
            seeds = [i for i, _ in chunk]
            records = [{"id": row["id"], "gold": row["answer"], "rationale_steps": len(gold_results(row.get("rationale", "")))}
                       for _, row in chunk]
            for policy in args.policies:
                cells = run(loop, policy, asked, args.depths, args.budget, seeds, attention=args.attention)
                for record, (_, row), per_depth in zip(records, chunk, cells):
                    record[policy] = {}
                    for depth, cell in per_depth.items():
                        text = loop.decode(cell.pop("tokens"))
                        scored = score(text, row["answer"], row.get("answer_type", answer_type))
                        record[policy][depth] = {"text": text, "prediction": scored["prediction"],
                                                 "correct": bool(scored["correct"])} | cell
            for record in records:
                sink.write(json.dumps(record) + "\n")
            sink.flush()
            print(f"{dataset}: {len(done) + first + len(chunk)}/{len(indexed)} {time.time() - start:.0f}s", flush=True)
    records = [json.loads(line) for line in target.read_text().splitlines()]
    print(f"{dataset}, n={len(records)}, correct (4+ rationale steps in brackets):")
    for policy in args.policies:
        cells = []
        for depth in args.depths:
            key = str(depth)
            if records and key in records[0].get(policy, {}):
                hits = [r[policy][key]["correct"] for r in records]
                long = [r[policy][key]["correct"] for r in records if r["rationale_steps"] >= 4]
                cells.append(f"{depth}: {sum(hits)} ({sum(long)}/{len(long)})")
        print(f"  {policy:10} " + "  ".join(cells), flush=True)
