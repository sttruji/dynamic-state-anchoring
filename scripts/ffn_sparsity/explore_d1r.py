"""Exploratory reads of D1-R, never gating (results/ffn_sparsity/d1r/PLAN.md, "Reported, never gating").

    python scripts/ffn_sparsity/explore_d1r.py <run dir>

MoE at seed 7 against the dense references, by step count, the training-free arms on the first 100 questions, the
training losses against ft6_step's, and the router's held-out recall at conversion.
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
parser = argparse.ArgumentParser()
parser.add_argument("run", type=Path)
args = parser.parse_args()


def load(path):
    return {r["id"]: r for r in map(json.loads, Path(path).read_text().splitlines())}


def sign_p(wins: int, losses: int) -> float:
    n = wins + losses
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(min(wins, losses) + 1)) / 2 ** n) if n else 1.0


moe = load(args.run / "evals/moe-s7.jsonl")
base = load(args.run / "evals/base-dense.jsonl")
ft6 = {r["id"]: {"correct": r["arms"]["plain"]["correct"]}
       for r in map(json.loads, (ROOT / "results/inner_loop/stage1b/evals/ft6_step/gsm8k_valid.jsonl").open())}
o50 = load(ROOT / "results/ffn_sparsity/run/evals/O50.jsonl")
s50 = load(ROOT / "results/ffn_sparsity/run/evals/S50.jsonl")
print("MoE-50 fine-tuned at seed 7 (179) against:")
for name, other in (("ft6_step at seed 7", ft6), ("the base model", base), ("SPARSE-0's O50, untrained", o50)):
    wins = sum(moe[i]["correct"] and not other[i]["correct"] for i in moe)
    losses = sum(other[i]["correct"] and not moe[i]["correct"] for i in moe)
    print(f"  {name} ({sum(r['correct'] for r in other.values())}): {wins - losses:+d} (+{wins}/-{losses}, "
          f"p={sign_p(wins, losses):.4f})")
print(f"  answers at the 256-token cap: {sum(r['positions'][2] >= 256 for r in moe.values())}")

print("\nRight answers by the rationale's step count: MoE-50 at seed 7 | ft6_step at seed 7 | base")
for low, high, label in ((0, 2, "0-2"), (3, 3, "3"), (4, 4, "4"), (5, 99, "5+")):
    ids = [i for i, r in base.items() if low <= r["rationale_steps"] <= high]
    print(f"  {label:4} n={len(ids):3}: " + " | ".join(str(sum(x[i]["correct"] for i in ids)) for x in (moe, ft6, base)))

first = list(base)[:100]
print("\nTraining-free, on the first 100 questions (base, SPARSE-0's O50 and S50 for reference):")
for name, rows in (("base", base), ("O50", o50), ("S50", s50)):
    print(f"  {name}: {sum(rows[i]['correct'] for i in first)}")
for name in ("moe-0", "hash-0", "prune-0"):
    rows = load(args.run / f"evals/{name}.jsonl")
    print(f"  {name}: {sum(r['correct'] for r in rows.values())}, "
          f"{sum(r['positions'][2] >= 256 for r in rows.values())} at the token cap")

ours = [json.loads(x) for x in (args.run / "training/moe-s7/metrics.jsonl").read_text().splitlines()]
dense = {r["update"]: r for r in map(json.loads, (ROOT / "results/inner_loop/stage1b/training/ft6_step/metrics.jsonl").open())}
print("\nTraining losses, MoE-50 at seed 7 against ft6_step at seed 7 (answer CE / step loss):")
for row in ours:
    if row["update"] in (0, 100, 200, 400, 600, 767) and row["update"] in dense:
        other = dense[row["update"]]
        print(f"  update {row['update']:3}: {row['ce']:.3f} / {row['step']:.3f} against {other['ce']:.3f} / {other['step']:.3f}")
late = [r for r in ours if r["update"] >= 600]
late_dense = [r for r in dense.values() if r["update"] >= 600]
print(f"  mean from update 600: {statistics.mean(r['ce'] for r in late):.3f} / {statistics.mean(r['step'] for r in late):.3f}"
      f" against {statistics.mean(r['ce'] for r in late_dense):.3f} / {statistics.mean(r['step'] for r in late_dense):.3f}")

conversion = json.loads((args.run / "conversion.json").read_text())
print("\nRouter recall of the true top 4 on held-out calibration positions, per layer (Q / T / A; chance 0.33):")
for layer, entry in conversion["layers"].items():
    recall = entry["recall"]
    print(f"  layer {int(layer) + 1:2}: {recall['Q']:.2f} / {recall['T']:.2f} / {recall['A']:.2f}")
