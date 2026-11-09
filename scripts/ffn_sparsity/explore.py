"""Exploratory reads of SPARSE-0, never gating (results/ffn_sparsity/PLAN.md, "Reported, never gating").

    python scripts/ffn_sparsity/explore.py <run dir>

Scores by step count, the paired loss of each 25% oracle arm on long problems, and how the static arms fail.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("run", type=Path)
args = parser.parse_args()
ARMS = ("ID", "O50", "O25", "S50", "S25", "O25-T", "O25-Q")
evals = {name: {r["id"]: r for r in map(json.loads, (args.run / f"evals/{name}.jsonl").read_text().splitlines())}
         for name in ARMS}


def sign_p(wins: int, losses: int) -> float:
    n = wins + losses
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(min(wins, losses) + 1)) / 2 ** n) if n else 1.0


print("Right answers by the rationale's step count")
print(f"  {'steps':6} {'n':>4} " + " ".join(f"{name:>6}" for name in ARMS))
for low, high, label in ((0, 2, "0-2"), (3, 3, "3"), (4, 4, "4"), (5, 99, "5+")):
    ids = [i for i, r in evals["ID"].items() if low <= r["rationale_steps"] <= high]
    print(f"  {label:6} {len(ids):4} " + " ".join(f"{sum(evals[a][i]['correct'] for i in ids):6}" for a in ARMS))

print("\nOn problems with 4 or more steps, each 25% oracle arm against ID")
long = [i for i, r in evals["ID"].items() if r["rationale_steps"] >= 4]
for name in ("O25", "O25-T", "O25-Q"):
    wins = sum(evals[name][i]["correct"] and not evals["ID"][i]["correct"] for i in long)
    losses = sum(evals["ID"][i]["correct"] and not evals[name][i]["correct"] for i in long)
    print(f"  {name}: {wins - losses:+d} (+{wins}/-{losses}, p={sign_p(wins, losses):.3f}) of {len(long)}")

print("\nHow the static arms fail: answers that run to the 256-token cap, and right answers among the rest")
for name in ("ID", "S50", "S25"):
    rows = evals[name].values()
    capped = sum(r["positions"][2] >= 256 for r in rows)
    print(f"  {name}: {capped} capped; {sum(r['correct'] for r in rows if r['positions'][2] < 256)} right of "
          f"{len(rows) - capped} uncapped")

print("\nPositions per question under ID: question tokens, thinking positions, answer positions")
counts = [r["positions"] for r in evals["ID"].values()]
q, t, a = (sum(c[k] for c in counts) / len(counts) for k in range(3))
print(f"  {q:.1f} / {t:.1f} / {a:.1f}: thinking is {t / (q + t + a):.1%} of positions, reading {q / (q + t + a):.1%}")
