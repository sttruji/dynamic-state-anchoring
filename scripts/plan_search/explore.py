"""Exploratory reads of Stage 0 of H, never gating (results/plan_search/stage0/PLAN.md, "Reported, never gating").

    python scripts/plan_search/explore.py <stage0 output dir>

Asks whether gold still beats the wrong target once their starting log-probabilities are matched, since the plan's
matching is loose where no integer near gold is as likely as gold.
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("run", type=Path)
args = parser.parse_args()

searched = [json.loads(x) for x in (args.run / "search.jsonl").read_text().splitlines()]
gap = [r["base_logp_gold"] - r["base_logp_wrong"] for r in searched]


def sign_p(a, b):
    """Exact two-sided sign test of a against b."""
    m = a + b
    return min(1.0, 2 * sum(math.comb(m, i) for i in range(min(a, b) + 1)) / 2 ** m) if m else 1.0


def paired(rows, label):
    gold = sum(r["gold"]["accepted"] for r in rows)
    wrong = sum(r["wrong"]["accepted"] for r in rows)
    only_gold = sum(r["gold"]["accepted"] and not r["wrong"]["accepted"] for r in rows)
    only_wrong = sum(r["wrong"]["accepted"] and not r["gold"]["accepted"] for r in rows)
    ratio = f"{gold / wrong:.2f}" if wrong else "inf"
    print(f"  {label}: n={len(rows)}, gold {gold} ({gold / max(1, len(rows)):.0%}), wrong {wrong} "
          f"({wrong / max(1, len(rows)):.0%}), ratio {ratio}; gold only {only_gold}, wrong only {only_wrong}, "
          f"sign p={sign_p(only_gold, only_wrong):.2g}")


print(f"Starting log-prob gap, gold minus wrong: median {statistics.median(gap):+.2f}; within 0.5 nats on "
      f"{sum(abs(g) <= 0.5 for g in gap)} of {len(searched)}; wrong more likely on {sum(g < 0 for g in gap)}")
print("Gold against wrong, by how closely the wrong target was matched:")
paired(searched, "all")
for limit in (0.25, 0.5, 1.0, 2.0):
    paired([r for r, g in zip(searched, gap) if abs(g) <= limit], f"gap within {limit}")
paired([r for r, g in zip(searched, gap) if g <= 0], "wrong at least as likely")
paired([r for r, g in zip(searched, gap) if g > 2], "gold over 2 nats more likely")

print("Acceptance by the target's own starting log-prob:")
edges = [-math.inf, -8, -6, -5, -4, -3, -2, -1, 0.01]
for low, high in zip(edges, edges[1:]):
    cells = []
    for name in ("gold", "wrong"):
        hits = [r[name]["accepted"] for r in searched if low <= r[f"base_logp_{name}"] < high]
        cells.append(f"{name} {sum(hits)}/{len(hits)} ({sum(hits) / len(hits):.0%})" if hits else f"{name} -")
    print(f"  [{low:g}, {high:g}): " + "; ".join(cells))

print("Gold against wrong, by the size of the gold answer:")
for low, high in ((0, 10), (10, 100), (100, 1000), (1000, math.inf)):
    paired([r for r in searched if low <= r["gold_answer"] < high], f"gold in [{low}, {high:g})")

print("Rejected plans:")
rejected = [r for r in searched if not r["gold"]["accepted"]]
print(f"  gold: {len(rejected)}, of which {sum(r['gold']['prediction'] == r['greedy_prediction'] for r in rejected)} "
      f"still give greedy's answer")
rejected = [r for r in searched if not r["wrong"]["accepted"]]
print(f"  wrong: {len(rejected)}, of which {sum(r['wrong']['prediction'] == r['greedy_prediction'] for r in rejected)} "
      f"give greedy's answer and {sum(r['wrong']['prediction'] == str(r['gold_answer']) for r in rejected)} give gold")
print(f"Median |wrong target - gold|: {statistics.median(abs(r['wrong']['target'] - r['gold_answer']) for r in searched)}")
