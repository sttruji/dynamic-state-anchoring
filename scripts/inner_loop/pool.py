"""Pool two seeds of Stage 1b under the two-seed rule written before seed 8 ran (results/inner_loop/README.md).

    python scripts/inner_loop/pool.py results/inner_loop/stage1b results/inner_loop/stage1b-seed8

GO needs all three: loop_ce@k3 beats ft6_ce by +7 or more per seed on average with p < 0.05 on the pooled paired sign test;
k3 >= k1 within loop_ce in each seed; and loop_ce@k3 pooled at least ft6_step. Anything else is STOP.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("runs", type=Path, nargs=2, help="two Stage 1b folders, each holding evals/<arm>/gsm8k_valid.jsonl")
args = parser.parse_args()


def load(folder, arm, key):
    rows = [json.loads(x) for x in (folder / "evals" / arm / "gsm8k_valid.jsonl").read_text().splitlines()]
    return {r["id"]: bool(r["arms"][key]["correct"]) for r in rows}


def sign_p(wins: int, losses: int) -> float:
    """Exact two-sided sign test."""
    n = wins + losses
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(min(wins, losses) + 1)) / 2 ** n) if n else 1.0


def compare(label, first, second):
    """Paired over every (seed, question): the mean difference per seed, its sign-test p, and each seed's difference."""
    wins = losses = 0
    diffs, parts = [], []
    for folder in args.runs:
        a, b = load(folder, *first), load(folder, *second)
        assert a.keys() == b.keys(), f"{folder}: the two arms scored different questions"
        w, l = sum(a[i] and not b[i] for i in a), sum(b[i] and not a[i] for i in a)
        wins, losses = wins + w, losses + l
        diffs.append(w - l)
        parts.append(f"{folder.name} {sum(a.values())} vs {sum(b.values())} ({w - l:+d}, +{w}/-{l})")
    mean, p = (wins - losses) / len(args.runs), sign_p(wins, losses)
    print(f"  {label}: {'; '.join(parts)}; pooled {mean:+.1f} per seed (+{wins}/-{losses}, p={p:.3f})")
    return mean, p, diffs


k3, k1, ce, step = ("loop_ce", "loop3"), ("loop_ce", "loop1"), ("ft6_ce", "plain"), ("ft6_step", "plain")
print(f"Stage 1b pooled over {', '.join(str(r) for r in args.runs)}")
gain, p, _ = compare("1. loop_ce k3 vs ft6_ce", k3, ce)
_, _, dose = compare("2. loop_ce k3 vs k1", k3, k1)
best, _, _ = compare("3. loop_ce k3 vs ft6_step", k3, step)
compare("recipe: ft6_ce vs ft6_step", ce, step)
checks = {"1 (+7 per seed, p < 0.05)": gain >= 7 and p < 0.05, "2 (k3 >= k1 in each seed)": all(d >= 0 for d in dose),
          "3 (at least ft6_step)": best >= 0}
print("  " + "; ".join(f"rule {name}: {'holds' if ok else 'fails'}" for name, ok in checks.items()))
print(f"  {'GO' if all(checks.values()) else 'STOP'}")
