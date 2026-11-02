"""Free checks on the CODI step sweep: ceiling, agreement between depths as a confidence signal, and voting.

    python scripts/codi_trail/sweep_signals.py results/codi_trail/sweep
"""
import json, math, sys
from collections import Counter
from itertools import combinations
from pathlib import Path

depth = Path(sys.argv[1])
STEPS = (1, 3, 6, 12, 24)

def load(n):
    lines = (depth / "predictions" / f"codi-iter{n}" / "gsm8k_valid.jsonl").read_text().splitlines()
    return {r["id"]: r for r in map(json.loads, lines[1:])}

def num(t):
    try:
        return float(str(t).replace(",", ""))
    except ValueError:
        return None

def sign_p(x, y):
    n, k = x + y, min(x, y)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n) if n else 1.0

def fisher(a, b, c, d):
    n1, n2, t = a + b, c + d, a + c
    prob = lambda x: math.comb(n1, x) * math.comb(n2, t - x) / math.comb(n1 + n2, t)
    base = prob(a)
    return sum(p for p in (prob(x) for x in range(max(0, t - n2), min(t, n1) + 1)) if p <= base * (1 + 1e-9))

rows = {n: load(n) for n in STEPS}
ids = list(rows[6])
gold = {i: rows[6][i]["gold"] for i in ids}

print("ceiling: right at 6 or at any other depth (needs an oracle to pick; bound only)")
for group in ((6, 12), (6, 3), (3, 6, 12), (3, 6, 12, 24), (1, 3, 6, 12, 24)):
    print(" ", group, len({i for n in group for i in ids if rows[n][i]["correct"]}))
print("  right at all of 3, 6, 12, 24:", sum(all(rows[n][i]["correct"] for n in (3, 6, 12, 24)) for i in ids))
print("  lost by 24 but right at 6, and gained by 24:", sum(rows[6][i]["correct"] and not rows[24][i]["correct"] for i in ids), sum(rows[24][i]["correct"] and not rows[6][i]["correct"] for i in ids))

print("\nagreement as a signal, scored on the 6-step answer")
for a, b in ((6, 12), (3, 6), (6, 24)):
    same = [i for i in ids if rows[a][i]["prediction"] == rows[b][i]["prediction"]]
    diff = [i for i in ids if rows[a][i]["prediction"] != rows[b][i]["prediction"]]
    x, y = sum(rows[6][i]["correct"] for i in same), sum(rows[6][i]["correct"] for i in diff)
    print(f"  {a} vs {b}: agree {x}/{len(same)} = {100*x/len(same):.1f}%  disagree {y}/{len(diff)} = {100*y/len(diff):.1f}%  Fisher p={fisher(x, len(same)-x, y, len(diff)-y):.2g}")
for label, group in (("3=6=12", (3, 6, 12)), ("3=6=12=24", (3, 6, 12, 24))):
    same = [i for i in ids if len({rows[n][i]["prediction"] for n in group}) == 1]
    diff = [i for i in ids if i not in set(same)]
    x, y = sum(rows[6][i]["correct"] for i in same), sum(rows[6][i]["correct"] for i in diff)
    print(f"  {label}: all agree {x}/{len(same)} = {100*x/len(same):.1f}%  not all {y}/{len(diff)} = {100*y/len(diff):.1f}%  Fisher p={fisher(x, len(same)-x, y, len(diff)-y):.2g}")
    print(f"     answering only those {len(same)} ({100*len(same)/500:.0f}% coverage) gives {100*x/len(same):.1f}% precision vs 41.2% answering all")

print("\nvoting, ties broken toward the depth closest to 6")
pref = {n: abs(math.log2(n / 6)) for n in STEPS}
for group in ((3, 6, 12), (3, 6, 24), (6, 12, 24), (3, 6, 12, 24), (1, 3, 6, 12, 24)):
    right = changed_good = changed_bad = 0
    for i in ids:
        votes = Counter(rows[n][i]["prediction"] for n in group)
        top = max(votes.values())
        pick = min((n for n in group if votes[rows[n][i]["prediction"]] == top), key=pref.__getitem__)
        answer = rows[pick][i]["prediction"]
        good = num(answer) == num(gold[i])
        right += good
        changed_good += good and not rows[6][i]["correct"]
        changed_bad += rows[6][i]["correct"] and not good
    print(f"  {group}: {right} ({100*right/500:.1f}%)  gained {changed_good}, lost {changed_bad}, p={sign_p(changed_good, changed_bad):.3f}")
