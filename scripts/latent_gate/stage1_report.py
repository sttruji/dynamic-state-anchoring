"""Apply results/latent_gate/stage1/PLAN.md to the Stage 1 evaluations.

    python scripts/latent_gate/stage1_report.py <folder with <arm>/gsm8k_valid.jsonl, including base/>
"""
import json
import math
import sys
from pathlib import Path

folder = Path(sys.argv[1])
ARMS = ("base", "ft6", "ft6x2", "ft12", "ftrand", "window12", "rand12", "gate12", "gate12_hard", "gate12_label")
ALPHA = 0.025  # H2 is tested for gate12 and gate12_hard, Bonferroni over the two


def sign_p(x, y):
    n, k = x + y, min(x, y)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n) if n else 1.0


runs = {}
for arm in ARMS:
    path = folder / arm / "gsm8k_valid.jsonl"
    if path.exists():
        runs[arm] = [json.loads(line) for line in path.read_text().splitlines()]
ids = None
for arm, rows in runs.items():
    order = [r["id"] for r in rows]
    assert ids is None or order == ids, f"{arm} covers different questions"
    ids = order
if not runs:
    raise SystemExit(f"no evaluations under {folder}")
long = [r["rationale_steps"] >= 4 for r in next(iter(runs.values()))]
print(f"gsm8k_valid, n={len(ids)}, {sum(long)} with 4+ rationale steps. Correct: all (4+)")
for arm, rows in runs.items():
    for policy in [p for p in ("plain", "window", "random", "logn", "logscale", "dedup_soft", "gate", "gate_soft") if p in rows[0]]:
        cells = []
        for depth in ("6", "12", "24"):
            if depth in rows[0][policy]:
                hits = [r[policy][depth]["correct"] for r in rows]
                cells.append(f"{depth}: {sum(hits):3} ({sum(h for h, l in zip(hits, long) if l):2})")
        print(f"  {arm:12} {policy:10} " + "  ".join(cells))


def cell(arm, policy, depth):
    return [r[policy][depth]["correct"] for r in runs[arm]] if arm in runs and depth in runs[arm][0].get(policy, {}) else None


def compare(label, a, b, subset=None):
    if a is None or b is None:
        print(f"  {label}: missing")
        return None
    pairs = [(x, y) for x, y, l in zip(a, b, long) if subset is None or l == subset]
    x = sum(p and not q for p, q in pairs)
    y = sum(q and not p for p, q in pairs)
    p = sign_p(x, y)
    print(f"  {label}: {sum(p_ for p_, _ in pairs)} vs {sum(q for _, q in pairs)}; only first {x}, only second {y}, p={p:.3g}")
    return x, y, p


print("\nDecision rules (PLAN.md)")
gated = [arm for arm in ("gate12", "gate12_hard") if arm in runs]
for best in gated:
    print(f" {best}, at p < {ALPHA}")
    main = compare(f"H2: {best} gate@12 vs ft6 plain@6, 4+ problems", cell(best, "gate", "12"), cell("ft6", "plain", "6"), True)
    overall = compare(f"H2: {best} gate@12 vs ft6 plain@6, all", cell(best, "gate", "12"), cell("ft6", "plain", "6"))
    hold = compare(f"H2: {best} gate@24 vs gate@12, all", cell(best, "gate", "24"), cell(best, "gate", "12"))
    if main and overall and hold:
        passed = main[0] > main[1] and main[2] < ALPHA and overall[0] >= overall[1] and not (hold[1] > hold[0] and hold[2] < 0.05)
        print(f"  => H2 {'PASSES' if passed else 'fails'} for {best}")
    compare(f"compute: {best} gate@12 vs ft6x2 plain@6, 4+ problems", cell(best, "gate", "12"), cell("ft6x2", "plain", "6"), True)
    for rival, policy in (("ft12", "plain"), ("ftrand", "plain"), ("window12", "window"), ("rand12", "random"),
                          ("base", "logn"), ("base", "logscale"), ("base", "dedup_soft")):
        compare(f"rival: {best} gate@12 vs {rival} {policy}@12, 4+ problems", cell(best, "gate", "12"), cell(rival, policy, "12"), True)
    compare(f"cost of hard eviction: {best} gate@12 vs gate_soft@12, all", cell(best, "gate", "12"), cell(best, "gate_soft", "12"))
if len(gated) == 2:
    compare("soft vs hard training: gate12_hard gate@12 vs gate12 gate@12, all", cell("gate12_hard", "gate", "12"), cell("gate12", "gate", "12"))
print("\nStop rule: does any 12-step arm beat ft6 at 6 on the 4+ problems?")
beats = []
for arm, policy in (("ft12", "plain"), ("ftrand", "plain"), ("window12", "window"), ("rand12", "random"),
                    ("gate12", "gate"), ("gate12_hard", "gate"), ("gate12_label", "gate")):
    result = compare(f"{arm} {policy}@12 vs ft6 plain@6", cell(arm, policy, "12"), cell("ft6", "plain", "6"), True)
    if result and result[0] > result[1] and result[2] < 0.05:
        beats.append(arm)
print(f"  => {'continue: ' + ', '.join(beats) if beats else 'STOP: no 12-step arm beats ft6 on the 4+ problems'}")
