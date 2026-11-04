"""Apply results/latent_gate/heldout_scale/PLAN.md to the held-out/scale evaluations.

    python scripts/latent_gate/heldout_report.py <folder with <model>/<dataset>.jsonl>

Models are folders such as simcot-gpt2, codi-gpt2, codi-llama1b, simcot-llama1b, simcot-llama3b, and the GPT-2 arms
<base>-gate12_fixed and <base>-ft6. Tests are exact two-sided paired sign tests; intervals are 95% Wald intervals on the
paired accuracy difference, in points.
"""
import json
import math
import sys
from pathlib import Path

folder = Path(sys.argv[1])
PRIMARY = ("gsm8k_test", "gsm_hard", "multiarith")
SECONDARY = ("svamp",)
MARGIN = 2.0
BASES = ("simcot-gpt2", "codi-gpt2", "codi-llama1b", "simcot-llama1b", "simcot-llama3b")


def sign_p(x, y):
    n, k = x + y, min(x, y)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n) if n else 1.0


def load(model, datasets):
    rows = []
    for dataset in datasets:
        path = folder / model / f"{dataset}.jsonl"
        if not path.exists():
            return None
        rows += [json.loads(line) for line in path.read_text().splitlines()]
    return rows


def cell(rows, policy, depth):
    if rows is None or depth not in rows[0].get(policy, {}):
        return None
    return [r[policy][depth]["correct"] for r in rows]


def compare(label, a, b, subset=None):
    if a is None or b is None:
        print(f"  {label}: missing")
        return None
    pairs = [(x, y) for i, (x, y) in enumerate(zip(a, b)) if subset is None or subset[i]]
    n = len(pairs)
    x = sum(p and not q for p, q in pairs)
    y = sum(q and not p for p, q in pairs)
    diff = (x - y) / n
    half = 1.96 * math.sqrt(max(0.0, x + y - (x - y) ** 2 / n)) / n
    result = {"x": x, "y": y, "p": sign_p(x, y), "diff": 100 * diff, "low": 100 * (diff - half), "high": 100 * (diff + half)}
    print(f"  {label}: {sum(p for p, _ in pairs)} vs {sum(q for _, q in pairs)} of {n}; +{x}/-{y}, p={result['p']:.3g}, "
          f"difference {result['diff']:+.1f} points [{result['low']:+.1f}, {result['high']:+.1f}]")
    return result


def table(model):
    for dataset in ("gsm8k_valid",) + PRIMARY + SECONDARY:
        rows = load(model, (dataset,))
        if rows is None:
            continue
        print(f"  {dataset} (n={len(rows)})")
        for policy in [p for p in rows[0] if isinstance(rows[0][p], dict)]:
            cells = [f"{d}: {sum(r[policy][d]['correct'] for r in rows):4}" for d in sorted(rows[0][policy], key=int)]
            print(f"    {policy:10} " + "  ".join(cells))


drift_and_fix = {}
print(f"Primary pool: {' + '.join(PRIMARY)}; secondary: {', '.join(SECONDARY)}; noninferiority margin {MARGIN} points")
for model in [m for m in BASES if (folder / m).exists()]:
    print(f"\n== {model}")
    table(model)
    for pool, datasets in (("primary", PRIMARY), ("secondary", SECONDARY)):
        rows = load(model, datasets)
        if rows is None:
            print(f" {pool}: not scored")
            continue
        print(f" {pool} pool, n={len(rows)}")
        d1 = compare("D1 plain@24 vs plain@6", cell(rows, "plain", "24"), cell(rows, "plain", "6"))
        drift = bool(d1) and d1["y"] > d1["x"] and d1["p"] < 0.05
        if d1:
            print(f"  => D1 {'holds' if drift else 'fails'}")
        fixes = {}
        for name, policy in (("D2", "dedup_soft"), ("D3", "keep_first")):
            beats = compare(f"{name} {policy}@24 vs plain@24", cell(rows, policy, "24"), cell(rows, "plain", "24"))
            holds = compare(f"{name} {policy}@24 vs plain@6", cell(rows, policy, "24"), cell(rows, "plain", "6"))
            if beats and holds:
                fixes[name] = beats["x"] > beats["y"] and beats["p"] < 0.05 and holds["low"] > -MARGIN
                print(f"  => {name} {'holds' if fixes[name] else 'fails'}")
        d4 = compare("D4 shift@6 vs plain@6", cell(rows, "shift", "6"), cell(rows, "plain", "6"))
        if d4:
            print(f"  => D4: positions {'are a confound to report' if d4['diff'] < -MARGIN else 'cost under 2 points'}")
        gains = [compare(f"G dedup_soft@{d} vs plain@6", cell(rows, "dedup_soft", d), cell(rows, "plain", "6")) for d in ("12", "24")]
        if all(gains):
            gain = any(g["x"] > g["y"] and g["p"] < 0.025 for g in gains)
            print(f"  => G {'passes: a free gain' if gain else 'fails'}")
        if pool == "primary" and d1 and "D2" in fixes:
            drift_and_fix[model] = (drift, fixes["D2"], all(gains) and gain)
    if "llama" in model:
        valid = load(model, ("gsm8k_valid",))
        if valid and "12" in valid[0].get("dedup_soft", {}):
            long = [r["rationale_steps"] >= 4 for r in valid]
            dedup, plain = cell(valid, "dedup_soft", "12"), cell(valid, "plain", "6")
            later = [j for r in valid for j in r["dedup_soft"]["12"].get("repeats", []) if j > 6]
            share = len(later) / (6 * len(valid))
            ok = sum(d for d, l in zip(dedup, long) if l) >= sum(p for p, l in zip(plain, long) if l)
            print(f" Part C trigger, second route (gsm8k_valid): dedup_soft@12 on 4+ problems "
                  f"{sum(d for d, l in zip(dedup, long) if l)} vs plain@6 {sum(p for p, l in zip(plain, long) if l)}; "
                  f"entries 7-12 that are near-copies of an earlier one: {share:.0%} => {'met' if ok and share < 0.5 else 'not met'}")

for base in ("simcot-gpt2", "codi-gpt2"):
    gated, ft6 = folder / f"{base}-gate12_fixed", folder / f"{base}-ft6"
    if not gated.exists():
        continue
    print(f"\n== {base}-gate12_fixed (A2)")
    table(f"{base}-gate12_fixed")
    rows, reference, plain = load(f"{base}-gate12_fixed", PRIMARY), load(f"{base}-ft6", PRIMARY), load(base, PRIMARY)
    held = compare("gate@24 vs gate@12", cell(rows, "gate", "24"), cell(rows, "gate", "12"))
    beats = compare("gate@24 vs ft6 plain@24", cell(rows, "gate", "24"), cell(reference, "plain", "24"))
    compare("gate@24 vs base dedup_soft@24 (descriptive)", cell(rows, "gate", "24"), cell(plain, "dedup_soft", "24"))
    if held and beats:
        ok = held["low"] > -MARGIN and beats["x"] > beats["y"] and beats["p"] < 0.05
        print(f"  => the fixed gate {'holds depth and beats ft6 at 24' if ok else 'fails its rule'}")

print("\nWhat the paper may claim (primary pool)")
for model, (drift, fix, gain) in drift_and_fix.items():
    print(f"  {model}: drift {'yes' if drift else 'no'}, soft dedup fix {'holds' if fix else 'fails'}, free gain {'yes' if gain else 'no'}")
drifting = [m for m, (d, _, _) in drift_and_fix.items() if d]
print(f"  The fix generalizes: {'yes' if drifting and all(drift_and_fix[m][1] for m in drifting) else 'no' if drifting else 'no model drifts'}")
print(f"  It holds across scale: {'yes' if any('llama' in m and d and f for m, (d, f, _) in drift_and_fix.items()) else 'not shown'}")
