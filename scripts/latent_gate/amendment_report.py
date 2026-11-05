"""Apply Amendment 1 of results/latent_gate/heldout_scale/PLAN.md: re-anchoring at 1B (B3) and the untriggered Part C.

    python scripts/latent_gate/amendment_report.py <folder> [--verdict <json>]

B3 reads <model>-anchor/<dataset>.jsonl (`anchor` and `dedup_anchor` at 12 and 24) and pairs it, question by question, with
Part B's records in <model>/. Part C reads <model>-{ft6,ft12,gate12_fixed}/gsm8k_valid.jsonl, and the held-out sets of an arm
that passes. With --verdict, the arms that pass Part C's rule are written there as JSON. Tests are exact two-sided paired sign
tests; intervals are 95% Wald intervals on the paired accuracy difference, in points.
"""
import argparse
import json
import math
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("folder", type=Path)
parser.add_argument("--verdict", type=Path)
args = parser.parse_args()
folder = args.folder
PRIMARY = ("gsm8k_test", "gsm_hard", "multiarith")
SECONDARY = ("svamp",)
MARGIN = 2.0
LLAMAS = ("codi-llama1b", "simcot-llama1b", "simcot-llama3b")


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


def joined(model, datasets):
    """Part B's records for `model` with the re-anchoring policies added, paired by question."""
    base, anchored = load(model, datasets), load(f"{model}-anchor", datasets)
    if base is None or anchored is None:
        return None
    assert [r["id"] for r in base] == [r["id"] for r in anchored], f"{model}: the runs cover different questions"
    return [b | {k: v for k, v in a.items() if isinstance(v, dict)} for b, a in zip(base, anchored)]


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


def table(rows, label, long=None):
    print(f"  {label} (n={len(rows)})")
    for policy in [p for p in rows[0] if isinstance(rows[0][p], dict)]:
        cells = []
        for depth in sorted(rows[0][policy], key=int):
            hits = [r[policy][depth]["correct"] for r in rows]
            extra = f" ({sum(h for h, l in zip(hits, long) if l)})" if long else ""
            cells.append(f"{depth}: {sum(hits):4}{extra}")
        print(f"    {policy:12} " + "  ".join(cells))


print("Amendment 1. B3: re-anchoring at 1B; noninferiority margin 2 points")
anchored_fix = {}
for model in [m for m in LLAMAS if (folder / f"{m}-anchor").exists()]:
    print(f"\n== {model}")
    for dataset in ("gsm8k_valid",) + PRIMARY + SECONDARY:
        rows = joined(model, (dataset,))
        if rows:
            table(rows, dataset)
    for pool, datasets in (("primary", PRIMARY), ("secondary", SECONDARY)):
        rows = joined(model, datasets)
        if rows is None:
            print(f" {pool}: not scored")
            continue
        print(f" {pool} pool, n={len(rows)}")
        later = [max(r["dedup_anchor"]["24"]["slots"]) - 6 for r in rows if "slots" in r["dedup_anchor"]["24"]]
        if later:
            print(f"  positions used after entry 6 at 24 steps (dedup_anchor): mean {sum(later) / len(later):.2f} of 18")
        rules = {}
        for name, policy in (("R1", "dedup_anchor"), ("R2", "anchor")):
            beats = compare(f"{name} {policy}@24 vs plain@24", cell(rows, policy, "24"), cell(rows, "plain", "24"))
            holds = compare(f"{name} {policy}@24 vs plain@6", cell(rows, policy, "24"), cell(rows, "plain", "6"))
            if beats and holds:
                rules[name] = beats["x"] > beats["y"] and beats["p"] < 0.05 and holds["low"] > -MARGIN
                print(f"  => {name} {'holds' if rules[name] else 'fails'}")
        gains = [compare(f"G2 dedup_anchor@{d} vs plain@6", cell(rows, "dedup_anchor", d), cell(rows, "plain", "6"))
                 for d in ("12", "24")]
        if all(gains):
            gain = any(g["x"] > g["y"] and g["p"] < 0.025 for g in gains)
            print(f"  => G2 {'passes: a free gain' if gain else 'fails'}")
        compare("descriptive: dedup_anchor@24 vs dedup_soft@24 (what moving the answer adds)",
                cell(rows, "dedup_anchor", "24"), cell(rows, "dedup_soft", "24"))
        compare("descriptive: anchor@24 vs plain@24", cell(rows, "anchor", "24"), cell(rows, "plain", "24"))
        if pool == "primary" and "R1" in rules:
            anchored_fix[model] = rules["R1"]
if anchored_fix:
    print("\nWhat the paper may claim (primary pool; D1 holds on both 1B models in Part B)")
    for model, fix in anchored_fix.items():
        print(f"  {model}: re-anchored soft dedup {'holds' if fix else 'fails'}")
    print(f"  The training-free fix holds across scale once the answer is re-anchored: "
          f"{'yes' if any(anchored_fix.values()) else 'not shown'}")

passing = []
for model in [m for m in LLAMAS if (folder / f"{m}-ft6").exists()]:
    print(f"\n== Part C (untriggered, exploratory): {model}")
    arms = {arm: load(f"{model}-{arm}", ("gsm8k_valid",)) for arm in ("ft6", "ft12", "gate12_fixed")}
    base = load(model, ("gsm8k_valid",))
    if arms["ft6"] is None:
        print("  ft6 is not scored yet")
        continue
    long = [r["rationale_steps"] >= 4 for r in arms["ft6"]]
    for arm, rows in (("base", base),) + tuple(arms.items()):
        if rows:
            assert [r["id"] for r in rows] == [r["id"] for r in arms["ft6"]], f"{arm} covers different questions"
            table(rows, f"{arm}, gsm8k_valid, correct (4+ problems, of {sum(long)})", long)
    ft6 = cell(arms["ft6"], "plain", "6")
    for arm, policy in (("ft12", "plain"), ("gate12_fixed", "gate")):
        rows = arms[arm]
        if rows is None:
            print(f"  {arm}: not scored")
            continue
        main = compare(f"{arm} {policy}@12 vs ft6 plain@6, 4+ problems", cell(rows, policy, "12"), ft6, long)
        overall = compare(f"{arm} {policy}@12 vs ft6 plain@6, all", cell(rows, policy, "12"), ft6)
        hold = compare(f"{arm} {policy}@24 vs {policy}@12, all", cell(rows, policy, "24"), cell(rows, policy, "12"))
        compare(f"descriptive: {arm} {policy}@12 vs base plain@6, all", cell(rows, policy, "12"), cell(base, "plain", "6"))
        if main and overall and hold:
            ok = main["x"] > main["y"] and main["p"] < 0.025 and overall["x"] >= overall["y"] and \
                not (hold["y"] > hold["x"] and hold["p"] < 0.05)
            print(f"  => {arm} {'PASSES' if ok else 'fails'} the Part C rule")
            if ok:
                passing.append(arm)
                held, reference, plain = (load(f"{model}-{arm}", PRIMARY), load(f"{model}-ft6", PRIMARY),
                                          load(model, PRIMARY))
                if held and reference:
                    print(f"  held-out confirmation, primary pool")
                    a = compare(f"{arm} {policy}@12 vs ft6 plain@6", cell(held, policy, "12"), cell(reference, "plain", "6"))
                    b = compare(f"{arm} {policy}@12 vs base plain@6", cell(held, policy, "12"), cell(plain, "plain", "6"))
                    if a and b:
                        confirmed = all(r["x"] > r["y"] and r["p"] < 0.05 for r in (a, b))
                        print(f"  => the gain is {'confirmed' if confirmed else 'not confirmed'} on held-out data")
    compare("descriptive: ft6 plain@6 vs base plain@6, all", ft6, cell(base, "plain", "6"))
    print(f"  => passing arms: {', '.join(passing) or 'none'}")
if args.verdict:
    args.verdict.write_text(json.dumps({"passing": passing}) + "\n")
