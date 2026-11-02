"""Apply results/codi_trail/DYNAMICS_PLAN.md to the post hoc dynamics run. Exploratory throughout.

    python scripts/codi_trail/dynamics_report.py results/codi_trail/probe results/codi_trail/dynamics
"""
import json, math, sys
from pathlib import Path

probe_dir, dynamics_dir = Path(sys.argv[1]), Path(sys.argv[2])

def sign_p(x, y):
    n, k = x + y, min(x, y)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n) if n else 1.0

def load_dynamics(name):
    path = dynamics_dir / f"{name}.jsonl"
    rows = [json.loads(l) for l in path.read_text().splitlines()] if path.exists() else []
    return {r["id"]: r for r in rows}

probe_rows = {n: [json.loads(l) for l in (probe_dir / f"{n}.jsonl").read_text().splitlines()] for n in ("gsm8k_valid", "svamp")}
valid = load_dynamics("gsm8k_valid")
base = {r["id"]: r for r in probe_rows["gsm8k_valid"]}
assert set(valid) == set(base), (len(valid), len(base))
bad = [(i, k) for i, r in valid.items() for k in ("6", "12") if r["plain"][k]["text"] != base[i]["plain"][k]["text"]]
print(f"CONTROL {'PASS' if not bad else 'FAIL'}: {2 * len(valid) - len(bad)}/{2 * len(valid)} texts match at 6 and 12")
if bad:
    sys.exit(1)

ids = list(base)
def right(depth):
    source = base if depth in ("3", "24") else valid
    return {i: source[i]["plain"][depth]["correct"] for i in ids}
curve = {d: sum(right(str(d)).values()) for d in (2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 24)}
print("\nD1 correct by depth (1 step was 126 in the sweep):", curve)
gains = losses = wins = 0
for even, odd in ((4, 5), (6, 5), (6, 7), (8, 7), (8, 9), (10, 9)):
    a, b = right(str(even)), right(str(odd))
    x, y = sum(a[i] and not b[i] for i in ids), sum(b[i] and not a[i] for i in ids)
    gains, losses, wins = gains + x, losses + y, wins + (curve[even] > curve[odd])
    print(f"  even {even} ({curve[even]}) vs odd {odd} ({curve[odd]}): {x}/{y}, p={sign_p(x, y):.3g}")
print(f"  even ahead in {wins} of 6; pooled {gains}/{losses}, p={sign_p(gains, losses):.3g} (pairs share depths, so the pooled p is optimistic)")
print("  PARITY EFFECT" if wins >= 5 and gains > losses and sign_p(gains, losses) < 0.01 else "  no parity effect by the rule")

def compare(title, left, reference):
    x, y = sum(left[i] and not reference[i] for i in ids), sum(reference[i] and not left[i] for i in ids)
    print(f"  {title}: {sum(left.values())} vs {sum(reference.values())}, gained {x}, lost {y}, p={sign_p(x, y):.3g}")
    return x, y

print("\nD2 drift source")
six = right("6")
for label in ("pos_reset", "late_six"):
    for depth in ("12", "24"):
        got = {i: valid[i][label][depth]["correct"] for i in ids}
        x, y = compare(f"{label} at {depth} vs plain {depth}", got, right(depth))
        compare(f"{label} at {depth} vs plain 6 ", got, six)
        same = sum(valid[i][label][depth]["prediction"] == base[i]["plain"]["6"]["prediction"] for i in ids)
        print(f"     answers equal to plain 6: {same}/500 (plain {depth}: {sum(base[i]['plain'][depth]['prediction'] == base[i]['plain']['6']['prediction'] for i in ids)})")
        if depth == "24":
            print("     RECOVERS by the rule" if sum(got.values()) >= 190 and x > y and sign_p(x, y) < 0.05 else "     does not recover by the rule")

print("\nD3 trail content")
def content(name, records, rows):
    order = [r["id"] for r in rows]
    by_id = {r["id"]: r for r in rows}
    own = {i: by_id[i]["plain"]["6"] for i in order}
    print(name, "plain 6 correct", sum(o["correct"] for o in own.values()))
    for label in ("swap", "mean"):
        got = {i: records[i][label] for i in order}
        x = sum(got[i]["correct"] and not own[i]["correct"] for i in order)
        y = sum(own[i]["correct"] and not got[i]["correct"] for i in order)
        keep = sum(got[i]["prediction"] == own[i]["prediction"] for i in order)
        line = f"  {label}: correct {sum(g['correct'] for g in got.values())}, gained {x}, lost {y}, p={sign_p(x, y):.3g}; same answer as own trail {keep}/{len(order)}"
        if label == "swap":
            donor = sum(got[i]["prediction"] == own[order[records[i]["donor"]]]["prediction"] for i in order)
            chance = sum(own[i]["prediction"] == own[order[records[i]["donor"]]]["prediction"] for i in order)
            line += f"; same as donor's answer {donor}/{len(order)} (own vs donor by chance {chance})"
        print(line)
    return sum(records[i]["swap"]["correct"] for i in order)

score = content("gsm8k_valid", valid, probe_rows["gsm8k_valid"])
print("  rule:", "little question-specific content" if score >= 185 else "real content if answers follow donor" if score < 100 else "mixed")
svamp = load_dynamics("svamp")
if svamp:
    content("svamp", svamp, probe_rows["svamp"])
    import torch
    import torch.nn.functional as F
    z = torch.load(probe_dir / "svamp.latents.pt", map_location="cpu", weights_only=True)["z"].double()
    c = F.cosine_similarity(z[:, 6], z[:, 7], dim=-1)
    rows = probe_rows["svamp"]
    for title, mask in (("fixed point, cos(6,7)>=.95", c >= .95), ("cycling, cos(6,7)<.6", c < .6)):
        chosen = [rows[i]["id"] for i in mask.nonzero().flatten().tolist()]
        own = {r["id"]: r["plain"]["6"] for r in rows}
        print(f"  {title}: n={len(chosen)} plain {sum(own[i]['correct'] for i in chosen)}, swap {sum(svamp[i]['swap']['correct'] for i in chosen)}, mean {sum(svamp[i]['mean']['correct'] for i in chosen)}; swap same as own {sum(svamp[i]['swap']['prediction'] == own[i]['prediction'] for i in chosen)}")
