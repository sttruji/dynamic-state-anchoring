"""Does CODI drift when its latent loop runs past the 6 steps it trained on?

    python scripts/codi_trail/sweep_report.py results/codi_trail/sweep results
"""
import json, math, sys
from pathlib import Path

depth, committed = Path(sys.argv[1]), Path(sys.argv[2])
STEPS = (1, 3, 6, 12, 24)

def load(path):
    lines = path.read_text().splitlines()
    return {r["id"]: r for r in map(json.loads, lines[1:])}

def sign_p(x, y):
    n, k = x + y, min(x, y)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n) if n else 1.0

def num(t):
    try:
        return float(str(t).replace(",", ""))
    except ValueError:
        return None

def wilson(k, n, z=1.96):
    p, d = k / n, 1 + z * z / n
    c, h = (p + z * z / (2 * n)) / d, z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return round(100 * (c - h), 1), round(100 * (c + h), 1)

rows = {n: load(depth / "predictions" / f"codi-iter{n}" / "gsm8k_valid.jsonl") for n in STEPS}
base = load(committed / "predictions" / "codi" / "gsm8k_valid.jsonl")
ref = rows[6]
print("control: rerun of 6 steps vs the committed codi row")
print("  correct", sum(r["correct"] for r in ref.values()), "vs", sum(r["correct"] for r in base.values()),
      "| identical text", sum(ref[i]["generated_text"] == base[i]["generated_text"] for i in ref), "/ 500")
print("\nsteps correct  acc%   95% CI        valid%  vs 6: only N / only 6    p   same answer  N closer / 6 closer   p")
for n in STEPS:
    r = rows[n]
    k = sum(x["correct"] for x in r.values())
    x = sum(r[i]["correct"] and not ref[i]["correct"] for i in r)
    y = sum(ref[i]["correct"] and not r[i]["correct"] for i in r)
    same = sum(r[i]["prediction"] == ref[i]["prediction"] for i in r)
    cn = c6 = 0
    for i in r:
        a, b, g = num(r[i]["prediction"]), num(ref[i]["prediction"]), num(r[i]["gold"])
        if None in (a, b, g) or a == b:
            continue
        cn += abs(a - g) < abs(b - g)
        c6 += abs(b - g) < abs(a - g)
    valid = 100 * sum(v["valid_answer"] for v in r.values()) / len(r)
    print(f"{n:5} {k:6} {100 * k / len(r):6.1f}  {str(wilson(k, len(r))):14} {valid:6.1f}  {x:6} / {y:<6} {sign_p(x, y):6.3f}  {same:8}  {cn:8} / {c6:<8} {sign_p(cn, c6):.3f}")

print("\nby reasoning steps in the sft output (accuracy % at each CODI step count)")
sft = load(committed / "predictions" / "gpt2-sft" / "gsm8k_valid.jsonl")
count = {i: sft[i]["generated_text"].count("<<") for i in ref}
print("sft steps   n  " + "".join(f"{n:>8}" for n in STEPS))
for lo, hi in ((0, 2), (3, 3), (4, 4), (5, 99)):
    q = [i for i in ref if lo <= count[i] <= hi]
    print(f"{lo}-{hi:<7} {len(q):4}  " + "".join(f"{100 * sum(rows[n][i]['correct'] for i in q) / len(q):8.1f}" for n in STEPS))

print("\ntypical output")
for n in STEPS:
    sample = list(rows[n].values())[:2]
    print(n, [s["generated_text"][:44] for s in sample], "| distinct", len({v["generated_text"] for v in rows[n].values()}),
          "| seconds/q", round(sum(v["seconds"] for v in rows[n].values()) / len(rows[n]), 3))
