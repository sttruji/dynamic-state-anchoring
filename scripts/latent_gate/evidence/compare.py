"""Exploratory: CODI against SIM-CoT (CODI with per-step supervision) past 6 latent steps, same probe and measures.

    python compare.py <codi_probe.jsonl> <simcot_probe.jsonl>   # each needs its .latents.pt beside it
"""
import json, math, sys
from pathlib import Path

import torch
import torch.nn.functional as F


def sign_p(x, y):
    n, k = x + y, min(x, y)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n) if n else 1.0


def cos(z, a, b):
    return F.cosine_similarity(z[:, a], z[:, b], dim=-1)


models = {}
for label, path in (("CODI", Path(sys.argv[1])), ("SIM-CoT", Path(sys.argv[2]))):
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    saved = torch.load(path.with_suffix(".latents.pt"), map_location="cpu", weights_only=True)
    assert saved["ids"] == [r["id"] for r in rows]
    models[label] = (rows, saved["z"].double(), saved["h"].double())
ids = [r["id"] for r in models["CODI"][0]]
assert ids == [r["id"] for r in models["SIM-CoT"][0]], "both runs must cover the same questions in order"
n = len(ids)
print(f"n={n} questions, gsm8k_valid")

for label, (rows, z, h) in models.items():
    print(f"\n=== {label}")
    right = {d: [r["plain"][d]["correct"] for r in rows] for d in ("3", "6", "12", "24")}
    print(" correct:", {d: sum(v) for d, v in right.items()})
    for d in ("12", "24"):
        x = sum(a and not b for a, b in zip(right[d], right["6"]))
        y = sum(b and not a for a, b in zip(right[d], right["6"]))
        print(f"  {d} vs 6: gained {x}, lost {y}, p={sign_p(x, y):.3g}")
    same = {d: sum(r["plain"][d]["prediction"] == r["plain"]["6"]["prediction"] for r in rows) for d in ("12", "24")}
    print(" answer unchanged from 6:", same)
    for name, x in (("z", z), ("h", h)):
        print(f" {name}: mean cos(t,t+1) | cos(t,t+2), t=1..10:")
        print("  ", " ".join(f"{float(cos(x, t, t + 1).mean()):.2f}|{float(cos(x, t, t + 2).mean()):.2f}" for t in range(1, 11)))
    print(" z: same, t=12,16,20:", " ".join(f"{float(cos(z, t, t + 1).mean()):.2f}|{float(cos(z, t, t + 2).mean()):.2f}" for t in (12, 16, 20)))
    alternating = int((cos(z, 6, 8) > cos(z, 6, 7) + 0.2).sum())
    print(f" questions with cos(6,8) > cos(6,7) + 0.2: {alternating}/{n}; settled cos(6,7) >= .95: {int((cos(z, 6, 7) >= .95).sum())}")
    print(" within-6 pairwise mean cos, odd steps | even steps | across:", end=" ")
    odd, even = (1, 3, 5), (2, 4, 6)
    within = lambda s: sum(float(cos(z, a, b).mean()) for i, a in enumerate(s) for b in s[i + 1:]) / 3
    print(f"{within(odd):.3f} | {within(even):.3f} | {sum(float(cos(z, a, b).mean()) for a in odd for b in even) / 9:.3f}")
    r = z - z.mean(0)
    print(" question-specific energy share, t=2,6,12,24:", [round(float(r[:, t].pow(2).sum(-1).mean() / z[:, t].pow(2).sum(-1).mean()), 3) for t in (2, 6, 12, 24)])
    print(" raw cos(6,t):     ", [(t, round(float(cos(z, 6, t).mean()), 3)) for t in (8, 12, 24)])
    print(" residual cos(6,t):", [(t, round(float(cos(r, 6, t).mean()), 3)) for t in (8, 12, 24)])
    print(" residual cos(5,t):", [(t, round(float(cos(r, 5, t).mean()), 3)) for t in (7, 11, 23)])

    def retrieval(a, b):
        ra, rb = F.normalize(r[:, a], dim=-1), F.normalize(r[:, b], dim=-1)
        return round(float(((ra @ rb.T).argmax(1) == torch.arange(n)).double().mean()), 3)
    print(" top-1 retrieval from step 6:", [(t, retrieval(6, t)) for t in (8, 12, 24)], " from step 5:", [(t, retrieval(5, t)) for t in (7, 11, 23)])

print("\n=== paired, SIM-CoT against CODI")
for d in ("6", "12", "24"):
    a = [r["plain"][d]["correct"] for r in models["SIM-CoT"][0]]
    b = [r["plain"][d]["correct"] for r in models["CODI"][0]]
    x, y = sum(p and not q for p, q in zip(a, b)), sum(q and not p for p, q in zip(a, b))
    print(f" {d} steps: SIM-CoT {sum(a)} vs CODI {sum(b)}; SIM-CoT only {x}, CODI only {y}, p={sign_p(x, y):.3g}")
