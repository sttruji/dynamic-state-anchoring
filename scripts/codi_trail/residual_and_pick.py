"""Post hoc: how fast question-specific latent content decays, and whether answer confidence can pick a depth per question.

    python scripts/codi_trail/residual_and_pick.py results/codi_trail/probe   # needs the latents
"""
import json, math, sys
from pathlib import Path
import torch, torch.nn.functional as F

folder = Path(sys.argv[1])

def sign_p(x, y):
    n, k = x + y, min(x, y)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n) if n else 1.0

for name in ("gsm8k_valid", "svamp"):
    rows = [json.loads(l) for l in (folder / f"{name}.jsonl").read_text().splitlines()]
    z = torch.load(folder / f"{name}.latents.pt", map_location="cpu", weights_only=True)["z"].double()
    n = len(rows)
    means = z.mean(0)
    r = z - means
    masks = {"all": torch.ones(n, dtype=torch.bool)}
    if name == "svamp":
        c = F.cosine_similarity(z[:, 6], z[:, 7], dim=-1)
        masks = {"cycling": c < .6, "fixed": c >= .95}
    print(f"\n=== {name}, n={n}")
    print(" share of latent energy that is question-specific, step t:", [(t, round(float(r[:, t].pow(2).sum(-1).mean() / z[:, t].pow(2).sum(-1).mean()), 3)) for t in (0, 2, 4, 6, 8, 12, 24)])
    for label, m in masks.items():
        print(f" {label} n={int(m.sum())}: residual cosine after removing the per-step dataset mean")
        print("   raw cos(6,t):     ", [(t, round(float(F.cosine_similarity(z[m, 6], z[m, t], dim=-1).mean()), 3)) for t in (8, 10, 12, 24)])
        print("   residual cos(6,t):", [(t, round(float(F.cosine_similarity(r[m, 6], r[m, t], dim=-1).mean()), 3)) for t in (2, 4, 8, 10, 12, 24)])
        print("   residual cos(2,t):", [(t, round(float(F.cosine_similarity(r[m, 2], r[m, t], dim=-1).mean()), 3)) for t in (4, 6, 8, 12, 24)])
    def retrieval(a, b):
        ra = F.normalize(r[:, a], dim=-1)
        rb = F.normalize(r[:, b], dim=-1)
        return float(((ra @ rb.T).argmax(1) == torch.arange(n)).double().mean())
    print(" can a question be found from its later latent? top-1 retrieval among all questions, using step 6 as the key:", [(t, round(retrieval(6, t), 3)) for t in (2, 4, 8, 12, 24)])
    print(" same with step 2 as the key:", [(t, round(retrieval(2, t), 3)) for t in (4, 6, 8, 12, 24)])

    print(" pick depth by sum of token logprobs; bar: above 6 steps with sign test p<0.05 on both sets")
    six = [r_["plain"]["6"]["correct"] for r_ in rows]
    oracle = sum(any(r_["plain"][d]["correct"] for d in ("3", "6", "12", "24")) for r_ in rows)
    print(f"  6 steps {sum(six)}; oracle over 3,6,12,24: {oracle}")
    for depths in (("6", "12"), ("3", "6"), ("3", "6", "12"), ("6", "12", "24"), ("3", "6", "12", "24")):
        for label, value in (("sum", lambda x: sum(x["logprobs"])), ("min", lambda x: min(x["logprobs"]))):
            picked = [max((r_["plain"][d] for d in depths), key=value)["correct"] for r_ in rows]
            x = sum(p and not s for p, s in zip(picked, six))
            y = sum(s and not p for p, s in zip(picked, six))
            print(f"  depths {','.join(depths):10} by {label}: {sum(picked)}  gained {x}, lost {y}, p={sign_p(x, y):.3g}")
