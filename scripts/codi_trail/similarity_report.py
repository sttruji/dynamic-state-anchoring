"""Exploratory: does similarity between latent states at different depths predict correctness or later drift?

    python scripts/codi_trail/similarity_report.py results/codi_trail/probe   # needs the latents
"""
import json, sys
from pathlib import Path

import torch
import torch.nn.functional as F

folder = Path(sys.argv[1])

def auroc(scores, labels):
    scores = torch.as_tensor(scores, dtype=torch.float64)
    labels = torch.as_tensor(labels, dtype=torch.bool)
    pos, neg = scores[labels], scores[~labels]
    if not len(pos) or not len(neg):
        return float("nan")
    wins = (pos[:, None] > neg[None]).double().sum() + 0.5 * (pos[:, None] == neg[None]).double().sum()
    return float(wins / (len(pos) * len(neg)))

def cosine(x, a, b):
    return F.cosine_similarity(x[:, a], x[:, b], dim=-1)

for name in ("gsm8k_valid", "svamp"):
    path = folder / f"{name}.jsonl"
    if not path.exists():
        continue
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    saved = torch.load(folder / f"{name}.latents.pt", map_location="cpu", weights_only=True)
    assert saved["ids"] == [r["id"] for r in rows]
    correct = [r["plain"]["6"]["correct"] for r in rows]
    stable12 = [r["plain"]["12"]["prediction"] == r["plain"]["6"]["prediction"] for r in rows]
    stable24 = [r["plain"]["24"]["prediction"] == r["plain"]["6"]["prediction"] for r in rows]
    settled3 = [r["plain"]["3"]["prediction"] == r["plain"]["6"]["prediction"] for r in rows]
    print(f"\n=== {name}, n={len(rows)}: right at 6 {sum(correct)}; same answer 6 vs 12 {sum(stable12)}; 6 vs 24 {sum(stable24)}; 3 vs 6 {sum(settled3)}")
    for label, x in (("z (projected latent)", saved["z"]), ("h (last hidden)", saved["h"])):
        print(label)
        print("  mean step-to-step cosine t-1 to t:", [round(float(cosine(x, t - 1, t).mean()), 3) for t in (1, 2, 3, 4, 5, 6, 7, 8, 12, 24)])
        print("  mean cosine to step 6:           ", [(t, round(float(cosine(x, 6, t).mean()), 3)) for t in (3, 5, 7, 12, 24)])
        candidates = {"cos(2,3)": cosine(x, 2, 3), "cos(3,6)": cosine(x, 3, 6), "cos(5,6)": cosine(x, 5, 6),
                      "mean cos steps 1-6": torch.stack([cosine(x, t - 1, t) for t in range(1, 7)]).mean(0),
                      "cos(6,7)": cosine(x, 6, 7), "cos(6,12)": cosine(x, 6, 12), "cos(6,24)": cosine(x, 6, 24)}
        print("  score                   AUROC: right at 6 | no change 6-12 | no change 6-24 | no change 3-6")
        for key, value in candidates.items():
            cells = [auroc(value, target) for target in (correct, stable12, stable24, settled3)]
            print(f"  {key:22}" + "".join(f"{cell:10.3f}" for cell in cells))
