"""Exploratory, post hoc: is the CODI latent trail periodic, and do same-phase scores differ from cross-phase ones?

    python scripts/codi_trail/phase_report.py results/codi_trail/probe   # needs the latents
"""
import json, sys
from pathlib import Path
import torch, torch.nn.functional as F

folder = Path(sys.argv[1])

def auroc(scores, labels):
    scores = torch.as_tensor(scores, dtype=torch.float64)
    labels = torch.as_tensor(labels, dtype=torch.bool)
    pos, neg = scores[labels], scores[~labels]
    wins = (pos[:, None] > neg[None]).double().sum() + 0.5 * (pos[:, None] == neg[None]).double().sum()
    return float(wins / (len(pos) * len(neg)))

cos = lambda x, a, b: F.cosine_similarity(x[:, a], x[:, b], dim=-1)

for name in ("gsm8k_valid", "svamp"):
    rows = [json.loads(l) for l in (folder / f"{name}.jsonl").read_text().splitlines()]
    saved = torch.load(folder / f"{name}.latents.pt", map_location="cpu", weights_only=True)
    for label, x in (("z", saved["z"]), ("h", saved["h"])):
        x = x.double()
        print(f"\n=== {name} {label} shape {tuple(x.shape)}; norm of step t, t=0..12:", [round(float(x[:, t].norm(dim=-1).mean()), 1) for t in range(13)])
        print("mean cos(t, t+k), rows t=1,2,3,4,5,6,8,12,16; columns k=1..8")
        for t in (1, 2, 3, 4, 5, 6, 8, 12, 16):
            print(f"  t={t:2}", " ".join(f"{float(cos(x, t, t + k).mean()):6.3f}" if t + k <= 24 else "   .  " for k in range(1, 9)))
        print("mean cos(0, t) t=1..8:", [round(float(cos(x, 0, t).mean()), 3) for t in range(1, 9)])
        # A per-question view: how many questions show the alternation, not just the mean
        even, odd = cos(x, 6, 8), cos(x, 6, 7)
        print(f"questions with cos(6,8) > cos(6,7): {int((even > odd).sum())}/{len(even)}; quartiles cos(6,8): {[round(float(q), 3) for q in torch.quantile(even, torch.tensor([.25, .5, .75], dtype=torch.float64))]}")
    z = saved["z"].double()
    correct = [r["plain"]["6"]["correct"] for r in rows]
    stable24 = [r["plain"]["24"]["prediction"] == r["plain"]["6"]["prediction"] for r in rows]
    settled3 = [r["plain"]["3"]["prediction"] == r["plain"]["6"]["prediction"] for r in rows]
    print("\nsame-phase scores on z and h, AUROC: right at 6 | no change 6-24 | no change 3-6")
    for label, x in (("z", saved["z"].double()), ("h", saved["h"].double())):
        for a, b in ((2, 4), (4, 6), (2, 6), (6, 8), (6, 12), (1, 3), (3, 5), (5, 7), (7, 9)):
            s = cos(x, a, b)
            print(f"  {label} cos({a},{b}) mean {float(s.mean()):.3f}: {auroc(s, correct):.3f} {auroc(s, stable24):.3f} {auroc(s, settled3):.3f}")
