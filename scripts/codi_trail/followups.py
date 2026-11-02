"""Exploratory, post hoc checks on the probe runs. Nothing here counts toward the preregistered verdicts.

    python scripts/codi_trail/followups.py results/codi_trail/probe   # needs the latents
"""
import json, math, random, sys
from collections import Counter
from pathlib import Path
import torch, torch.nn.functional as F

folder = Path(sys.argv[1])
torch.manual_seed(7)

def auroc(scores, labels):
    scores = torch.as_tensor(scores, dtype=torch.float64)
    labels = torch.as_tensor(labels, dtype=torch.bool)
    pos, neg = scores[labels], scores[~labels]
    if not len(pos) or not len(neg):
        return float("nan")
    wins = (pos[:, None] > neg[None]).double().sum() + 0.5 * (pos[:, None] == neg[None]).double().sum()
    return float(wins / (len(pos) * len(neg)))

def load(name):
    rows = [json.loads(l) for l in (folder / f"{name}.jsonl").read_text().splitlines()]
    saved = torch.load(folder / f"{name}.latents.pt", map_location="cpu", weights_only=True)
    z = saved["z"].double()
    cos = lambda a, b: F.cosine_similarity(z[:, a], z[:, b], dim=-1)
    six = [r["plain"]["6"] for r in rows]
    return dict(
        rows=rows, cos=cos,
        correct=torch.tensor([s["correct"] for s in six], dtype=torch.float64),
        stable=torch.tensor([r["plain"]["24"]["prediction"] == r["plain"]["6"]["prediction"] for r in rows], dtype=torch.float64),
        b1=torch.tensor([sum(s["logprobs"]) for s in six], dtype=torch.float64),
        mean_lp=torch.tensor([sum(s["logprobs"]) / len(s["logprobs"]) for s in six], dtype=torch.float64),
        length=torch.tensor([len(s["logprobs"]) for s in six], dtype=torch.float64),
        count=torch.tensor([sum(r["plain"][k]["prediction"] == r["plain"]["6"]["prediction"] for k in ("3", "12", "24")) for r in rows], dtype=torch.float64),
        agree=torch.tensor([len({r["plain"][k]["prediction"] for k in ("3", "6", "12")}) == 1 for r in rows], dtype=torch.bool),
    )

data = {name: load(name) for name in ("gsm8k_valid", "svamp")}

print("=== shape of the 2-cycle: cos(6,7) histogram and 6-step accuracy in each bin")
for name, d in data.items():
    c = d["cos"](6, 7)
    edges = [-1, .2, .4, .6, .8, .95, .999, 1.01]
    print(name)
    for lo, hi in zip(edges, edges[1:]):
        m = (c >= lo) & (c < hi)
        if m.sum():
            print(f"  cos(6,7) in [{lo:5.2f},{hi:5.2f}): n={int(m.sum()):4}  right at 6 {int(d['correct'][m].sum()):4}  same answer 6 vs 24 {int(d['stable'][m].sum()):4}")

print("\n=== is the logprob baseline an artifact of output length?")
for name, d in data.items():
    print(name, "text sample:", [r["plain"]["6"]["text"][:40] for r in d["rows"][:3]])
    print(" tokens in answer:", sorted(Counter(d["length"].int().tolist()).items()))
    print(f" AUROC sum {auroc(d['b1'], d['correct']):.3f}  mean {auroc(d['mean_lp'], d['correct']):.3f}  length alone {auroc(-d['length'], d['correct']):.3f}")
    for n, count in Counter(d["length"].int().tolist()).most_common(3):
        m = d["length"] == n
        print(f"  length {n}: n={count}, accuracy {float(d['correct'][m].mean()):.3f}, AUROC within length {auroc(d['b1'][m], d['correct'][m].bool()):.3f}")

print("\n=== R3 weak spot: agreement effect within logprob quintiles, weighted so thin cells count little")
for name, d in data.items():
    order = d["b1"].argsort().tolist()
    n = len(order)
    bins = [order[q * n // 5:(q + 1) * n // 5] for q in range(5)]
    def mh(selected):
        top = bottom = 0.0
        for members in selected:
            m = torch.tensor(members)
            yes, no = d["agree"][m], ~d["agree"][m]
            a, b = int(yes.sum()), int(no.sum())
            if a and b:
                w = a * b / (a + b)
                top += w * float(d["correct"][m][yes].mean() - d["correct"][m][no].mean())
                bottom += w
        return top / bottom
    rng = random.Random(7)
    draws = sorted(mh([[rng.choice(m) for _ in m] for m in bins]) for _ in range(2000))
    print(f"  {name}: Mantel-Haenszel difference {100 * mh(bins):+.1f} pts, CI [{100 * draws[50]:+.1f}, {100 * draws[1949]:+.1f}]")

print("\n=== does anything add to logprob? fit on validation, score on SVAMP. Features standardized on validation.")
def fit(x, y, steps=30, ridge=1e-3):
    x = torch.cat([torch.ones(len(x), 1, dtype=x.dtype), x], 1)
    w = torch.zeros(x.shape[1], dtype=x.dtype)
    for _ in range(steps):
        p = torch.sigmoid(x @ w).clamp(1e-6, 1 - 1e-6)
        g = x.T @ (p - y) + ridge * w
        H = (x.T * (p * (1 - p))) @ x + ridge * torch.eye(len(w), dtype=x.dtype)
        w = w - torch.linalg.solve(H, g)
    return w

def features(d, spec):
    columns = []
    for key in spec:
        if isinstance(key, tuple):
            columns.append(-torch.log1p(-d["cos"](*key).clamp(max=1 - 1e-9)))
        else:
            columns.append(d[key])
    return torch.stack(columns, 1)

train, test = data["gsm8k_valid"], data["svamp"]
def run(target, specs):
    print(f"target: {target}")
    y, yt = train[target], test[target]
    base = None
    for label, spec in specs:
        x, xt = features(train, spec), features(test, spec)
        mean, std = x.mean(0), x.std(0)
        w = fit((x - mean) / std, y)
        score = torch.cat([torch.ones(len(xt), 1, dtype=xt.dtype), (xt - mean) / std], 1) @ w
        value = auroc(score, yt.bool())
        line = f"  {label:34} SVAMP AUROC {value:.3f}"
        if base is None:
            base = score
        else:
            rng = random.Random(7)
            n = len(yt)
            diffs = sorted(auroc(score[i], yt[i].bool()) - auroc(base[i], yt[i].bool()) for i in (torch.tensor([rng.randrange(n) for _ in range(n)]) for _ in range(500)))
            line += f"  vs first row {value - auroc(base, yt.bool()):+.3f} CI [{diffs[12]:+.3f}, {diffs[487]:+.3f}]"
        print(line)

specs = [("logprob", ["b1"]), ("logprob + agreement count", ["b1", "count"]), ("logprob + cos(4,6)", ["b1", (4, 6)]),
         ("logprob + cos(2,6)", ["b1", (2, 6)]), ("logprob + cos(6,8)", ["b1", (6, 8)]), ("logprob + cos(6,7)", ["b1", (6, 7)]),
         ("cos(4,6) alone", [(4, 6)]), ("agreement count alone", ["count"])]
run("correct", specs)
# The agreement count includes 6 vs 24, which is the stability target itself, so it is left out here.
run("stable", [(label, spec) for label, spec in specs if "count" not in spec])
