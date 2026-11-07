"""Verdicts for Run 1 of step selection, by the criteria frozen in results/step_select/run1/PLAN.md.

    python scripts/step_select/run1_report.py <run1 output dir> [--expected 1319]

A: the V1 vote over greedy (1.01) and the first 3 reruns at each rerun step; CONFIRM needs a net of +8 and a two-sided
question-level sign test p < 0.05.

B: a conditional-logit probe on a 64-dimensional PCA of [hidden; latent], cross-fitted over five outer folds hashed from
the question id, with the L2 weight and the candidate-0 margin chosen on four inner folds. GO needs within-set AUC >= 0.65,
a net of +27 for the one-deviation selector at K = 4 and p < 0.05. REPORT is an AUC pass with a gain failure, and KILL
is AUC < 0.65 or fewer than 300 mixed sets.

A run short of --expected questions, or one where greedy does not match the reference on every question, cannot declare
CONFIRM or GO.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("run", type=Path)
parser.add_argument("--expected", type=int, default=1319, help="questions a complete run holds")
parser.add_argument("--components", type=int, default=64)
args = parser.parse_args()

L2S, MARGINS = (0.01, 0.1, 1.0, 10.0), (0.0, 0.25, 0.5, 1.0)
VOTE_RERUNS, K_SELECT = 3, 4
A_NET, B_AUC, B_NET, MIN_SETS, ALPHA = 8, 0.65, 27, 300, 0.05


def sign_p(wins: int, losses: int) -> float:
    n = wins + losses
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(min(wins, losses) + 1)) / 2 ** n) if n else 1.0


def fold(qid: str, salt: str, k: int) -> int:
    return int(hashlib.sha256(f"run1-{salt}:{qid}".encode()).hexdigest(), 16) % k


def vote(spine, hedges) -> bool:
    scores = Counter()
    for entry, weight in [(spine, 1.01)] + [(h, 1.0) for h in hedges]:
        scores["gold" if entry["correct"] else ("wrong", entry["prediction"])] += weight
    return max(scores, key=scores.get) == "gold"


def net(new, old):
    wins = sum(a and not b for a, b in zip(new, old))
    losses = sum(b and not a for a, b in zip(new, old))
    return wins - losses, wins, losses, sign_p(wins, losses)


records = [json.loads(x) for x in (args.run / "records.jsonl").read_text().splitlines()]
n, steps, reruns = len(records), len(records[0]["branches"]), len(records[0]["branches"][0])
greedy = [r["greedy"]["correct"] for r in records]
identity = sum(r["greedy"].get("matches_reference", False) for r in records)
complete = n == args.expected and identity == n
print(f"Run 1: {n}/{args.expected} questions, {steps} rerun steps x {reruns} reruns; greedy {sum(greedy)} "
      f"({sum(greedy) / n:.1%}); greedy matches the reference on {identity}/{n}"
      f"{'' if complete else ' -> no CONFIRM or GO can be declared'}")

# A: the steps 1-3 vote.
voted = [vote(r["greedy"], [b for bs in r["branches"] for b in bs[:VOTE_RERUNS]]) for r in records]
gain, wins, losses, p = net(voted, greedy)
confirm = gain >= A_NET and p < ALPHA
print(f"\nA. Vote over greedy and the first {VOTE_RERUNS} reruns at steps 1-{steps}: {sum(voted)} vs greedy {sum(greedy)}, "
      f"net {gain:+d} (+{wins}/-{losses}, p={p:.3f})")
print(f"  A: {'CONFIRM' if confirm and complete else 'KILL' if not confirm else 'NO VERDICT (incomplete run)'}")

# B: the cross-fitted hidden-state probe.
print("\nB. Hidden-state probe")
if reruns < 7:
    print(f"  not run: only {reruns} reruns per step (the preflight downgrade); Stage 0b takes its place")
    raise SystemExit
chunks = sorted((args.run / "features").glob("c*.npz"), key=lambda p: int(p.stem[1:]))
ids, hidden, latent = [], [], []
for path in chunks:
    with np.load(path) as part:
        ids += [str(x) for x in part["ids"]]
        hidden.append(part["hidden"].astype(np.float32))
        latent.append(part["latent"].astype(np.float32))
position = {qid: i for i, qid in enumerate(ids)}
order = [position[r["id"]] for r in records]
x = np.concatenate([np.concatenate(hidden), np.concatenate(latent)], -1)[order]  # [n, steps, 1 + reruns, features]
y = np.array([[[r["greedy"]["correct"]] + [b["correct"] for b in bs] for bs in r["branches"]] for r in records])
qids = [r["id"] for r in records]
outer = np.array([fold(q, "outer", 5) for q in qids])
inner = np.array([fold(q, "inner", 4) for q in qids])
mixed = y.any(-1) & ~y.all(-1)  # [n, steps]
print(f"  {int(mixed.sum())} mixed sets of {n * steps}")


def fit(train):
    """Standardization, PCA and whitening on every candidate of the `train` questions: a feature map."""
    flat = x[train].reshape(-1, x.shape[-1])
    mean, std = flat.mean(0), flat.std(0) + 1e-6
    z = (flat - mean) / std
    values, vectors = np.linalg.eigh(z.T @ z / len(z))
    top = vectors[:, np.argsort(values)[::-1][:args.components]]
    scale = (z @ top).std(0) + 1e-6
    return lambda q: (((q - mean) / std) @ top) / scale


def clogit(features, labels, l2, iterations=400, lr=0.5):
    """Conditional logit over each set's candidates: minimizes the summed -log mass on right candidates plus
    l2/2 ||w||^2. features [sets, candidates, d], labels [sets, candidates] bool."""
    w = np.zeros(features.shape[-1])
    sets = max(1, len(features))
    for _ in range(iterations):
        s = features @ w
        s = s - s.max(1, keepdims=True)
        prob = np.exp(s)
        prob /= prob.sum(1, keepdims=True)
        right = prob * labels
        right /= right.sum(1, keepdims=True)
        grad = ((prob - right)[:, :, None] * features).sum((0, 1)) + l2 * w
        w -= lr * grad / sets
    return w


def within_auc(scores, labels, keep) -> float:
    wins = ties = pairs = 0.0
    for s, l in zip(scores[keep], labels[keep]):
        right, wrong = s[l], s[~l]
        if len(right) and len(wrong):
            diff = right[:, None] - wrong[None, :]
            wins += (diff > 0).sum()
            ties += (diff == 0).sum()
            pairs += diff.size
    return (wins + 0.5 * ties) / pairs if pairs else float("nan")


def selector(scores, labels, questions, margin):
    """The one-deviation selector at K = 4: switch to the rerun (step, k >= 1) with the largest score above its own set's
    candidate 0 when that exceeds `margin`; correctness per question."""
    out = []
    for q in questions:
        lift = scores[q, :, 1:K_SELECT] - scores[q, :, :1]
        s, k = np.unravel_index(np.argmax(lift), lift.shape)
        out.append(bool(labels[q, s, k + 1]) if lift[s, k] > margin else bool(labels[q, 0, 0]))
    return out


def scored(train, test, l2s):
    """Out-of-sample scores [len(test), steps, candidates] from probes fitted on `train`, one per L2 weight in `l2s`;
    the feature map is fitted once and shared."""
    project = fit(train)
    keep = mixed[train]
    fitted, held = project(x[train][keep]), project(x[test])
    return {l2: held @ clogit(fitted, y[train][keep], l2) for l2 in l2s}


all_q = np.arange(n)
scores = np.zeros(y.shape)
chosen = []
for f in range(5):
    train, test = all_q[outer != f], all_q[outer == f]
    inner_scores = {l2: np.zeros(y.shape) for l2 in L2S}
    per_fold = {l2: [] for l2 in L2S}
    for g in range(4):
        tr, te = train[inner[train] != g], train[inner[train] == g]
        for l2, s_te in scored(tr, te, L2S).items():
            inner_scores[l2][te] = s_te
            per_fold[l2].append(within_auc(s_te, y[te], mixed[te]))
    inner_auc = {l2: float(np.nanmean(v)) for l2, v in per_fold.items()}  # the mean of the inner folds' AUCs
    l2 = max(L2S, key=lambda v: (inner_auc[v], v))
    gains = {m: net(selector(inner_scores[l2], y, train, m), [bool(y[q, 0, 0]) for q in train])[0] for m in MARGINS}
    margin = max(MARGINS, key=lambda m: (gains[m], m))
    scores[test] = scored(train, test, [l2])[l2]
    chosen.append((l2, margin))
    print(f"  outer fold {f}: {len(test)} questions, L2 {l2}, margin {margin} (inner AUC {inner_auc[l2]:.3f}, "
          f"inner net {gains[margin]:+d})")

auc = within_auc(scores, y, mixed)
selected = []
for f, (_, margin) in enumerate(chosen):
    test = all_q[outer == f]
    selected += list(zip(test, selector(scores, y, test, margin)))
selected = [c for _, c in sorted(selected)]
gain_b, wins_b, losses_b, p_b = net(selected, greedy)
detector = np.zeros(y.shape)
detector[:, :, 0] = 1.0
print(f"  within-set AUC {auc:.3f} over {int(mixed.sum())} mixed sets (a candidate-0 detector: "
      f"{within_auc(detector, y, mixed):.3f})")
print(f"  one-deviation selector at K={K_SELECT}: {sum(selected)} vs greedy {sum(greedy)}, net {gain_b:+d} "
      f"(+{wins_b}/-{losses_b}, p={p_b:.3f})")
for share in (40, 80):  # reported, never gating
    partial = np.zeros(y.shape)
    for f, (l2, _) in enumerate(chosen):
        train = all_q[(outer != f) & np.array([fold(q, "fraction", 100) < share for q in qids])]
        partial[all_q[outer == f]] = scored(train, all_q[outer == f], [l2])[l2]
    print(f"  learning curve: within-set AUC {within_auc(partial, y, mixed):.3f} with {share}% of the training questions")

if mixed.sum() < MIN_SETS or auc < B_AUC:
    verdict = "KILL: every selector route ends"
elif gain_b >= B_NET and p_b < ALPHA and complete:
    verdict = "GO: Run 2 (Stage 0b) runs"
else:
    verdict = "REPORT: write up the AUC and the gain; no more compute" + ("" if complete else " (incomplete run)")
print(f"  B: {verdict}")
