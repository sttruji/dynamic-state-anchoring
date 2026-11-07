"""E0 of the multi-candidate extension (docs/brainstorm/MULTI_CANDIDATE_SELECTION.md, section 5): offline checks on Stage 0
records, with greedy as the spine and Stage 0's one-step branches as the hedges. No model runs.

    python scripts/step_select/e0.py results/step_select/stage0/evals/seed1.jsonl results/step_select/stage0/evals/seed2.jsonl

The vote is V1: each answer sums its chains' weights, the spine weighs 1 + 0.01 so it wins ties, and every hedge weighs 1.
Right answers count as one answer (they all equal the gold); wrong ones are grouped by prediction.

Gate, fixed in MULTI_CANDIDATE_SELECTION.md before Stage 0 ran: drop keep-alive multi-select if vote-reachable coverage
(greedy right, or at least 2 branches right) adds fewer than 20 of 500 questions over greedy, averaged over the seeds.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path

import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("runs", type=Path, nargs="+", help="Stage 0 eval.py outputs, one per sampling seed")
parser.add_argument("--decoder-tokens", type=float, help="mean decoded-step length; default: measured with GPT-2's tokenizer")
parser.add_argument("--answer-tokens", type=float, default=7.0, help="mean answer length in tokens")
args = parser.parse_args()
sys.path.insert(0, str(Path(__file__).resolve().parent))
from stepsel import as_number, choose, rank  # noqa: E402

EPS, GATE = 0.01, 20 / 500


def sign_p(wins: int, losses: int) -> float:
    n = wins + losses
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(min(wins, losses) + 1)) / 2 ** n) if n else 1.0


def vote(spine, hedges, weights=None) -> bool:
    """V1 over the spine and hedges (dicts with prediction and correct): whether the winning answer is right."""
    scores = Counter()
    for entry, weight in [(spine, 1 + EPS)] + list(zip(hedges, weights or [1.0] * len(hedges))):
        scores["gold" if entry["correct"] else ("wrong", entry["prediction"])] += weight
    return max(scores, key=scores.get) == "gold"


def tally(name, results, greedy, cost=None):
    """Accuracy per seed, the pooled paired sign test against greedy, and an optional cost in token-forwards."""
    per_seed = [sum(r) for r in results]
    wins = sum(a and not b for res, g in zip(results, greedy) for a, b in zip(res, g))
    losses = sum(b and not a for res, g in zip(results, greedy) for a, b in zip(res, g))
    extra = "" if cost is None else f"; about {cost:.0f} token-forwards per question"
    print(f"  {name:52} {' / '.join(map(str, per_seed))}   vs greedy {wins - losses:+d} (+{wins}/-{losses}, "
          f"p={sign_p(wins, losses):.3f}){extra}")


def branch_answer_in_results(record, s, b) -> bool:
    """Is the branch's final answer among the decoded results of greedy's slots before s or of the rerun's own slot?"""
    value = as_number(b["prediction"])
    results = [f["result"] for f in record["greedy"]["steps"][:s]] + [b.get("result")]
    return value is not None and any(x is not None and abs(x - value) < 1e-6 for x in results)


def gate_features(record, s, b) -> list[float]:
    """Features known before a hedge runs: step, parse, arithmetic, grounding, confidence, disagreement with greedy."""
    own = record["greedy"]["steps"][s]
    disagree = b.get("result") is None or own.get("result") is None or abs(b["result"] - own["result"]) > 1e-6
    return [float(s == k) for k in range(1, 7)] + [float(b["parsed"]), float(b["valid"]), float(b["grounded"]),
                                                   float(b["logprob"]), float(disagree)]


def fit_logistic(x, y, l2=1.0, steps=2000, lr=0.5):
    mean, std = x.mean(0), x.std(0) + 1e-6
    z = (x - mean) / std
    w, b = np.zeros(z.shape[1]), 0.0
    for _ in range(steps):
        p = 1 / (1 + np.exp(-(z @ w + b)))
        w -= lr * (z.T @ (p - y) / len(y) + l2 * w / len(y))
        b -= lr * float((p - y).mean())
    return lambda q: 1 / (1 + np.exp(-(((q - mean) / std) @ w + b)))


runs = [[json.loads(x) for x in path.read_text().splitlines()] for path in args.runs]
n = len(runs[0])
depth = len(runs[0][0]["branches"])
greedy = [[r["greedy"]["correct"] for r in recs] for recs in runs]
answer_tokens = args.answer_tokens
decoder_tokens = args.decoder_tokens
if decoder_tokens is None:
    try:
        from transformers import GPT2Tokenizer
        tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        texts = [b["text"] for b in [x for bs in runs[0][0]["branches"] for x in bs]] + \
                [b["text"] for r in runs[0] for bs in r["branches"] for b in bs]
        decoder_tokens = sum(len(tokenizer.encode(t)) + 1 for t in texts) / len(texts)  # end-of-text included
    except Exception as error:  # the tokenizer may be offline
        print(f"(decoder-step length not measured: {error}; using 9)")
        decoder_tokens = 9.0
greedy_cost = depth + answer_tokens
chain_cost = depth + answer_tokens
print(f"E0 on {len(runs)} run(s), n={n} each. Greedy {' / '.join(str(sum(g)) for g in greedy)}. Costs in "
      f"token-forwards: decoded step {decoder_tokens:.1f}, answer {answer_tokens:.1f}, one chain {chain_cost:.0f}.")

print("\nO1. Diversity: per (question, step), greedy plus 3 reruns")
for recs in runs:
    distinct = Counter()
    lonely = shared = 0
    rescue_counts = Counter()
    for r in recs:
        for s, bs in enumerate(r["branches"], 1):
            group = [r["greedy"]] + bs
            distinct[len({("gold" if e["correct"] else e["prediction"]) for e in group})] += 1
            right = sum(e["correct"] for e in group)
            if 0 < right < len(group):
                lonely += right == 1
                shared += right >= 2
        if not r["greedy"]["correct"]:
            right = sum(b["correct"] for bs in r["branches"] for b in bs)
            if right:
                rescue_counts[min(right, 4)] += 1
    print(f"  distinct answers per 4-set {dict(sorted(distinct.items()))}; mixed sets with the right answer held by "
          f"one candidate {lonely}, by two or more {shared}")
    print(f"  questions greedy gets wrong with k of 18 branches right (4 = four or more): "
          f"{dict(sorted(rescue_counts.items()))}")

print("\nO2. Vote-reachable coverage (the gate): greedy right, or at least 2 branches right")
reach_gain = []
for recs, g in zip(runs, greedy):
    pick = sum(gi or any(b["correct"] for bs in r["branches"] for b in bs) for gi, r in zip(g, recs))
    reach = sum(gi or sum(b["correct"] for bs in r["branches"] for b in bs) >= 2 for gi, r in zip(g, recs))
    reach_gain.append((reach - sum(g)) / n)
    print(f"  pick-reachable {pick} (+{pick - sum(g)}), vote-reachable {reach} (+{reach - sum(g)}) of {n}")
mean_gain = sum(reach_gain) / len(reach_gain)
gate = "PASS: keep-alive multi-select is worth building" if mean_gain >= GATE else \
    "DROP: keep-alive multi-select has too little vote-reachable room"
print(f"  gate: vote-reachable coverage adds {mean_gain:.1%} over greedy (bar {GATE:.0%}) -> {gate}")

print("\nO3. Fixed hedging policies, V1 vote, greedy spine (accuracy per seed)")
read_cost = depth * 3 * decoder_tokens + depth * decoder_tokens  # every rerun read, and greedy's slots
policies = {
    **{f"(i) all 3 reruns at step {s}": (lambda r, s=s: [(s, b) for b in r["branches"][s - 1]], False)
       for s in range(1, depth + 1)},
    "(ii) all reruns at steps 1-3": (lambda r: [(s, b) for s in (1, 2, 3) for b in r["branches"][s - 1]], False),
    "(iii) all 18 reruns": (lambda r: [(s, b) for s in range(1, depth + 1) for b in r["branches"][s - 1]], False),
    "(iv) reruns whose decoded result differs from greedy's": (
        lambda r: [(s, b) for s in range(1, depth + 1) for b in r["branches"][s - 1]
                   if gate_features(r, s, b)[-1]], True),
    "(v) reruns whose rule rank ties or beats candidate 0's": (
        lambda r: [(s, b) for s in range(1, depth + 1) for b in r["branches"][s - 1]
                   if rank(b) >= rank(r["greedy"]["steps"][s])], True),
}
for name, (policy, reads) in policies.items():
    results, costs = [], []
    for recs in runs:
        out = []
        for r in recs:
            kept = policy(r)
            out.append(vote(r["greedy"], [b for _, b in kept]))
            # A kept hedge pays its rerun step unless every rerun was already run to be read.
            costs.append(greedy_cost + sum((0 if reads else 1) + (depth - s) + answer_tokens for s, _ in kept)
                         + (read_cost + depth * 3 if reads else 0))
        results.append(out)
    tally(name, results, greedy, sum(costs) / len(costs))

print("\nO4. Commit to the rule's pick at one step, or vote over every candidate tied for the rule's best rank")
for s in range(1, depth + 1):
    commit, hedge = [], []
    for recs in runs:
        c_out, h_out = [], []
        for r in recs:
            group = [{**r["greedy"]["steps"][s], "prediction": r["greedy"]["prediction"],
                      "correct": r["greedy"]["correct"]}] + r["branches"][s - 1]
            k = choose(group)
            c_out.append(group[k]["correct"])
            best = max(rank(e) for e in group)
            h_out.append(vote(r["greedy"], [e for e in group[1:] if rank(e) == best]))
        commit.append(c_out)
        hedge.append(h_out)
    tally(f"step {s}: commit to the rule's pick", commit, greedy)
    tally(f"step {s}: vote over the best-ranked", hedge, greedy)

print("\nO5. Votes over branches against votes over chains (V1, greedy as spine)")
tally("greedy + 18 branches", [[vote(r["greedy"], [b for bs in r["branches"] for b in bs]) for r in recs]
                                for recs in runs], greedy,
      greedy_cost + sum(1 + (depth - s) + answer_tokens for s in range(1, depth + 1) for _ in range(3)))
chains = len(runs[0][0]["chains"])
tally(f"greedy + {chains} chains", [[vote(r["greedy"], r["chains"]) for r in recs] for recs in runs], greedy,
      greedy_cost + chains * chain_cost)
branch_cost = sum(1 + (depth - s) + answer_tokens for s in range(1, depth + 1) for _ in range(3))
matched = int(round(branch_cost / chain_cost))
tally(f"greedy + {matched} chains (the branches' cost)", [[vote(r["greedy"], r["chains"][:matched]) for r in recs]
                                                           for recs in runs], greedy, greedy_cost + matched * chain_cost)
print("  one dropout chain alone, expected correct: "
      + " / ".join(f"{sum(sum(c['correct'] for c in r['chains']) / chains for r in recs):.1f}" for recs in runs))

print("\nO6. Level-balanced vote (each step's reruns share weight 1) against the flat vote")
tally("level-balanced, all 18 reruns", [[vote(r["greedy"], [b for bs in r["branches"] for b in bs],
                                              [1 / len(bs) for bs in r["branches"] for _ in bs]) for r in recs]
                                         for recs in runs], greedy)

print("\nO7. Decoded results as a consistency signal")
for recs in runs:
    same = Counter()
    pairs = wins = ties = 0
    for r in recs:
        for s, bs in enumerate(r["branches"], 1):
            own = r["greedy"]["steps"][s].get("result")
            for b in bs:
                agrees = own is not None and b.get("result") is not None and abs(b["result"] - own) < 1e-6
                same[(agrees, b["prediction"] == r["greedy"]["prediction"])] += 1
            group = [{"prediction": r["greedy"]["prediction"], "correct": r["greedy"]["correct"], "s": s, "b": None}] + \
                    [{**b, "s": s, "b": b} for b in bs]
            if any(e["correct"] for e in group) and not all(e["correct"] for e in group):
                scores = [branch_answer_in_results(r, s, e["b"]) if e["b"] else
                          branch_answer_in_results(r, s, {"prediction": e["prediction"],
                                                          "result": r["greedy"]["steps"][s].get("result")})
                          for e in group]
                right = [x for x, e in zip(scores, group) if e["correct"]]
                wrong = [x for x, e in zip(scores, group) if not e["correct"]]
                for x in right:
                    for y in wrong:
                        pairs, wins, ties = pairs + 1, wins + (x > y), ties + (x == y)
    p_same = same[(True, True)] / max(1, same[(True, True)] + same[(True, False)])
    p_diff = same[(False, True)] / max(1, same[(False, True)] + same[(False, False)])
    print(f"  P(same final answer as greedy | same decoded result) {p_same:.3f}, | different decoded result {p_diff:.3f}; "
          f"AUC of 'answer appears among decoded results' in mixed sets "
          f"{(wins + 0.5 * ties) / pairs if pairs else float('nan'):.3f}")

print("\nO8. A learned keep-gate from decoder features, fit on one seed and scored on the other")
if len(runs) >= 2:
    for train, test in ((0, 1), (1, 0)):
        x = np.array([gate_features(r, s, b) for r in runs[train] for s, bs in enumerate(r["branches"], 1) for b in bs])
        y = np.array([float(b["correct"]) for r in runs[train] for bs in r["branches"] for b in bs])
        predict = fit_logistic(x, y)
        line = []
        for threshold in (0.9, 0.7, 0.5, 0.3, 0.1):
            out, hedges = [], 0
            for r in runs[test]:
                kept = [b for s, bs in enumerate(r["branches"], 1) for b in bs
                        if predict(np.array(gate_features(r, s, b))) >= threshold]
                hedges += len(kept)
                out.append(vote(r["greedy"], kept))
            wins = sum(a and not b for a, b in zip(out, greedy[test]))
            losses = sum(b and not a for a, b in zip(out, greedy[test]))
            line.append(f"t={threshold}: {sum(out)} ({wins - losses:+d}, {hedges / n:.1f} hedges)")
        print(f"  fit on seed run {train + 1}, scored on run {test + 1}: " + "; ".join(line))
else:
    print("  needs two runs")

print("\nO9. Where to hedge against which candidate to trust (oracles)")
for recs, g in zip(runs, greedy):
    where = sum(gi or any(vote(r["greedy"], r["branches"][s - 1]) for s in range(1, depth + 1)) for gi, r in zip(g, recs))
    which = sum(gi or any(b["correct"] for bs in r["branches"] for b in bs) for gi, r in zip(g, recs))
    print(f"  best single step to hedge, then vote: {where} (+{where - sum(g)}); best single branch: {which} "
          f"(+{which - sum(g)})")

print(f"\nE0 gate: {gate}")
