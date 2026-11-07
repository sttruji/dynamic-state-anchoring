"""Report on Stage 0 of step-level selection (docs/brainstorm/STEP_SELECTION.md), pooled over sampling seeds.

    python scripts/step_select/report.py <seed1.jsonl> <seed2.jsonl> [--reference <inner_loop base-simcot jsonl>]

Per run: the identity check against the checkpoint's own answers, the room to gain from choosing among finished chains
and from choosing per step, per-step rescues and breaks, whether the fixed decoder rule picks right candidates more often
than chance, and the untrained selector against greedy and majority vote over 4 chains.

Decision rule, fixed before Stage 0 ran, pooled over the runs:
    go          the branches cover at least 8% of questions more than greedy (40 of 500), and the fixed rule picks a right
                candidate more often than chance in mixed sets at p < 0.05
    stop        the branches cover less than 4% more than greedy (20 of 500), or the rule shows no signal (p >= 0.05) and
                the untrained selector does no better than greedy
    ambiguous   anything else
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
parser.add_argument("runs", type=Path, nargs="+", help="eval.py outputs, one per sampling seed")
parser.add_argument("--reference", type=Path, help="the checkpoint's own answers: an inner_loop eval.py output with plain")
args = parser.parse_args()
sys.path.insert(0, str(Path(__file__).resolve().parent))
from stepsel import choose, rank  # noqa: E402


def sign_p(wins: int, losses: int) -> float:
    """Exact two-sided sign test."""
    n = wins + losses
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(min(wins, losses) + 1)) / 2 ** n) if n else 1.0


def paired(a, b):
    wins, losses = sum(x and not y for x, y in zip(a, b)), sum(y and not x for x, y in zip(a, b))
    return f"{sum(a)} vs {sum(b)}: {sum(a) - sum(b):+d} (+{wins}/-{losses}, p={sign_p(wins, losses):.3f})", wins, losses


def majority(entries) -> bool:
    """Whether the most common prediction (first seen wins a tie) is correct."""
    votes = Counter(e["prediction"] for e in entries)
    top = max(votes.values())
    return next(e for e in entries if votes[e["prediction"]] == top)["correct"]


def pass_at(n: int, c: int, k: int) -> float:
    """The unbiased chance that k of n samples, c of them correct, include a correct one."""
    return 1.0 if n - c < k else 1.0 - math.comb(n - c, k) / math.comb(n, k)


def upper_tail(probabilities, hits: int) -> float:
    """P(at least `hits` successes) for independent Bernoulli trials with these probabilities."""
    dist = np.ones(1)
    for q in probabilities:
        dist = np.convolve(dist, [1 - q, q])
    return float(dist[hits:].sum())


def auc(sets, key) -> float | None:
    """Within each mixed set, the chance a right candidate outranks a wrong one by `key` (ties count half)."""
    wins = ties = pairs = 0
    for candidates in sets:
        right = [key(f) for f in candidates if f["correct"]]
        wrong = [key(f) for f in candidates if not f["correct"]]
        for x in right:
            for y in wrong:
                pairs += 1
                wins += x > y
                ties += x == y
    return (wins + 0.5 * ties) / pairs if pairs else None


def mixed_sets(records):
    """Per (question, step): candidate 0 (greedy's own step and outcome) and the reruns, when right and wrong mix."""
    out = []
    for r in records:
        for s, reruns in enumerate(r["branches"], 1):
            candidates = [{**r["greedy"]["steps"][s], "correct": r["greedy"]["correct"]}] + reruns
            if any(c["correct"] for c in candidates) and not all(c["correct"] for c in candidates):
                out.append((s, candidates))
    return out


reference = None
if args.reference:
    reference = {r["id"]: r["arms"]["plain"]["text"] for r in map(json.loads, args.reference.read_text().splitlines())}
pooled = {"rows": 0, "headroom": [], "hits": 0, "chances": [], "rule_wins": 0, "rule_losses": 0}
for path in args.runs:
    records = [json.loads(x) for x in path.read_text().splitlines()]
    n = len(records)
    greedy = [r["greedy"]["correct"] for r in records]
    print(f"\n{path}: n={n}, greedy {sum(greedy)}")
    if reference:
        same = sum(reference.get(r["id"]) == r["greedy"]["text"] for r in records)
        print(f"  identity: greedy gives the checkpoint's own answer on {same}/{n} "
              f"({'PASS' if same == n else 'FAIL'})")

    chains = len(records[0]["chains"])
    per_k = {k: sum(pass_at(chains, sum(c["correct"] for c in r["chains"]), k) for r in records) for k in (1, 4, chains)}
    chain_cover = sum(g or any(c["correct"] for c in r["chains"]) for g, r in zip(greedy, records))
    branch_cover = sum(g or any(b["correct"] for bs in r["branches"] for b in bs) for g, r in zip(greedy, records))
    reruns = sum(len(bs) for bs in records[0]["branches"])
    print(f"  choosing among finished chains: any of k sampled chains right, k=1 {per_k[1]:.1f}, k=4 {per_k[4]:.1f}, "
          f"k={chains} {per_k[chains]:.1f}; greedy or any of {chains} chains {chain_cover} (+{chain_cover - sum(greedy)})")
    print(f"  choosing per step: greedy or any of {reruns} one-step branches {branch_cover} "
          f"(+{branch_cover - sum(greedy)} over greedy)")
    print("  per step: rescued (greedy wrong, a branch right) / broken (greedy right, a branch wrong) / answers changed")
    for s in range(1, len(records[0]["branches"]) + 1):
        branch = [r["branches"][s - 1] for r in records]
        rescued = sum(not g and any(b["correct"] for b in bs) for g, bs in zip(greedy, branch))
        broken = sum(g and any(not b["correct"] for b in bs) for g, bs in zip(greedy, branch))
        changed = sum(b["prediction"] != r["greedy"]["prediction"] for r, bs in zip(records, branch) for b in bs)
        print(f"    step {s}: {rescued:3} / {broken:3} / {changed}/{sum(len(bs) for bs in branch)}")

    sets = mixed_sets(records)
    hits = sum(c[choose(c)]["correct"] for _, c in sets)
    chances = [sum(x["correct"] for x in c) / len(c) for _, c in sets]
    keeps_own = sum(choose(c) == 0 for _, c in sets)
    print(f"  signal in {len(sets)} mixed sets: the fixed rule picks a right candidate {hits} times, chance "
          f"{sum(chances):.1f} (p={upper_tail(chances, hits):.3f}); it keeps candidate 0 in {keeps_own}")
    moves = [c for _, c in sets if choose(c) != 0]
    print(f"    always keeping candidate 0 would be right {sum(c[0]['correct'] for _, c in sets)} times; where the rule "
          f"moves off it ({len(moves)} sets), its pick is right {sum(c[choose(c)]['correct'] for c in moves)} times "
          f"and candidate 0 {sum(c[0]['correct'] for c in moves)}")
    for name, key in (("valid arithmetic", lambda f: int(f["valid"])), ("grounded operands", lambda f: int(f["grounded"])),
                      ("decoder confidence", lambda f: f["logprob"]), ("rule rank", rank)):
        value = auc([c for _, c in sets], key)
        print(f"    AUC {name}: {'n/a' if value is None else f'{value:.3f}'}")
    for s in range(1, len(records[0]["branches"]) + 1):
        at = [c for t, c in sets if t == s]
        print(f"    step {s}: {len(at)} mixed sets, rule right {sum(c[choose(c)]['correct'] for c in at)}, "
              f"chance {sum(sum(x['correct'] for x in c) / len(c) for c in at):.1f}")

    rule = [r["rule"]["correct"] for r in records]
    vs_greedy, wins, losses = paired(rule, greedy)
    vs_majority, _, _ = paired(rule, [majority(r["chains"][:4]) for r in records])
    moved = sum(any(k != 0 for k in r["rule"]["chosen"]) for r in records)
    print(f"  untrained selector vs greedy {vs_greedy}; vs majority of 4 chains {vs_majority}; "
          f"it left candidate 0 somewhere on {moved}/{n} questions")
    print(f"  majority of 4 chains {sum(majority(r['chains'][:4]) for r in records)}, "
          f"of {chains} chains {sum(majority(r['chains']) for r in records)}")

    pooled["rows"] += n
    pooled["headroom"].append((branch_cover - sum(greedy)) / n)
    pooled["hits"] += hits
    pooled["chances"] += chances
    pooled["rule_wins"] += wins
    pooled["rule_losses"] += losses

headroom = sum(pooled["headroom"]) / len(pooled["headroom"])
p_signal = upper_tail(pooled["chances"], pooled["hits"])
above = pooled["hits"] > sum(pooled["chances"])
gain = pooled["rule_wins"] - pooled["rule_losses"]
print(f"\nPooled over {len(args.runs)} run(s): branches cover {headroom:.1%} more than greedy; the rule is right in "
      f"{pooled['hits']} mixed sets against chance {sum(pooled['chances']):.1f} (p={p_signal:.3f}); untrained selector "
      f"vs greedy {gain:+d} (+{pooled['rule_wins']}/-{pooled['rule_losses']}, "
      f"p={sign_p(pooled['rule_wins'], pooled['rule_losses']):.3f})")
signal = above and p_signal < 0.05
if headroom >= 0.08 and signal:
    verdict = "GO: train the RL selector (Stage 1)"
elif headroom < 0.04 or (not signal and gain <= 0):
    verdict = "STOP: choosing per step has too little room, or the decoder cannot tell candidates apart"
else:
    verdict = "AMBIGUOUS: room without a decoder signal, or the reverse; decide whether an RL selector on hidden states is worth it"
print(f"  {verdict}")
