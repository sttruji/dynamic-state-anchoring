"""Check E1 of results/latent_gate/heldout_scale/PLAN.md: the batched evaluator's predictions against Stage 1's batch-1 records.

    python scripts/latent_gate/check_batched.py <batched gsm8k_valid.jsonl> <stage1 base gsm8k_valid.jsonl> [--threshold 0.99]

Exits non-zero unless every policy and depth the two share agrees on at least `threshold` of the questions.
"""
import argparse
import json
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("batched", type=Path)
parser.add_argument("reference", type=Path)
parser.add_argument("--threshold", type=float, default=0.99)
args = parser.parse_args()
load = lambda path: {r["id"]: r for r in map(json.loads, path.read_text().splitlines())}
batched, reference = load(args.batched), load(args.reference)
ids = sorted(set(batched) & set(reference))
assert ids, "no questions in common"
first, other = batched[ids[0]], reference[ids[0]]
shared = [p for p in first if isinstance(first[p], dict) and p in other]
failed = []
print(f"{len(ids)} questions in common")
for policy in shared:
    for depth in sorted(set(first[policy]) & set(other[policy]), key=int):
        same = sum(batched[i][policy][depth]["prediction"] == reference[i][policy][depth]["prediction"] for i in ids)
        share = same / len(ids)
        print(f"  {policy:10} @{depth:>2}: {same}/{len(ids)} predictions agree ({share:.1%})")
        if share < args.threshold:
            failed.append(f"{policy}@{depth}")
if failed:
    raise SystemExit(f"E1 fails below {args.threshold:.0%}: {', '.join(failed)}; use the batch-1 evaluator")
print(f"E1 passes: every shared policy and depth agrees on at least {args.threshold:.0%} of questions")
