"""D1-R's rules (results/ffn_sparsity/d1r/PLAN.md) on a run folder.

    python scripts/ffn_sparsity/report_d1r.py <run dir>

The folder holds evals/<name>.jsonl from eval_moe.py (base-dense, prune-0, moe-0, hash-0 and <arm>-s<seed> for the moe,
prune and hash arms) and training/dense-s7-check/check.json from train_moe.py. The dense control is Stage 1b's ft6_step.
Comparisons pool every (seed, question) pair as scripts/inner_loop/pool.py does: the mean difference per seed and the
exact two-sided sign test.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from moe import moe_active_params  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("run", type=Path)
parser.add_argument("--ft6", type=Path, nargs=2, default=[
    ROOT / "results/inner_loop/stage1b/evals/ft6_step/gsm8k_valid.jsonl",
    ROOT / "results/inner_loop/stage1b-seed8/evals/ft6_step/gsm8k_valid.jsonl"], help="ft6_step at seeds 7 and 8")
parser.add_argument("--seeds", type=int, nargs=2, default=[7, 8])
parser.add_argument("--kill", type=int, default=192, help="the early kill: MoE at the first seed at or below this")
args = parser.parse_args()
ARMS = ("moe", "prune", "hash")
EMBEDDINGS = 50260 * 768 + 1024 * 768  # the token table (tied to the LM head) and the position table


def sign_p(wins: int, losses: int) -> float:
    """Exact two-sided sign test."""
    n = wins + losses
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(min(wins, losses) + 1)) / 2 ** n) if n else 1.0


def load(name):
    path = args.run / f"evals/{name}.jsonl"
    return {r["id"]: r for r in map(json.loads, path.read_text().splitlines())} if path.exists() else None


def summary(name):
    path = args.run / f"evals/{name}.json"
    return json.loads(path.read_text()) if path.exists() else None


ft6 = {seed: {r["id"]: {"correct": bool(r["arms"]["plain"]["correct"]), "rationale_steps": r["rationale_steps"]}
              for r in map(json.loads, path.read_text().splitlines())} for seed, path in zip(args.seeds, args.ft6)}
runs = {(arm, seed): load(f"{arm}-s{seed}") for arm in ARMS for seed in args.seeds}
runs.update({("ft6", seed): ft6[seed] for seed in args.seeds})


def right(rows):
    return sum(r["correct"] for r in rows.values())


def compare(first, second):
    """first - second pooled over seeds: (mean per seed, wins, losses, p, per-seed lower bound at 1.645)."""
    wins = losses = 0
    for seed in args.seeds:
        a, b = runs[(first, seed)], runs[(second, seed)]
        assert a.keys() == b.keys(), f"{first} and {second} at seed {seed} scored different questions"
        wins += sum(a[i]["correct"] and not b[i]["correct"] for i in a)
        losses += sum(b[i]["correct"] and not a[i]["correct"] for i in a)
    n = len(args.seeds)
    return ((wins - losses) / n, wins, losses, sign_p(wins, losses),
            (wins - losses - 1.645 * math.sqrt(wins + losses)) / n)


def show(label, result):
    mean, wins, losses, p, bound = result
    print(f"  {label}: {mean:+.1f} per seed (+{wins}/-{losses}, p={p:.3f}; per-seed lower bound {bound:+.1f})")


def matches(arm):
    mean, _, _, _, bound = compare(arm, "ft6")
    return mean >= -5 and bound > -12


print(f"D1-R on {args.run}")
checks = {}
base, prune0 = summary("base-dense"), summary("prune-0")
check_path = args.run / "training/dense-s7-check/check.json"
dense = json.loads(check_path.read_text()) if check_path.exists() else None
checks["R0a"] = bool(base) and base["matched"] == base["questions"]
checks["R0b"] = bool(prune0) and prune0["matched"] == prune0["questions"]
checks["R0c"] = bool(dense) and dense["passed"]
print(f"R0a, the evaluation path: base SIM-CoT matched step selection's greedy answers on "
      f"{base['matched'] if base else None} of {base['questions'] if base else None}")
print(f"R0b, the conversion: Prune-50 untrained matched SPARSE-0's S50 on "
      f"{prune0['matched'] if prune0 else None} of {prune0['questions'] if prune0 else None}")
print(f"R0c, the training path: dense losses within {dense['tolerance'] if dense else None} of Stage 1b's ft6_step at "
      f"updates {dense['updates'] if dense else None}: max gap {dense['max_gap'] if dense else None}")

print("\nTraining-free (converted, not fine-tuned):")
for name in ("moe-0", "hash-0", "prune-0"):
    rows = load(name)
    print(f"  {name}: {f'{right(rows)} of {len(rows)}' if rows else None}")
print("\nFine-tuned, per seed (ft6_step is the dense control):")
for seed in args.seeds:
    cells = [f"{arm} {right(runs[(arm, seed)]) if runs[(arm, seed)] else '-'}" for arm in ("ft6",) + ARMS]
    print(f"  seed {seed}: " + "; ".join(cells))
complete = all(runs[(arm, seed)] is not None for arm in ARMS for seed in args.seeds)
if complete:
    print("\nPooled over both seeds:")
    for arm in ARMS:
        show(f"{arm} - ft6", compare(arm, "ft6"))
    show("moe - prune", compare("moe", "prune"))
    show("moe - hash", compare("moe", "hash"))
    print("\nRight answers by step count, summed over seeds (exploratory):")
    for arm in ("ft6",) + ARMS:
        short = sum(r["correct"] for s in args.seeds for r in runs[(arm, s)].values() if r["rationale_steps"] <= 3)
        long = sum(r["correct"] for s in args.seeds for r in runs[(arm, s)].values() if r["rationale_steps"] >= 4)
        print(f"  {arm:5}: <=3 steps {short}, 4+ steps {long}")

agreement = {name: summary(name) for name in ("moe-0", "hash-0") + tuple(f"moe-s{s}" for s in args.seeds)}
if any(s and "oracle_agreement" in s for s in agreement.values()):
    print("\nHow often the chosen experts are among the oracle's top 4, mean over layers (chance 0.33):")
    for name, s in agreement.items():
        if s and "oracle_agreement" in s:
            values = s["oracle_agreement"]
            cells = [f"{p} {sum(v for k, v in values.items() if k.endswith('/' + p)) / max(1, sum(k.endswith('/' + p) for k in values)):.2f}"
                     for p in "QTA"]
            print(f"  {name}: " + ", ".join(cells))

dense_params = moe_active_params(None)
print("\nActive params per token, non-embedding (and with the embeddings):")
for arm in (None,) + ARMS:
    params = moe_active_params(arm)
    print(f"  {arm or 'dense':5}: {params / 1e6:.2f}M, {params / dense_params:.1%} "
          f"({(params + EMBEDDINGS) / 1e6:.1f}M, {(params + EMBEDDINGS) / (dense_params + EMBEDDINGS):.1%})")

first = runs[("moe", args.seeds[0])]
if not all(checks.values()):
    verdict = f"NO VERDICT: {', '.join(k for k, ok in checks.items() if not ok)} failed or missing, fix the code"
elif first is not None and right(first) <= args.kill:
    verdict = f"NEGATIVE, early kill: MoE at seed {args.seeds[0]} scored {right(first)} <= {args.kill}"
elif not complete:
    verdict = "INCOMPLETE: not every fine-tuned arm has been evaluated"
else:
    mean, _, _, _, bound = compare("moe", "ft6")
    gap, _, _, p, _ = compare("moe", "prune")
    if not matches("moe"):
        verdict = "INCONCLUSIVE: MoE's estimate is within 5 but its bound misses" if mean >= -5 else \
            "NEGATIVE: MoE does not match the dense control"
    elif matches("prune"):
        verdict = "PRUNE-SUFFICES: MoE matches, and so does the pruned model"
    elif not (gap >= 10 and p < 0.05):
        verdict = "MATCH-UNRESOLVED: MoE matches, but its edge over pruning misses +10 with p < 0.05"
    elif compare("moe", "hash")[0] < 5:
        verdict = "CAPACITY-NOT-ROUTING: MoE matches and beats pruning, but the hash router does nearly as well"
    else:
        verdict = "MoE-MATCH: MoE matches the dense control and beats both pruning and the hash router"
print(f"\nD1-R verdict: {verdict}.")
