"""SPARSE-0's rules (results/ffn_sparsity/PLAN.md) on a run folder from run.py.

    python scripts/ffn_sparsity/report.py <run dir>

Every comparison is paired on the same questions with an exact two-sided sign test. B is ID's score.
R0 identity; R1 STOP if O50 <= B - 20; R2 STOP if O50 - S50 < +10 and O25 - S25 < +10; R3 GO otherwise, at 25% if
O25 >= B - 10, else at 50%; R4 phase: thinking is the sensitive phase if O25-T <= O25-Q - 10 with p < 0.05, reading in
the mirror case.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from sparsity import ARMS, FRACTIONS, active_params  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("run", type=Path)
args = parser.parse_args()


def sign_p(wins: int, losses: int) -> float:
    """Exact two-sided sign test."""
    n = wins + losses
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(min(wins, losses) + 1)) / 2 ** n) if n else 1.0


status = json.loads((args.run / "status.json").read_text())
evals = {name: {r["id"]: r for r in map(json.loads, (args.run / f"evals/{name}.jsonl").read_text().splitlines())}
         for name in ARMS}
ids = list(evals["ID"])
assert all(e.keys() == evals["ID"].keys() for e in evals.values()), "the arms scored different questions"
right = {name: sum(r["correct"] for r in e.values()) for name, e in evals.items()}
B = right["ID"]


def paired(first, second):
    """first - second on the same questions: (difference, wins, losses, p)."""
    wins = sum(evals[first][i]["correct"] and not evals[second][i]["correct"] for i in ids)
    losses = sum(evals[second][i]["correct"] and not evals[first][i]["correct"] for i in ids)
    return wins - losses, wins, losses, sign_p(wins, losses)


model = status.get("model", {"width": 768, "inner": 3072, "layers": 12})
dense = active_params(1.0, **model)
counts = [r["positions"] for r in evals["ID"].values()]  # positions fed per question: Q, T, A under the dense model


def flops_share(arm):
    """Non-embedding forward FLOPs per validation question, as a share of dense, at ID's positions."""
    if arm is None:
        return 1.0
    _, fraction, phases = arm
    total = sparse = 0
    for q, t, a in counts:
        for count, phase in ((q, "Q"), (t, "T"), (a, "A")):
            total += count * dense
            sparse += count * (active_params(fraction, **model) if phase in phases else dense)
    return sparse / total


identity = status.get("identity", {})
print(f"SPARSE-0 on {args.run}: {len(ids)} validation questions")
if identity.get("matched") is None:
    print("R0, identity: not checked (no reference given)")
else:
    print(f"R0, identity: ID matched the reference on {identity['matched']} of {identity['questions']}")
print(f"  B (ID's score) = {B}\n")
print(f"  {'arm':6} {'right':>5} {'vs ID':>14} {'p':>6} {'<=3 steps':>9} {'4+ steps':>8} {'same ans':>8} "
      f"{'active/token':>13} {'FLOPs':>6}")
for name, arm in ARMS.items():
    short = sum(r["correct"] for r in evals[name].values() if r["rationale_steps"] <= 3)
    long = sum(r["correct"] for r in evals[name].values() if r["rationale_steps"] >= 4)
    same = sum(evals[name][i]["prediction"] == evals["ID"][i]["prediction"] for i in ids)
    params = dense if arm is None else active_params(arm[1], **model)
    line = f"  {name:6} {right[name]:5}"
    if name == "ID":
        line += f" {'':>14} {'':>6}"
    else:
        diff, wins, losses, p = paired(name, "ID")
        line += f" {f'{diff:+d} (+{wins}/-{losses})':>14} {p:6.3f}"
    print(line + f" {short:9} {long:8} {same:8} {f'{params / 1e6:.2f}M {params / dense:.0%}':>13} "
                 f"{flops_share(arm):6.0%}")
print("  active/token: active non-embedding params at a sparsified position; FLOPs: per question, at ID's positions\n")

rules = {}
rules["R1"] = right["O50"] <= B - 20
gaps = {f: paired(f"O{round(f * 100)}", f"S{round(f * 100)}") for f in FRACTIONS}
rules["R2"] = all(gaps[f][0] < 10 for f in FRACTIONS)
print(f"R1, no headroom: O50 = {right['O50']} against B - 20 = {B - 20}: {'fires' if rules['R1'] else 'does not fire'}")
for f in FRACTIONS:
    diff, wins, losses, p = gaps[f]
    print(f"  O{round(f * 100)} - S{round(f * 100)} = {diff:+d} (+{wins}/-{losses}, p={p:.3f})")
print(f"R2, no routing value: both gaps under +10: {'fires' if rules['R2'] else 'does not fire'}")
if right["S50"] >= B - 10:
    print(f"  Pruning note: S50 = {right['S50']} >= B - 10, so static pruning to 50% of the FFN costs at most 10 "
          f"training-free.")
diff, wins, losses, p = paired("O25-T", "O25-Q")
print(f"R4, phase: O25-T - O25-Q = {diff:+d} (+{wins}/-{losses}, p={p:.3f})")
if diff <= -10 and p < 0.05:
    phase = "thinking is the sensitive phase"
elif diff >= 10 and p < 0.05:
    phase = "reading is the sensitive phase"
else:
    phase = "no phase asymmetry at 25%"

overlap_path = args.run / "overlap.json"
if overlap_path.exists():
    overlap = json.loads(overlap_path.read_text())
    print("\nOverlap of the oracle's kept sets between neighbouring positions (Jaccard; random sets give f / (2 - f))")
    for f in FRACTIONS:
        q, t = overlap[str(f)]["Q"], overlap[str(f)]["T"]
        print(f"  {f:.0%} (random {f / (2 - f):.2f}): Q " + " ".join(f"{x:.2f}" for x in q))
        print(f"  {'':16} T " + " ".join(f"{x:.2f}" for x in t))
    print("  one column per layer, 1 to 12")

if identity.get("matched") is not None and identity.get("matched") != identity.get("questions"):
    verdict = "NO VERDICT: R0 fails, fix the code"
elif rules["R1"]:
    verdict = "STOP, no headroom (R1): the MoE line closes"
elif rules["R2"]:
    verdict = "STOP, no routing value (R2): the MoE line closes"
else:
    verdict = f"GO (R3): D1-R at {'25%' if right['O25'] >= B - 10 else '50%'}"
print(f"\nSPARSE-0 verdict: {verdict}. Phase (R4): {phase}.")
