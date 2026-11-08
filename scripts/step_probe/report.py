"""The rules of the probe diagnosis (results/step_probe/PLAN.md).

    python scripts/step_probe/report.py <run output dir>

Per split, the verdict is INCONCLUSIVE, READOUT-LIMITED, COMPUTE-LIMITED or MIXED, from the answer probe's readings on
greedy-wrong questions, and the break profile is EARLY, LATE, SPREAD or NONE, from the first step that cannot be read.
The headline is validation.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("run", type=Path)
parser.add_argument("--max-steps", type=int, default=6)
args = parser.parse_args()

SENSITIVITY, HOLD, CONTROL, LOW, EXCESS = 0.5, 0.4, 2.0, 0.2, 2.0

with np.load(args.run / "readings.npz") as archive:
    data = {key: archive[key] for key in archive.files}
layers = [int(x) for x in data["layers"]]
splits = {
    "train": ([json.loads(x) for path in sorted(args.run.glob("train/chunk_*.jsonl")) for x in path.read_text().splitlines()],
              data["train"], data["train_targets"]),
    "valid": ([json.loads(x) for x in (args.run / "valid/records.jsonl").read_text().splitlines()],
              data["valid"], data["valid_targets"]),
}
positions = data["train"].shape[-1]
END = [positions - 2, positions - 1]  # the last latent step and <eot>
status = json.loads((args.run / "status.json").read_text()) if (args.run / "status.json").exists() else {}


def as_int(text):
    try:
        value = float(str(text).replace(",", ""))
    except ValueError:
        return None
    return int(value) if value == int(value) and 0 <= value <= 9999 else None


def rate(hits) -> float:
    hits = np.asarray(hits)
    return float(hits.mean()) if hits.size else float("nan")


def name(position) -> str:
    return "<bot>" if position == 0 else "<eot>" if position == positions - 1 else f"step {position}"


records, readings, targets = splits["train"]
right = np.array([r["correct"] for r in records])
steps = np.array([r["steps"] for r in records])

# The answer reading point, chosen on greedy-right training questions.
usable = right & (targets[:, -1] >= 0)
best = max(((rate(readings[usable, -1, l, p] == targets[usable, -1]), l, p) for l in range(len(layers)) for p in END))
_, L, P = best
# Each step's reading point, chosen the same way.
step_point = {}
for j in range(1, args.max_steps + 1):
    rows = right & (steps >= j) & (targets[:, j - 1] >= 0)
    if rows.any():
        step_point[j] = max(((rate(readings[rows, j - 1, l, p] == targets[rows, j - 1]), l, p)
                             for l in range(len(layers)) for p in range(positions)))[1:]

print(f"Identity: {status.get('valid', {}).get('identity_matched')} of {status.get('valid', {}).get('questions')} "
      f"validation predictions match the reference")
train_status = status.get("train", {})
print(f"Training extraction: {train_status.get('done')} of {train_status.get('selected')} questions, "
      f"{train_status.get('seconds', 0):.0f}s{', stopped at the cap' if train_status.get('capped') else ''}")
print(f"Answer reading point: layer {layers[L]}, {name(P)}")
print("Step reading points: " + ", ".join(f"c{j} at layer {layers[l]}, {name(p)}" for j, (l, p) in step_point.items()))

verdicts = {}
for split, (records, readings, targets) in splits.items():
    right = np.array([r["correct"] for r in records])
    steps = np.array([r["steps"] for r in records])
    gold = targets[:, -1]
    in_range = (steps >= 1) & (steps <= args.max_steps) & (gold >= 0)
    prediction = np.array([-1 if as_int(r["prediction"]) is None else as_int(r["prediction"]) for r in records])
    s = rate(readings[right & in_range, -1, L, P] == gold[right & in_range])
    scope = ~right & in_range & (prediction >= 0) & (prediction != gold)
    at_end, at_bot = readings[scope, -1, L, P], readings[scope, -1, L, 0]
    g, w = rate(at_end == gold[scope]), rate(at_end == prediction[scope])
    g0, w0 = rate(at_bot == gold[scope]), rate(at_bot == prediction[scope])
    if s < SENSITIVITY:
        verdict = "INCONCLUSIVE"
    elif g / s >= HOLD and g >= CONTROL * g0:
        verdict = "READOUT-LIMITED"
    elif w / s >= HOLD and g / s < LOW:
        verdict = "COMPUTE-LIMITED"
    else:
        verdict = "MIXED"

    # The break profile: the first step whose reading is not exact, on 3+-step questions with every value valid.
    fractions = {}
    for label, group in (("right", right), ("wrong", ~right)):
        early = late = total = 0
        for i in np.flatnonzero(group & (steps >= 3) & (steps <= args.max_steps)):
            n = steps[i]
            if (targets[i, :n] < 0).any() or any(j not in step_point for j in range(1, n + 1)):
                continue
            total += 1
            first = next((j for j in range(1, n + 1)
                          if readings[i, j - 1, step_point[j][0], step_point[j][1]] != targets[i, j - 1]), None)
            if first is not None:
                if (first - 1) / (n - 1) <= 0.5:
                    early += 1
                else:
                    late += 1
        fractions[label] = (early / max(1, total), late / max(1, total), total)
    e_early = max(0.0, fractions["wrong"][0] - fractions["right"][0])
    e_late = max(0.0, fractions["wrong"][1] - fractions["right"][1])
    profile = ("NONE" if e_early == 0 and e_late == 0 else "EARLY" if e_early >= EXCESS * e_late
               else "LATE" if e_late >= EXCESS * e_early else "SPREAD")
    verdicts[split] = (verdict, profile)

    print(f"\n{split}: {len(records)} questions, greedy right {int(right.sum())}; wrong in scope {int(scope.sum())} "
          f"(wrong with a non-integer or out-of-range prediction: {int((~right & in_range & (prediction < 0)).sum())})")
    print(f"  sensitivity s = {s:.3f} (gold read exactly on greedy-right questions)")
    print(f"  wrong questions, reading at the answer point: gold g = {g:.3f} (g/s = {g / s if s else float('nan'):.2f}), "
          f"own prediction w = {w:.3f} (w/s = {w / s if s else float('nan'):.2f}), other {1 - g - w:.3f}")
    print(f"  control at <bot>: gold g0 = {g0:.3f}, own prediction w0 = {w0:.3f}")
    print(f"  break profile (3+ steps): right early {fractions['right'][0]:.3f} late {fractions['right'][1]:.3f} "
          f"(n={fractions['right'][2]}); wrong early {fractions['wrong'][0]:.3f} late {fractions['wrong'][1]:.3f} "
          f"(n={fractions['wrong'][2]}); excess early {e_early:.3f}, late {e_late:.3f}")
    print("  by step count: n | greedy right | g | w")
    for n in range(1, args.max_steps + 1):
        rows = scope & (steps == n)
        print(f"    {n} | {int((right & (steps == n)).sum())}/{int((steps == n).sum())} | "
              f"{rate(readings[rows, -1, L, P] == gold[rows]):.2f} | {rate(readings[rows, -1, L, P] == prediction[rows]):.2f}"
              f" (wrong {int(rows.sum())})")
    print(f"  answer exact rate by position at layer {layers[L]}, right | wrong-reads-gold | wrong-reads-own:")
    for p in range(positions):
        print(f"    {name(p):>7}: {rate(readings[right & in_range, -1, L, p] == gold[right & in_range]):.3f} | "
              f"{rate(readings[scope, -1, L, p] == gold[scope]):.3f} | "
              f"{rate(readings[scope, -1, L, p] == prediction[scope]):.3f}")
    print("  step exact rate on greedy-right questions, by position (each step at its reading point's layer):")
    for j, (l, _) in step_point.items():
        rows = right & (steps >= j) & (targets[:, j - 1] >= 0)
        cells = " ".join(f"{rate(readings[rows, j - 1, l, p] == targets[rows, j - 1]):.2f}" for p in range(positions))
        literal = rate([r["in_question"][j - 1] for r, keep in zip(records, rows) if keep])
        print(f"    c{j} (layer {layers[l]}, n={int(rows.sum())}, {literal:.0%} literally in the question): {cells}")
    print("  gold exact rate on greedy-right questions by answer size:")
    for low, high in ((0, 10), (10, 100), (100, 1000), (1000, 10000)):
        rows = right & in_range & (gold >= low) & (gold < high)
        print(f"    [{low}, {high}): {rate(readings[rows, -1, L, P] == gold[rows]):.3f} (n={int(rows.sum())})")

print()
for split in ("train", "valid"):
    verdict, profile = verdicts[split]
    print(f"Step probe, {split}: {verdict}; break profile {profile}")
verdict, profile = verdicts["valid"]
print(f"\nStep probe diagnosis (validation): {verdict}, break profile {profile}")
