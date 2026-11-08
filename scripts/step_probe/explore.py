"""Exploratory reads of the probe diagnosis, never gating (results/step_probe/PLAN.md, "Reported, never gating").

    python scripts/step_probe/explore.py <run output dir>

The answer probe is too weak for the verdict, but the probes for the first steps are not. This asks whether wrong chains
hold the gold first and second step values less often than right chains do, within a step count and a value size, and
whether a misread first step is just another gold step done first.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("run", type=Path)
args = parser.parse_args()

with np.load(args.run / "readings.npz") as archive:
    data = {key: archive[key] for key in archive.files}
layers = [int(x) for x in data["layers"]]
splits = {
    "train": ([json.loads(x) for path in sorted(args.run.glob("train/chunk_*.jsonl")) for x in path.read_text().splitlines()],
              data["train"], data["train_targets"]),
    "valid": ([json.loads(x) for x in (args.run / "valid/records.jsonl").read_text().splitlines()],
              data["valid"], data["valid_targets"]),
}
FIRST = layers.index(3)  # every early step's reading point is layer 3 at the step's own input position


def exact(readings, targets, rows, j, position) -> str:
    hits = readings[rows, j - 1, FIRST, position] == targets[rows, j - 1]
    return f"{hits.mean():.2f} (n={int(rows.sum())})" if rows.any() else "-"


for split, (records, readings, targets) in splits.items():
    right = np.array([r["correct"] for r in records])
    steps = np.array([r["steps"] for r in records])
    print(f"{split}: step j read at layer 3, position j (its own input latent), right | wrong, by step count")
    for j in (1, 2, 3):
        cells = [f"n={n}: {exact(readings, targets, right & (steps == n) & (targets[:, j - 1] >= 0), j, j)} | "
                 f"{exact(readings, targets, ~right & (steps == n) & (targets[:, j - 1] >= 0), j, j)}"
                 for n in range(j, 7)]
        print(f"  c{j}: " + "; ".join(cells))

records, readings, targets = splits["train"]
right = np.array([r["correct"] for r in records])
steps = np.array([r["steps"] for r in records])
base = (steps >= 2) & (steps <= 4) & (targets[:, 0] >= 0)
print("\ntrain, 2-4 steps: c1 at layer 3, position 1, by the size of c1, right | wrong")
for low, high in ((0, 10), (10, 100), (100, 1000), (1000, 10000)):
    size = (targets[:, 0] >= low) & (targets[:, 0] < high)
    print(f"  [{low}, {high}): {exact(readings, targets, base & size & right, 1, 1)} | "
          f"{exact(readings, targets, base & size & ~right, 1, 1)}")
for label, group in (("right", right), ("wrong", ~right)):
    missed = np.flatnonzero(base & group & (readings[:, 0, FIRST, 1] != targets[:, 0]))
    other = [readings[i, 0, FIRST, 1] in set(targets[i, 1:6][targets[i, 1:6] >= 0].tolist()) for i in missed]
    print(f"  {label}: misread c1 equals another gold step's value on {np.mean(other):.2f} of {len(missed)}")
print("\ntrain: greedy accuracy when c1 is read exactly, against when it is not")
for n in range(3, 7):
    rows = (steps == n) & (targets[:, 0] >= 0)
    read = readings[rows, 0, FIRST, 1] == targets[rows, 0]
    print(f"  {n} steps: {right[rows][read].mean():.2f} (n={int(read.sum())}) against {right[rows][~read].mean():.2f} "
          f"(n={int((~read).sum())})")
usable = right & (targets[:, -1] >= 0)
reading, gold = readings[usable, -1, layers.index(12), -2], targets[usable, -1]
print(f"\ntrain, the answer probe at layer 12, step 6, on right questions: exact {np.mean(reading == gold):.3f}; units digit "
      f"{np.mean(reading % 10 == gold % 10):.3f}; tens digit {np.mean(reading // 10 % 10 == gold // 10 % 10):.3f}")
