"""Summarize stage0_labels.py: do decode labels pick the latent entries whose removal hurts the gold answer most?

    python scripts/latent_gate/stage0_labels_report.py results/latent_gate/stage0/labels-codi.jsonl [...]
"""
import json
import sys
from pathlib import Path


def auroc(scores, labels):
    pos = [s for s, l in zip(scores, labels) if l]
    neg = [s for s, l in zip(scores, labels) if not l]
    if not pos or not neg:
        return float("nan")
    return sum((p > q) + 0.5 * (p == q) for p in pos for q in neg) / (len(pos) * len(neg))


for path in map(Path, sys.argv[1:]):
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    depth = len(rows[0]["label"])
    print(f"\n=== {path.name}: n={len(rows)}, depth {depth}")
    effect = [e for r in rows for e in r["drop_effect"]]
    for name, spread in (("label", 0), ("spread-1 label", 1)):
        label = []
        for r in rows:
            lab = list(r["label"])
            for k in range(1, spread + 1):
                lab = [a or (j + k < depth and r["label"][j + k]) for j, a in enumerate(lab)]
            label += lab
        on = [e for e, l in zip(effect, label) if l]
        off = [e for e, l in zip(effect, label) if not l]
        print(f" {name}: {len(on)} labelled entries ({len(on) / len(label):.1%}); mean drop in gold log-prob when removed: "
              f"labelled {sum(on) / max(1, len(on)):.3f}, unlabelled {sum(off) / max(1, len(off)):.3f} nats")
        print(f"   AUROC of the label for the entry's removal effect being in the top quarter: "
              f"{auroc([float(l) for l in label], [e >= sorted(effect)[int(.75 * len(effect))] for e in effect]):.3f}")
    print(" mean removal effect by entry:", " ".join(f"{j + 1}:{sum(r['drop_effect'][j] for r in rows) / len(rows):.2f}" for j in range(depth)))
    print(" labelled share by entry:     ", " ".join(f"{j + 1}:{sum(r['label'][j] for r in rows) / len(rows):.2f}" for j in range(depth)))
    early = sum(sum(r["drop_effect"][:6]) for r in rows) / len(rows)
    late = sum(sum(r["drop_effect"][6:]) for r in rows) / len(rows)
    print(f" summed removal effect, entries 1-6: {early:.2f}; entries 7-{depth}: {late:.2f} nats")
