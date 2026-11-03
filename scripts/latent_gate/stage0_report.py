"""Merge the Stage 0 shards and apply results/latent_gate/stage0/PLAN.md.

    python scripts/latent_gate/stage0_report.py results/latent_gate/stage0 \
        codi=results/codi_trail/probe/gsm8k_valid.jsonl simcot=<SIM-CoT probe records>
"""
import json
import math
import sys
from pathlib import Path

folder = Path(sys.argv[1])
probes = dict(arg.split("=", 1) for arg in sys.argv[2:])
RULES = ("keep_first", "recency", "dedup", "attention", "random", "oracle")
HEURISTICS = ("recency", "dedup", "attention", "random")


def sign_p(x, y):
    n, k = x + y, min(x, y)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n) if n else 1.0


def paired(a, b):
    x = sum(p and not q for p, q in zip(a, b))
    y = sum(q and not p for p, q in zip(a, b))
    return x, y, sign_p(x, y)


for model, probe_path in probes.items():
    probe = [json.loads(line) for line in Path(probe_path).read_text().splitlines()]
    order = [r["id"] for r in probe]
    lines = {}
    shards = sorted(folder.glob(f"{model}.shard*.jsonl")) or [folder / f"{model}.jsonl"]  # merged once the shards are gone
    for path in shards:
        lines.update((json.loads(line)["id"], line) for line in path.read_text().splitlines())
    if set(lines) != set(order):
        print(f"\n{model}: {len(lines)} of {len(order)} records so far; not merged")
        continue
    (folder / f"{model}.jsonl").write_text("".join(lines[i] + "\n" for i in order))
    rows = [json.loads(lines[i]) for i in order]
    long = [r["rationale_steps"] >= 4 for r in rows]
    print(f"\n=================== {model}, n={len(rows)}, {sum(long)} with 4+ rationale steps")
    for d in ("6", "12", "24"):
        same = sum(r["plain"][d]["text"] == p["plain"][d]["text"] for r, p in zip(rows, probe))
        print(f" control: plain {d} texts match the probe: {same}/{len(rows)}")

    def hits(arm, d):
        source = {"plain": lambda r: r["plain"], "oracle_readout": lambda r: r["oracle_readout"]}.get(arm, lambda r: r["rules"][arm])
        return [source(r)[d]["correct"] for r in rows]

    six = hits("plain", "6")
    print(f"\n correct, all (4+):  plain 6 = {sum(six)} ({sum(h for h, l in zip(six, long) if l)})")
    print(f"  {'':15} {'12 steps':>22} {'24 steps':>22}   vs plain 6 at 24 | vs plain 24 at 24")
    verdicts = {}
    for arm in ("plain",) + RULES + ("oracle_readout",):
        cells = []
        for d in ("12", "24"):
            h = hits(arm, d)
            cells.append(f"{sum(h):3} ({sum(x for x, l in zip(h, long) if l):2})")
        h24 = hits(arm, "24")
        x6, y6, p6 = paired(h24, six)
        x24, y24, p24 = paired(h24, hits("plain", "24"))
        line = f"  {arm:15} {cells[0]:>22} {cells[1]:>22}   +{x6}/-{y6} p={p6:.3g} | +{x24}/-{y24} p={p24:.3g}"
        if arm != "plain":
            restores = not (y6 > x6 and p6 < 0.05) and x24 > y24 and p24 < 0.05
            verdicts[arm] = restores
            line += "  restores" if restores else ""
        print(line)
    print(f" rule 1, restores (not below plain 6, above plain 24, at 24 steps): {[a for a, v in verdicts.items() if v] or 'none'}")
    best = max(HEURISTICS, key=lambda a: sum(hits(a, "24")))
    x, y, p = paired(hits("oracle", "24"), hits(best, "24"))
    verdict = "label beats heuristics" if x > y and p < 0.05 else "label does not beat heuristics"
    print(f" rule 2, oracle vs best heuristic ({best}) at 24: +{x}/-{y} p={p:.3g} => {verdict}")
    print(f" {sum(1 for r in rows if any(l[1] for l in r['labels']))} questions have a labelled entry in the plain 24-step run")
