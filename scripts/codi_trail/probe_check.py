"""Control for probe.py: its texts at 3, 6, 12 and 24 steps must match the harness sweep exactly.

    python scripts/codi_trail/probe_check.py results/codi_trail/probe/gsm8k_valid.jsonl results/codi_trail/sweep
"""
import json, sys
from pathlib import Path
probe, sweep = Path(sys.argv[1]), Path(sys.argv[2])
rows = [json.loads(l) for l in probe.read_text().splitlines()]
bad = 0
for n in (3, 6, 12, 24):
    lines = (sweep / "predictions" / f"codi-iter{n}" / "gsm8k_valid.jsonl").read_text().splitlines()[1:]
    want = {r["id"]: r for r in map(json.loads, lines)}
    wrong = [r["id"] for r in rows if r["plain"][str(n)]["text"] != want[r["id"]]["generated_text"]]
    score = sum(r["plain"][str(n)]["correct"] for r in rows)
    bad += len(wrong)
    print(f"steps {n}: {len(rows) - len(wrong)}/{len(rows)} texts match; correct {score}", wrong[:3])
print("mean seconds per question", round(sum(r["seconds"] for r in rows) / len(rows), 2))
print("CONTROL PASS" if not bad else f"CONTROL FAIL: {bad} mismatches")
