"""The gate for Stage 0 of H, answer-conditioned plan search (results/plan_search/stage0/PLAN.md).

    python scripts/plan_search/report.py <stage0 output dir>

KILL if accepted gold plans are fewer than 1.5 x accepted wrong plans, or fewer than 150. GO otherwise.
"""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("run", type=Path)
args = parser.parse_args()

RATIO, FLOOR = 1.5, 150

phase1 = json.loads((args.run / "phase1.json").read_text())
greedy = [json.loads(x) for x in (args.run / "greedy.jsonl").read_text().splitlines()]
searched = [json.loads(x) for x in (args.run / "search.jsonl").read_text().splitlines()]
status = json.loads((args.run / "status.json").read_text()) if (args.run / "status.json").exists() else {}
n = len(searched)
gold = sum(r["gold"]["accepted"] for r in searched)
wrong = sum(r["wrong"]["accepted"] for r in searched)
both = sum(r["gold"]["accepted"] and r["wrong"]["accepted"] for r in searched)

print(f"Phase 1: {phase1['processed']} training questions, greedy right on "
      f"{sum(r['correct'] for r in greedy)} ({sum(r['correct'] for r in greedy) / max(1, len(greedy)):.1%}), "
      f"{phase1['wrong']} greedy-wrong; identity checked on {phase1['identity_checked']}; {phase1['seconds']:.0f}s")
print(f"Phase 2: searched {n} greedy-wrong questions"
      + (f" ({status.get('search_seconds', 0):.0f}s{', stopped at the cap' if status.get('capped') else ''})"
         if status else ""))
print(f"  gold plans accepted {gold} ({gold / max(1, n):.1%}); wrong plans accepted {wrong} ({wrong / max(1, n):.1%}); "
      f"both {both}, gold only {gold - both}, wrong only {wrong - both}")
if n:
    for name in ("gold", "wrong"):
        ends = sum(r[name]["at_boundary"] for r in searched)
        ratio = statistics.median(r[name]["distance"] / r["radius"] for r in searched if r["radius"] > 0)
        gain = statistics.median(r[name]["final_logp"] - r[f"base_logp_{name}"] for r in searched)
        print(f"  {name}: median distance {ratio:.2f} of the radius, {ends} ended on the boundary, median log-prob "
              f"gain {gain:+.2f}")
    print(f"  base log-prob, median: gold {statistics.median(r['base_logp_gold'] for r in searched):.2f}, wrong "
          f"{statistics.median(r['base_logp_wrong'] for r in searched):.2f}")

if gold < FLOOR or gold < RATIO * wrong:
    reason = f"fewer than {FLOOR} gold plans" if gold < FLOOR else f"gold under {RATIO}x wrong"
    print(f"\nH Stage 0: KILL ({reason}): the search writes answers into z1 rather than finding plans, or finds too few")
else:
    print(f"\nH Stage 0: GO: gold plans are specific ({gold} against {wrong}); Stage 1, distilling them, can be built")
