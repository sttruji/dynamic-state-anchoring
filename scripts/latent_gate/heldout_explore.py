"""Exploratory analyses of the held-out/scale evaluations, beyond results/latent_gate/heldout_scale/PLAN.md's rules.

    python scripts/latent_gate/heldout_explore.py <folder with <model>/<dataset>.jsonl>

- the LLaMA check-3 runs against the held-out `plain`@6 rows;
- `shift`@6 against `keep_first`@24, which are the same computation (latents 1-6, the answer at positions as after 24 steps);
- the depth loss split exactly into a position part (`shift`@6 - `plain`@6) and a content part (`plain`@24 - `shift`@6);
- the answer's attention share on latent entries on `gsm8k_valid`, averaged over layers and questions;
- the near-copy share of entries 7-24, `ft6` per depth, and what the fixed gates keep.
"""
import collections
import json
import sys
from pathlib import Path

folder = Path(sys.argv[1])
PRIMARY = ("gsm8k_test", "gsm_hard", "multiarith")
BASES = [m for m in ("simcot-gpt2", "codi-gpt2", "codi-llama1b", "simcot-llama1b", "simcot-llama3b") if (folder / m).exists()]


def load(model, datasets):
    rows = []
    for dataset in datasets:
        path = folder / model / f"{dataset}.jsonl"
        if not path.exists():
            return None
        rows += [json.loads(line) for line in path.read_text().splitlines()]
    return rows


def correct(rows, policy, depth):
    return sum(r[policy][depth]["correct"] for r in rows)


print("Exploratory, not preregistered.")
print("\n== Check 3 runs against the held-out plain@6 rows (gsm8k_test)")
for model in BASES:
    published, held = load(f"{model}-published", ("gsm8k_test",)), load(model, ("gsm8k_test",))
    if published and held:
        same = sum(a["plain"]["6"]["prediction"] == b["plain"]["6"]["prediction"] for a, b in zip(published, held))
        print(f"  {model}: {correct(published, 'plain', '6')}/{len(published)} "
              f"({100 * correct(published, 'plain', '6') / len(published):.1f}%); same prediction on {same}/{len(held)}")

print("\n== shift@6 and keep_first@24 give the same text")
for model in BASES:
    for dataset in ("gsm8k_valid",) + PRIMARY + ("svamp",):
        rows = load(model, (dataset,))
        if rows and "shift" in rows[0]:
            same = sum(r["shift"]["6"]["text"] == r["keep_first"]["24"]["text"] for r in rows)
            print(f"  {model} {dataset}: {same}/{len(rows)}")

print("\n== Depth loss, in points: total = position (shift@6 - plain@6) + content (plain@24 - shift@6)")
for model in BASES:
    for pool, datasets in (("primary", PRIMARY), ("svamp", ("svamp",)), ("valid", ("gsm8k_valid",))):
        rows = load(model, datasets)
        if rows is None:
            continue
        n, six, deep, shift = len(rows), correct(rows, "plain", "6"), correct(rows, "plain", "24"), correct(rows, "shift", "6")
        dedup = correct(rows, "dedup_soft", "24")
        print(f"  {model:15} {pool:8} total {100 * (deep - six) / n:+.1f}  position {100 * (shift - six) / n:+.1f}  "
              f"content {100 * (deep - shift) / n:+.1f}  dedup_soft@24 - shift@6 {100 * (dedup - shift) / n:+.1f}")

print("\n== Attention share of the answer's first query on latent entries (gsm8k_valid, mean over layers and questions)")
for model in BASES:
    rows = load(model, ("gsm8k_valid",))
    if rows is None or "attn" not in rows[0]["plain"]["6"]:
        continue
    cells = []
    for policy, depth in (("plain", "6"), ("shift", "6"), ("plain", "24"), ("dedup_soft", "24"), ("copy_last", "24")):
        share = lambda part: sum(sum(r[policy][depth]["attn"][part]) / len(r[policy][depth]["attn"][part])
                                 for r in rows) / len(rows)
        cells.append(f"{policy}@{depth} {share('latent_1_6'):.3f}/{share('latent_later'):.3f}")
    print(f"  {model:15} (entries 1-6 / later) " + "  ".join(cells))

print("\n== Entries 7-24 under dedup_soft@24 that are near-copies of an earlier entry (gsm8k_valid)")
for model in BASES:
    rows = load(model, ("gsm8k_valid",))
    if rows:
        later = sum(len([j for j in r["dedup_soft"]["24"].get("repeats", []) if j > 6]) for r in rows)
        print(f"  {model}: {later / (18 * len(rows)):.0%}")

print("\n== ft6 arms, plain at 6 / 12 / 24")
for base in BASES:
    for pool, datasets in (("primary", PRIMARY), ("svamp", ("svamp",))):
        rows = load(f"{base}-ft6", datasets)
        if rows:
            print(f"  {base}-ft6 {pool} (n={len(rows)}): " + " / ".join(str(correct(rows, "plain", d)) for d in ("6", "12", "24")))

print("\n== What the fixed gates keep (gsm8k_valid)")
for base in BASES:
    rows = load(f"{base}-gate12_fixed", ("gsm8k_valid",))
    if rows is None:
        continue
    for depth in ("12", "24"):
        sets = collections.Counter(tuple(r["gate"][depth]["kept"]) for r in rows)
        early = sum(sum(j <= 6 for j in r["gate"][depth]["kept"]) for r in rows) / (6 * len(rows))
        print(f"  {base}-gate12_fixed @{depth}: {early:.0%} of kept entries are 1-6; most common {sets.most_common(3)}")
