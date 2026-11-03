"""Merge the shards and apply PLAN.md: controls, attention shares, and the copied-entry decision for each model.

    python report.py <attn-copy dir> <codi probe.jsonl> <simcot probe.jsonl>
"""
import json, math, sys
from pathlib import Path

folder = Path(sys.argv[1])
probes = {"codi": Path(sys.argv[2]), "simcot": Path(sys.argv[3])}
NAMES = {"codi": "CODI", "simcot": "SIM-CoT"}


def sign_p(x, y):
    n, k = x + y, min(x, y)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n) if n else 1.0


def auroc(scores, labels):
    pos = [s for s, l in zip(scores, labels) if l]
    neg = [s for s, l in zip(scores, labels) if not l]
    return sum((p > q) + 0.5 * (p == q) for p in pos for q in neg) / (len(pos) * len(neg))


def mean_parts(cells, key):
    """Mean over questions and layers of each part's share."""
    layers = len(cells[0][key])
    return [sum(sum(layer[j] for layer in c[key]) / layers for c in cells) / len(cells) for j in range(6)]


for model, probe_path in probes.items():
    probe = [json.loads(line) for line in probe_path.read_text().splitlines()]
    lines = {}
    for path in sorted(folder.glob(f"{model}.shard*.jsonl")) or [folder / f"{model}.jsonl"]:  # merged once shards are gone
        lines.update((json.loads(line)["id"], line) for line in path.read_text().splitlines())
    order = [r["id"] for r in probe]
    assert set(lines) == set(order), f"{model}: {len(lines)} records, expected {len(order)}"
    (folder / f"{model}.jsonl").write_text("".join(lines[i] + "\n" for i in order))
    rows = [json.loads(lines[i]) for i in order]
    n = len(rows)
    print(f"\n=================== {NAMES[model]}, n={n}")
    for d in ("6", "12", "24"):
        print(f" control: plain {d} texts match the probe: {sum(r['plain'][d]['text'] == p['plain'][d]['text'] for r, p in zip(rows, probe))}/{n}")
    print(f" control: explicit-position 6 matches plain 6: {sum(r['control']['6']['text'] == r['plain']['6']['text'] for r in rows)}/{n}")

    parts = rows[0]["parts"]
    print("\n attention share, mean over questions and layers:", " | ".join(parts))
    cells = [("plain", "6"), ("plain", "12"), ("plain", "24"), ("copy_last", "12"), ("copy_last", "24"), ("copy_pair", "12"), ("copy_pair", "24")]
    for key in ("attn_eot", "attn_answer"):
        print(f"  {key}")
        for arm, d in cells:
            m = mean_parts([r[arm][d] for r in rows], key)
            print(f"   {arm:9} {d:>2}: " + " ".join(f"{v:.3f}" for v in m))
    print("  <eot> share on latents 1-6 by layer, plain 6 -> plain 24:")
    for d in ("6", "24"):
        per_layer = [sum(r["plain"][d]["attn_eot"][layer][3] for r in rows) / n for layer in range(12)]
        print(f"   {d:>2}: " + " ".join(f"{v:.3f}" for v in per_layer))
    per_layer = [sum(r["plain"]["24"]["attn_eot"][layer][4] for r in rows) / n for layer in range(12)]
    print("  <eot> share on latents 7-24 by layer, plain 24:\n      " + " ".join(f"{v:.3f}" for v in per_layer))

    six = [r["plain"]["6"]["correct"] for r in rows]
    plain_x = sum(a and not b for a, b in zip([r["plain"]["24"]["correct"] for r in rows], six))
    plain_y = sum(b and not a for a, b in zip([r["plain"]["24"]["correct"] for r in rows], six))
    plain_net = plain_y - plain_x
    print(f"\n correct: plain 6 {sum(six)}; plain 24 {sum(r['plain']['24']['correct'] for r in rows)} (gained {plain_x}, lost {plain_y}, net loss {plain_net})")
    for arm in ("copy_last", "copy_pair"):
        for d in ("12", "24"):
            got = [r[arm][d]["correct"] for r in rows]
            x = sum(a and not b for a, b in zip(got, six))
            y = sum(b and not a for a, b in zip(got, six))
            changed = sum(r[arm][d]["prediction"] != r["plain"]["6"]["prediction"] for r in rows)
            p = sign_p(x, y)
            line = f"  {arm} {d}: {sum(got)}  vs 6: gained {x}, lost {y}, net loss {y - x}, p={p:.3g}; answer changed {changed}"
            if d == "24":
                if y > x and p < 0.05 and y - x >= 0.5 * plain_net:
                    verdict = "copies reproduce the drift"
                elif y - x < 0.25 * plain_net:
                    verdict = "copies do not cause the drift"
                else:
                    verdict = "partial"
                vs = [r["plain"]["24"]["correct"] for r in rows]
                x2 = sum(a and not b for a, b in zip(got, vs))
                y2 = sum(b and not a for a, b in zip(got, vs))
                line += f"  => {verdict} (net loss {y - x} vs plain {plain_net}); vs plain 24: copy-only {x2}, plain-only {y2}, p={sign_p(x2, y2):.3g}"
            print(line)
    changed = [r["plain"]["24"]["prediction"] != r["plain"]["6"]["prediction"] for r in rows]
    later = [sum(layer[4] for layer in r["plain"]["24"]["attn_eot"]) / 12 for r in rows]
    print(f"\n secondary: <eot> share on latents 7-24 predicting an answer change 6->24 ({sum(changed)} changed): AUROC {auroc(later, changed):.3f}")
