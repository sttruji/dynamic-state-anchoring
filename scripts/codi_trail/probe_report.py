"""Apply the preregistered rules in codi-probe/PREREGISTRATION.md to the instrumented CODI runs.

    python scripts/codi_trail/probe_report.py results/codi_trail/probe
"""
import json, math, random, sys
from pathlib import Path

folder = Path(sys.argv[1])
BOOT = 10_000

def load(name):
    return [json.loads(line) for line in (folder / f"{name}.jsonl").read_text().splitlines()]

def sign_p(x, y):
    n, k = x + y, min(x, y)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n) if n else 1.0

def fisher(a, b, c, d):
    n1, n2, t = a + b, c + d, a + c
    log_c = lambda n, k: math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)
    prob = lambda x: math.exp(log_c(n1, x) + log_c(n2, t - x) - log_c(n1 + n2, t))
    base = prob(a)
    return sum(p for p in (prob(x) for x in range(max(0, t - n2), min(t, n1) + 1)) if p <= base * (1 + 1e-9))

def auroc(scores, labels):
    pairs = sorted(zip(scores, labels))
    rank, i = 0.0, 0
    n = len(pairs)
    positive_ranks = 0.0
    while i < n:
        j = i
        while j + 1 < n and pairs[j + 1][0] == pairs[i][0]:
            j += 1
        rank = (i + j) / 2 + 1
        positive_ranks += rank * sum(pairs[k][1] for k in range(i, j + 1))
        i = j + 1
    pos = sum(labels)
    neg = n - pos
    return (positive_ranks - pos * (pos + 1) / 2) / (pos * neg) if pos and neg else float("nan")

def features(rows):
    out = []
    for r in rows:
        six = r["plain"]["6"]
        out.append({
            "A": len({r["plain"][k]["prediction"] for k in ("3", "6", "12")}) == 1,
            "count": sum(r["plain"][k]["prediction"] == six["prediction"] for k in ("3", "12", "24")),
            "correct": six["correct"],
            "B1 sum logprob": sum(six["logprobs"]),
            "B2 min token prob": math.exp(min(six["logprobs"])),
            "B3 short question": -r["question_tokens"],
        })
    return out

BASELINES = ("B1 sum logprob", "B2 min token prob", "B3 short question")

def top_precision(rows, key, count):
    # Ties are broken by id order so the choice is deterministic; rows are already in that order.
    ranked = sorted(range(len(rows)), key=lambda i: (-rows[i][key], i))[:count]
    return set(ranked)

def report(name, rows):
    feats = features(rows)
    n = len(feats)
    agree = [i for i in range(n) if feats[i]["A"]]
    other = [i for i in range(n) if not feats[i]["A"]]
    a, b = sum(feats[i]["correct"] for i in agree), sum(feats[i]["correct"] for i in other)
    print(f"\n=== {name}: n={n}, 6-step accuracy {100 * (a + b) / n:.1f}%")
    print(f"R1 agree {a}/{len(agree)} = {100 * a / len(agree):.1f}%  otherwise {b}/{len(other)} = {100 * b / len(other):.1f}%  Fisher p={fisher(a, len(agree) - a, b, len(other) - b):.3g}")
    labels = [int(f["correct"]) for f in feats]
    aurocs = {key: auroc([f[key] for f in feats], labels) for key in BASELINES}
    print("AUROC  agreement count", round(auroc([f["count"] for f in feats], labels), 3), {k: round(v, 3) for k, v in aurocs.items()})
    rng = random.Random(7)
    for key in BASELINES:
        chosen = top_precision(feats, key, len(agree))
        base_precision = sum(feats[i]["correct"] for i in chosen) / len(chosen)
        signal_precision = a / len(agree)
        agree_set = set(agree)
        diffs = []
        for _ in range(BOOT):
            sample = [rng.randrange(n) for _ in range(n)]
            in_a = [i for i in sample if i in agree_set]
            count = len(in_a)
            ranked = sorted(sample, key=lambda i: (-feats[i][key], i))[:count]
            if count:
                diffs.append(sum(feats[i]["correct"] for i in in_a) / count - sum(feats[i]["correct"] for i in ranked) / count)
        diffs.sort()
        low, high = diffs[int(0.025 * len(diffs))], diffs[int(0.975 * len(diffs))]
        verdict = "excludes 0" if low > 0 or high < 0 else "includes 0"
        print(f"R2 {key:18} top {len(chosen)} precision {100 * base_precision:.1f}% vs agreement {100 * signal_precision:.1f}%  diff {100 * (signal_precision - base_precision):+.1f} pts  CI [{100 * low:+.1f}, {100 * high:+.1f}] {verdict}")
    return feats, aurocs

def stratified(feats, key):
    order = sorted(range(len(feats)), key=lambda i: (feats[i][key], i))
    size = len(order) / 5
    bins = [order[int(q * size):int((q + 1) * size)] for q in range(5)]
    def effect(selected):
        terms = []
        for members in selected:
            yes = [i for i in members if feats[i]["A"]]
            no = [i for i in members if not feats[i]["A"]]
            if yes and no:
                terms.append(sum(feats[i]["correct"] for i in yes) / len(yes) - sum(feats[i]["correct"] for i in no) / len(no))
        return sum(terms) / len(terms)
    for q, members in enumerate(bins, 1):
        yes = [i for i in members if feats[i]["A"]]
        no = [i for i in members if not feats[i]["A"]]
        print(f"   quintile {q}: agree {sum(feats[i]['correct'] for i in yes)}/{len(yes)}  otherwise {sum(feats[i]['correct'] for i in no)}/{len(no)}")
    rng = random.Random(7)
    draws = sorted(effect([[rng.choice(m) for _ in m] for m in bins]) for _ in range(BOOT))
    low, high = draws[int(0.025 * BOOT)], draws[int(0.975 * BOOT)]
    print(f"R3 within quintiles of {key}: mean effect {100 * effect(bins):+.1f} pts  CI [{100 * low:+.1f}, {100 * high:+.1f}] {'excludes 0' if low > 0 or high < 0 else 'includes 0'}")

valid = load("gsm8k_valid")
feats_valid, aurocs_valid = report("gsm8k_valid (discovery, does not count)", valid)
best = max(BASELINES, key=aurocs_valid.__getitem__)
print("best baseline on validation:", best)
stratified(feats_valid, best)
if (folder / "svamp.jsonl").exists():
    feats_svamp, _ = report("SVAMP (replication, counts)", load("svamp"))
    stratified(feats_svamp, best)

print("\n=== HOLD TEST, gsm8k_valid")
def right(label, depth):
    return {r["id"]: r[label][depth]["correct"] for r in valid}
def compare(title, left, reference):
    x = sum(left[i] and not reference[i] for i in left)
    y = sum(reference[i] and not left[i] for i in left)
    print(f"{title}: {sum(left.values())} vs {sum(reference.values())}  gained {x}, lost {y}, p={sign_p(x, y):.3g}")
plain6, plain12, plain24 = right("plain", "6"), right("plain", "12"), right("plain", "24")
for label in ("hold_next", "hold_fed"):
    print(label)
    h12, h24 = right(label, "12"), right(label, "24")
    compare("  H1 at 24 vs plain 24", h24, plain24)
    compare("  H2 at 24 vs plain 6 ", h24, plain6)
    compare("  H3 at 12 vs plain 6 ", h12, plain6)
    compare("     at 12 vs plain 12", h12, plain12)
    for depth in ("12", "24"):
        same = sum(r[label][depth]["prediction"] == r["plain"]["6"]["prediction"] for r in valid)
        print(f"     answers equal to 6 steps at {depth}: {same}/500 (plain: {sum(r['plain'][depth]['prediction'] == r['plain']['6']['prediction'] for r in valid)})")
