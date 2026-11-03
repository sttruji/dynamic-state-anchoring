"""Exploratory: do individual latents decode (last hidden state through the LM head) to the gold intermediate results?

    python decode_latents.py <repo> <probe.jsonl> [checkpoint_dir]   # needs <probe>.latents.pt
"""
import json, re, sys
from collections import Counter
from pathlib import Path

repo, probe = Path(sys.argv[1]), Path(sys.argv[2])
checkpoint = sys.argv[3] if len(sys.argv) > 3 else None
sys.path.insert(0, str(repo))
import torch
from dsa.bench import data as bench_data
from dsa.bench.models.codi import CODIGPT2

codi = CODIGPT2(checkpoint=checkpoint, device="cpu")
tok = codi.tokenizer
rows = {r["id"]: r for r in bench_data.load_eval_rows("gsm8k_valid", repo / "data")[0]}
records = [json.loads(line) for line in probe.read_text().splitlines()]
saved = torch.load(probe.with_suffix(".latents.pt"), map_location="cpu", weights_only=True)
h = saved["h"]  # [n, 25, 768], after ln_f
with torch.no_grad():
    top = (h @ codi.model.lm_head.weight.T.float()).topk(5, -1).indices  # [n, 25, 5]


def clean(token_id):
    return tok.decode([token_id]).strip()


def number(text):
    text = text.replace(",", "")
    try:
        return float(text)
    except ValueError:
        return None


steps_total, depth = 25, Counter()
hit1 = [0] * steps_total
hit5 = [0] * steps_total
final1 = [0] * steps_total
which = [Counter() for _ in range(steps_total)]
examples = Counter()
for i, rid in enumerate(saved["ids"]):
    results = [number(m) for m in re.findall(r"=\s*([-\d.,]+)\s*>>", rows[rid]["rationale"])]
    results = [x for x in results if x is not None]
    final = number(rows[rid]["answer"])
    for t in range(steps_total):
        decoded = [number(clean(int(k))) for k in top[i, t]]
        if decoded[0] is not None and decoded[0] in results:
            hit1[t] += 1
            which[t][results.index(decoded[0]) + 1] += 1
        if any(d is not None and d in results for d in decoded):
            hit5[t] += 1
        if decoded[0] is not None and decoded[0] == final:
            final1[t] += 1
        if t <= 8:
            examples[(t, clean(int(top[i, t, 0])))] += 1
n = len(saved["ids"])
print(f"n={n}; share of questions whose latent at step t decodes (top-1 / top-5) to one of its gold intermediate results")
for t in range(steps_total):
    print(f" t={t:2}: top1 {hit1[t] / n:.3f}  top5 {hit5[t] / n:.3f}  final answer top1 {final1[t] / n:.3f}  matched step index {dict(sorted(which[t].items()))}")
print("most common top-1 tokens, t=0..8:")
for t in range(9):
    common = sorted(((c, s) for (tt, s), c in examples.items() if tt == t), reverse=True)[:6]
    print(f" t={t}: " + ", ".join(f"{s!r} {c}" for c, s in common))
