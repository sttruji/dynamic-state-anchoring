"""Chance baseline for decode_latents.py (match against the next question's results) and accuracy by rationale length."""
import json, re, sys
from pathlib import Path
repo, probe = Path(sys.argv[1]), Path(sys.argv[2])
checkpoint = sys.argv[3] if len(sys.argv) > 3 else None
sys.path.insert(0, str(repo))
import torch
from dsa.bench import data as bench_data
from dsa.bench.models.codi import CODIGPT2
codi = CODIGPT2(checkpoint=checkpoint, device="cpu")
rows = {r["id"]: r for r in bench_data.load_eval_rows("gsm8k_valid", repo / "data")[0]}
saved = torch.load(probe.with_suffix(".latents.pt"), map_location="cpu", weights_only=True)
records = {json.loads(l)["id"]: json.loads(l) for l in probe.read_text().splitlines()}
with torch.no_grad():
    top = (saved["h"] @ codi.model.lm_head.weight.T.float()).argmax(-1)
def num(s):
    try: return float(s.replace(",", "").strip())
    except ValueError: return None
res = [[x for x in (num(m) for m in re.findall(r"=\s*([-\d.,]+)\s*>>", rows[i]["rationale"])) if x is not None] for i in saved["ids"]]
n = len(res)
for t in (0, 2, 4, 6, 8, 12, 24):
    d = [num(codi.tokenizer.decode([int(top[i, t])])) for i in range(n)]
    own = sum(d[i] is not None and d[i] in res[i] for i in range(n)) / n
    other = sum(d[i] is not None and d[i] in res[(i + 1) % n] for i in range(n)) / n
    print(f" t={t:2}: own results {own:.3f}  next question's results {other:.3f}")
print(" accuracy by number of rationale steps (6 / 12 / 24 steps):")
by = {}
for i, rid in enumerate(saved["ids"]):
    k = min(len(res[i]), 5)
    by.setdefault(k, []).append(records[rid]["plain"])
for k in sorted(by):
    p = by[k]
    print(f"  {k}{'+' if k == 5 else ''} steps, n={len(p)}: " + " / ".join(str(sum(x[d]['correct'] for x in p)) for d in ("6", "12", "24")))
