"""Checks E2 of results/latent_gate/heldout_scale/PLAN.md on a real CODI-style LLaMA checkpoint.

    python scripts/latent_gate/check_llama.py --llama codi-llama1b                     # checks 1 and 2
    python scripts/latent_gate/check_llama.py --llama codi-llama1b --published <dir>/gsm8k_test.jsonl   # check 3

1. A zero latent bias gives exactly the logits of transformers' own padding mask.
2. k exact copies of a latent entry, each given -log k, give the single entry's logits within 1e-4 (FP32).
   Both run on the first 8 gsm8k_valid questions, batched, after 6 latent steps, at the <eot> query.
3. `plain` at 6 steps on gsm8k_test is within 3 points of the published GSM8K accuracy.
Exits non-zero on any failure.
"""
import argparse
import json
import math
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from dsa.bench.models.codi_llama import CHECKPOINTS

parser = argparse.ArgumentParser()
parser.add_argument("--llama", choices=sorted(CHECKPOINTS), required=True)
parser.add_argument("--published", type=Path, help="a heldout_eval gsm8k_test.jsonl of this model: run check 3 only")
parser.add_argument("--tolerance", type=float, default=3.0, help="check 3, in accuracy points")
parser.add_argument("--device", default="cuda")
parser.add_argument("--data-dir", type=Path, default=None)
args = parser.parse_args()

if args.published:
    rows = [json.loads(line) for line in args.published.read_text().splitlines()]
    accuracy = 100 * sum(r["plain"]["6"]["correct"] for r in rows) / len(rows)
    published = CHECKPOINTS[args.llama]["published_gsm8k"]
    print(f"check 3: plain@6 on gsm8k_test {accuracy:.1f}% (n={len(rows)}) against the published {published}%")
    if len(rows) != 1319 or abs(accuracy - published) > args.tolerance:
        raise SystemExit(f"check 3 fails: off by {accuracy - published:+.1f} points (n={len(rows)}); debug or drop the model")
    print("check 3 passes")
    raise SystemExit

import torch
from batched import Loop, ask, forward
from dsa.bench import data as bench_data
from dsa.bench.models.codi_llama import CODILlama

codi = CODILlama(checkpoint=args.llama, device=args.device)
loop = Loop.of(codi)
rows, _ = bench_data.load_eval_rows("gsm8k_valid", args.data_dir or Path(os.environ.get("DSA_DATA_DIR", ROOT / "data")))
with torch.no_grad():
    asked = ask(loop, [loop.encode(row["question"]) for row in rows[:8]], codi.device)
    batch, question = asked.mask.shape
    ones = lambda n: torch.ones(batch, n, dtype=asked.mask.dtype, device=codi.device)
    cache, latent = asked.cache, asked.first
    for step in range(1, 7):
        out = forward(loop, embeds=latent, cache=cache, mask=torch.cat([asked.mask, ones(step)], 1),
                      positions=(asked.bot + step)[:, None], hidden=True)
        cache, latent = out.past_key_values, loop.projection(out.hidden_states[-1][:, -1:, :])
    eot = codi.model.get_input_embeddings()(torch.full((batch, 1), loop.eot_id, device=codi.device))
    where = (asked.bot + 7)[:, None]

    def logits(cache, n, bias=None):
        mask = torch.cat([asked.mask, ones(n + 1)], 1)
        return forward(loop, embeds=eot, cache=cache, mask=mask, positions=where, bias=bias).logits[:, -1]

    native = codi.model(inputs_embeds=eot, past_key_values=cache, attention_mask=torch.cat([asked.mask, ones(7)], 1),
                        position_ids=where, use_cache=True).logits[:, -1]
    zero = logits(cache, 6, torch.zeros(batch, question + 7, device=codi.device))
    exact = torch.equal(zero, native)
    print(f"check 1: zero bias equals transformers' mask exactly: {exact} (max diff {float((zero - native).abs().max()):.2e})")
    k, column = 4, question + 1  # latent entry 2
    copied = tuple((torch.cat([key] + [key[:, :, column:column + 1]] * (k - 1), 2),
                    torch.cat([value] + [value[:, :, column:column + 1]] * (k - 1), 2)) for key, value in cache)
    bias = torch.zeros(batch, question + 6 + k, device=codi.device)
    bias[:, [column] + list(range(question + 6, question + 6 + k - 1))] = -math.log(k)
    again = logits(copied, 6 + k - 1, bias)
    gap = float((again - native).abs().max())
    print(f"check 2: {k} copies at -log {k} against one entry: max logit difference {gap:.2e} "
          f"(argmax agrees on {int((again.argmax(-1) == native.argmax(-1)).sum())}/{batch})")
    if not exact or gap > 1e-4:
        raise SystemExit("E2 checks 1-2 fail")
print("E2 checks 1-2 pass")
