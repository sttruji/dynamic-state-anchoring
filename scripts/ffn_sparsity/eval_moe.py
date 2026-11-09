"""D1-R's evaluation (results/ffn_sparsity/d1r/PLAN.md): greedy on the validation questions under one arm's mask, run as
dsa/bench/models/codi.py runs it. With --reference, the predictions must match it on every question, or nothing is
written. The moe and hash arms also log how often their experts are the oracle's.

Writes --output (one record per question) and a summary beside it (<output stem>.json).

    python scripts/ffn_sparsity/eval_moe.py --checkpoint <dir> --arm moe --conversion <conversion.pt> \
        --output <file.jsonl> --device cuda
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", required=True, help="a directory holding pytorch_model.bin")
parser.add_argument("--arm", required=True, choices=("dense", "moe", "hash", "prune"))
parser.add_argument("--conversion", type=Path, help="conversion.pt from convert.py, for every arm but dense")
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--valid-file", type=Path, default=None, help="a JSON list in place of the locked gsm8k_valid")
parser.add_argument("--reference", type=Path, default=None,
                    help="predictions by id to match: step selection's format (greedy.prediction) or an eval file's")
parser.add_argument("--limit", type=int, help="only the first this many questions")
parser.add_argument("--depth", type=int, default=6)
parser.add_argument("--device", default="cuda")
args = parser.parse_args()
assert (args.arm == "dense") != (args.conversion is not None), "every arm but dense needs --conversion"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch
from dsa.bench import data as bench_data
from dsa.bench.models.codi import CODIGPT2
from dsa.bench.scoring import score
from moe import MoEMasks, step_key
from sparsity import phase_tags

codi = CODIGPT2(checkpoint=args.checkpoint, device=args.device)
model, projection, tokenizer, device = codi.model, codi.projection, codi.tokenizer, codi.device
model.eval()
projection.eval()
embed = model.get_input_embeddings()
masks = MoEMasks(model, torch.load(args.conversion, map_location="cpu", weights_only=True) if args.conversion else None)
vocab = codi.vocab_size


@torch.no_grad()
def greedy(ids):
    """Greedy text and the positions fed per phase (Q, T, A), as dsa/bench/models/codi.py runs it."""
    masks.phase = phase_tags(len(ids) - 1, 1, device=device)
    masks.keys = torch.tensor([ids[:-1] + [step_key(vocab, 0)]], device=device)
    out = model(input_ids=torch.tensor([ids], device=device), use_cache=True, output_hidden_states=True)
    cache, latent = out.past_key_values, projection(out.hidden_states[-1][:, -1:, :])
    masks.phase = phase_tags(latent=1, device=device)
    for step in range(1, args.depth + 1):
        masks.keys = torch.tensor([[step_key(vocab, step)]], device=device)
        out = model(inputs_embeds=latent, past_key_values=cache, use_cache=True, output_hidden_states=True)
        cache, latent = out.past_key_values, projection(out.hidden_states[-1][:, -1:, :])
    masks.phase = phase_tags(answer=1, device=device)
    token, tokens = codi.eot_id, []
    for _ in range(codi.max_new_tokens):
        masks.keys = torch.tensor([[token]], device=device)
        out = model(inputs_embeds=embed(torch.tensor([[token]], device=device)), past_key_values=cache, use_cache=True)
        cache = out.past_key_values
        token = int(out.logits[0, -1, :codi.vocab_size - 1].argmax())
        tokens.append(token)
        if token == tokenizer.eos_token_id:
            break
    text = tokenizer.decode([t for t in tokens if t < len(tokenizer)], skip_special_tokens=True)
    return text, [len(ids) - 1, 1 + args.depth, len(tokens)]


if args.valid_file:
    rows = json.loads(args.valid_file.read_text())
else:
    rows, _ = bench_data.load_eval_rows("gsm8k_valid", Path(os.environ.get("DSA_DATA_DIR", ROOT / "data")))
rows = rows[:args.limit] if args.limit else rows
masks.arm = None if args.arm == "dense" else args.arm
masks.audit = {} if args.arm in ("moe", "hash") else None
began, records = time.time(), []
for row in rows:
    ids = tokenizer.encode(row["question"].strip().replace("  ", " ")) + [codi.bot_id]
    text, positions = greedy(ids)
    scored = score(text, str(row["answer"]), "numeric")
    records.append({"id": row["id"], "rationale_steps": len(re.findall(r"<<", row.get("rationale", ""))),
                    "text": text, "prediction": scored["prediction"], "correct": bool(scored["correct"]),
                    "positions": positions})
summary = {"arm": args.arm, "checkpoint": str(args.checkpoint), "questions": len(records),
           "right": sum(r["correct"] for r in records), "seconds": time.time() - began, "matched": None}
if args.reference:
    reference = {}
    for line in args.reference.read_text().splitlines():
        entry = json.loads(line)
        reference[entry["id"]] = entry["greedy"]["prediction"] if "greedy" in entry else entry["prediction"]
    summary["matched"] = sum(reference.get(r["id"]) == r["prediction"] for r in records)
    summary["reference"] = str(args.reference)
    if summary["matched"] != len(records):
        raise SystemExit(f"identity check failed: {summary['matched']} of {len(records)} predictions match "
                         f"{args.reference}")
if masks.audit is not None:
    summary["oracle_agreement"] = {f"{layer}/{phase}": total / max(1, n) for (layer, phase), (total, n) in
                                   sorted(masks.audit.items())}
args.output.parent.mkdir(parents=True, exist_ok=True)
args.output.with_suffix(".json").write_text(json.dumps(summary, indent=1) + "\n")
args.output.write_text("".join(json.dumps(r) + "\n" for r in records))  # written last: marks the arm done
print(f"{args.arm} ({args.checkpoint}): {summary['right']} of {len(records)} in {summary['seconds']:.0f}s"
      + (f"; matched {summary['matched']}" if summary["matched"] is not None else ""), flush=True)
