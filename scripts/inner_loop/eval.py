"""Evaluate a CODI-style checkpoint with the inner loop (idea A): Stage 0's checks, and the k sweep of trained loops.

Arms, each at the default --depth or at the depth after "@" (plain@12):
    plain       GPT-2's own forward per latent step: the reference (with the checkpoint's KK-LSE bias, if it has a kk.pt)
    nokk        plain with the KK-LSE bias switched off: what the bias itself does at test time
    loopK       the inner loop at k passes, with the checkpoint's loop.pt if it has one, else zero loop parameters, in
                which case every k must reproduce `plain` answer for answer (the identity check)
    dampKxG     training-free: the block re-applied k times with a constant gate G, no re-injection, no LoRA
    rerunR      training-free: the whole step re-run R times at the same position, each pass fed the projection of the
                last, keeping only the last pass's cache entry (so the chain advances 6R times in 6 entries)

With --decoder, SIM-CoT's step decoder reads every latent of every arm back into a `<<a op b=c>>` step (slot 0 is the
latent from <bot>, slot i the one after latent step i), scored against the gold rationale.

    python scripts/inner_loop/eval.py --checkpoint <dir> --dataset gsm8k_valid --output <file.jsonl> \
        --arms plain loop1 loop2 loop4 damp2x0.5 rerun2 --decoder
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
parser.add_argument("--checkpoint", help="a directory holding pytorch_model.bin (and loop.pt or kk.pt if trained)")
parser.add_argument("--dataset", required=True)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--arms", nargs="+", default=["plain", "loop1", "loop2", "loop4"])
parser.add_argument("--depth", type=int, default=6)
parser.add_argument("--first", type=int, default=4, help="first looped layer, for arms without a loop.pt")
parser.add_argument("--last", type=int, default=8, help="one past the last looped layer")
parser.add_argument("--kv", choices=("first", "last"), default="first")
parser.add_argument("--decoder", action="store_true", help="read latents back with SIM-CoT's step decoder")
parser.add_argument("--device", default="cuda")
parser.add_argument("--data-dir", type=Path, default=None, help="default: $DSA_DATA_DIR or <repo>/data")
parser.add_argument("--limit", type=int)
parser.add_argument("--shard", type=int, default=0)
parser.add_argument("--shards", type=int, default=1)
args = parser.parse_args()
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch
from dsa.bench import data as bench_data
from dsa.bench.models.codi import CODIGPT2
from dsa.bench.scoring import score
from kk import load_kk
from looped import (LoopParams, arithmetic, decode_steps, latent_step, load_loop, load_step_decoder, parse_step,
                    step_targets)

ARM = re.compile(r"^(plain|nokk|loop|damp|rerun)(\d*)(?:x([\d.]+))?(?:@(\d+))?$")


def parse_arm(name: str) -> dict:
    match = ARM.match(name)
    assert match, f"unknown arm {name!r}"
    kind, count, gate, depth = match.groups()
    assert kind in ("plain", "nokk") or count, f"{name}: give a pass count"
    assert (kind == "damp") == (gate is not None), f"{name}: only damp takes a gate"
    return {"kind": kind, "count": int(count or 1), "gate": float(gate or 0), "depth": int(depth or args.depth)}


arms = {name: parse_arm(name) for name in args.arms}
data_dir = args.data_dir or Path(os.environ.get("DSA_DATA_DIR", ROOT / "data"))
rows, _ = bench_data.load_eval_rows(args.dataset, data_dir)
answer_type = bench_data.EVAL_DATASETS[args.dataset]["answer_type"]
codi = CODIGPT2(checkpoint=args.checkpoint, device=args.device)
model, projection, tokenizer, device = codi.model, codi.projection, codi.tokenizer, codi.device
embed = model.get_input_embeddings()
loop_file = Path(args.checkpoint) / "loop.pt" if args.checkpoint else None
trained = load_loop(model, loop_file).eval() if loop_file and loop_file.exists() else None
zero = LoopParams(model, first=args.first, last=args.last, rank=1)  # gate 0: loopK is an identity
damped = LoopParams(model, first=args.first, last=args.last, rank=1)
kk_file = Path(args.checkpoint) / "kk.pt" if args.checkpoint else None
kk = load_kk(model, kk_file).eval() if kk_file and kk_file.exists() else None
assert kk is not None or not any(a["kind"] == "nokk" for a in arms.values()), "nokk needs a checkpoint with a kk.pt"
decoder = load_step_decoder(device) if args.decoder else None


def chain(arm, asked, first, question):
    """Latents and readout of one arm: (prediction text, latents fed in order incl. the last one made)."""
    cache, latent, latents = asked, first, [first[0, 0]]
    loop = trained if trained is not None else zero
    if arm["kind"] == "damp":
        loop = damped.fill(arm["gate"])
    if kk is not None:  # the latents sit after the question and <bot>
        kk.start, kk.count, kk.active = question + 1, arm["depth"], arm["kind"] != "nokk"
    for step in range(1, arm["depth"] + 1):
        position = torch.tensor([[question + step]], device=device)
        if arm["kind"] in ("plain", "nokk"):
            out = model(inputs_embeds=latent, past_key_values=cache, use_cache=True, output_hidden_states=True,
                        position_ids=position)
            hidden, cache = out.hidden_states[-1], out.past_key_values
        elif arm["kind"] == "rerun":
            for _ in range(arm["count"]):
                hidden, fresh = latent_step(model, None, latent, cache, position)
                latent = projection(hidden)
            cache = fresh
        else:
            hidden, cache = latent_step(model, loop, latent, cache, position, k=arm["count"], kv=args.kv)
        if arm["kind"] != "rerun":
            latent = projection(hidden)
        latents.append(latent[0, 0])
    return readout(cache, question + arm["depth"] + 1), torch.stack(latents)


def readout(cache, position):
    step, tokens = embed(torch.tensor([[codi.eot_id]], device=device)), []
    for offset in range(codi.max_new_tokens):
        out = model(inputs_embeds=step, past_key_values=cache, use_cache=True,
                    position_ids=torch.tensor([[position + offset]], device=device))
        cache = out.past_key_values
        token = int(out.logits[0, -1, :codi.vocab_size - 1].argmax())
        tokens.append(token)
        if token == tokenizer.eos_token_id:
            break
        step = embed(torch.tensor([[token]], device=device))
    return tokenizer.decode([t for t in tokens if t < len(tokenizer)], skip_special_tokens=True)


def read_steps(latents, gold):
    """Per slot: the decoded text, its parse, and how it compares with the gold step of that slot."""
    out = []
    for slot, ids in enumerate(decode_steps(decoder, embed, latents, eos=tokenizer.eos_token_id)):
        text = tokenizer.decode(ids)
        parsed = parse_step(text)
        value = arithmetic(parsed[0]) if parsed else None
        result = as_number(parsed[1]) if parsed else None
        target = gold[slot] if slot < len(gold) else None
        target_text = tokenizer.decode(target[:-1]) if target else None
        out.append({"text": text, "parsed": parsed is not None,
                    "consistent": None if value is None or result is None else abs(value - result) < 1e-6,
                    "exact": None if target_text is None else _squash(text) == _squash(target_text),
                    "gold": target_text})
    return out


def as_number(text):
    try:
        return float(text.replace(",", ""))
    except ValueError:
        return None


def _squash(text):
    return re.sub(r"\s+", "", text or "")


indexed = list(enumerate(rows))[:args.limit] if args.limit else list(enumerate(rows))
indexed = indexed[args.shard::args.shards]
args.output.parent.mkdir(parents=True, exist_ok=True)
start = time.time()
with torch.no_grad(), args.output.open("w") as sink:
    for number, (index, row) in enumerate(indexed, 1):
        encoded = tokenizer.encode(row["question"].strip().replace("  ", " ")) + [codi.bot_id]
        question = len(encoded) - 1  # <bot> sits at this cache index, latent step t at question + t
        if kk is not None:
            kk.start = None  # no latents in the question pass
        out = model(input_ids=torch.tensor([encoded], device=device), use_cache=True, output_hidden_states=True)
        asked, first = out.past_key_values, projection(out.hidden_states[-1][:, -1:, :])
        rationale = row.get("rationale", "")
        record = {"id": row["id"], "gold": row["answer"],
                  "rationale_steps": len(re.findall(r"<<", rationale)), "arms": {}}
        reference = None
        for name, arm in arms.items():
            text, latents = chain(arm, asked, first, question)
            scored = score(text, row["answer"], row.get("answer_type", answer_type))
            entry = {"text": text, "prediction": scored["prediction"], "correct": bool(scored["correct"])}
            if name == "plain":
                reference = latents
            elif reference is not None and len(latents) == len(reference):
                entry["latent_drift"] = float((latents - reference).abs().max())
            if decoder is not None:
                entry["steps"] = read_steps(latents, step_targets(tokenizer, rationale, len(latents)))
            record["arms"][name] = entry
        sink.write(json.dumps(record) + "\n")
        sink.flush()
        if number % 50 == 0:
            print(f"{args.dataset} shard {args.shard}: {number}/{len(indexed)} {time.time() - start:.0f}s", flush=True)

records = [json.loads(line) for line in args.output.read_text().splitlines()]
print(f"{args.dataset}, n={len(records)}, correct (4+ rationale steps in brackets); vs plain: same answer, wins/losses")
for name in arms:
    hits = [r["arms"][name]["correct"] for r in records]
    long = [r["arms"][name]["correct"] for r in records if r["rationale_steps"] >= 4]
    line = f"  {name:12} {sum(hits)} ({sum(long)}/{len(long)})"
    if "plain" in arms and name != "plain":
        same = sum(r["arms"][name]["prediction"] == r["arms"]["plain"]["prediction"] for r in records)
        wins = sum(r["arms"][name]["correct"] and not r["arms"]["plain"]["correct"] for r in records)
        losses = sum(r["arms"]["plain"]["correct"] and not r["arms"][name]["correct"] for r in records)
        drifts = [r["arms"][name]["latent_drift"] for r in records if "latent_drift" in r["arms"][name]]
        line += f"   same {same}/{len(records)}  +{wins}/-{losses}"
        if drifts:  # only arms at plain's depth
            line += f"  max latent drift {max(drifts):.2e}"
    print(line)
    if decoder is not None:
        slots = len(records[0]["arms"][name]["steps"]) if records else 0
        cells = []
        for s in range(slots):
            steps = [r["arms"][name]["steps"][s] for r in records]
            scored = [x["exact"] for x in steps if x["exact"] is not None]
            cells.append(f"{s}:{sum(x['parsed'] for x in steps)}p/{sum(scored)}of{len(scored)}")
        print(f"  {'':12} steps (parsed / exact of gold-scored) " + " ".join(cells))
print(f"finished in {time.time() - start:.0f}s", flush=True)
