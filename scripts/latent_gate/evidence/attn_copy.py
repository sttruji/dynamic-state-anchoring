"""Exploratory, no training: where CODI's readout attends past 6 steps, and whether copied cache entries alone move answers.

Plan: PLAN.md beside this script.

    python attn_copy.py --repo <repo> --dataset gsm8k_valid --output out.jsonl [--checkpoint <dir>] [--shard k --shards n]
"""
import argparse, json, sys, time
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--repo", type=Path, required=True)
parser.add_argument("--device", default="mps")
parser.add_argument("--dataset", required=True)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--checkpoint")
parser.add_argument("--limit", type=int)
parser.add_argument("--shard", type=int, default=0)
parser.add_argument("--shards", type=int, default=1)
args = parser.parse_args()
sys.path.insert(0, str(args.repo))

import torch
from dsa.bench import data as bench_data
from dsa.bench.models.codi import CODIGPT2
from dsa.bench.scoring import score

TRAINED, PLAIN = 6, (6, 12, 24)
COPIES = {"copy_last": ((TRAINED,), (6, 18)), "copy_pair": ((TRAINED - 1, TRAINED), (3, 9)), "control": ((TRAINED,), (0,))}
PARTS = ("first", "question", "bot", "latent_1_6", "latent_later", "readout")
rows, _ = bench_data.load_eval_rows(args.dataset, args.repo / "data")
answer_type = bench_data.EVAL_DATASETS[args.dataset]["answer_type"]
codi = CODIGPT2(checkpoint=args.checkpoint, device=args.device)
model, projection, device = codi.model, codi.projection, codi.device
embed = model.get_input_embeddings()


def advance(latent, cache):
    output = model(inputs_embeds=latent, past_key_values=cache, use_cache=True, output_hidden_states=True)
    assert isinstance(output.past_key_values, tuple), "branching relies on an immutable cache"
    return output.past_key_values, projection(output.hidden_states[-1][:, -1:, :])


def copied(cache, question, entries, times):
    """Append `times` repeats of latent entries (1-based steps) to every layer's keys and values."""
    index = torch.tensor([question + step for step in entries], device=device)  # step t sits at question + t
    return tuple((k if not times else torch.cat([k, k[:, :, index].repeat(1, 1, times, 1)], 2),
                  v if not times else torch.cat([v, v[:, :, index].repeat(1, 1, times, 1)], 2)) for k, v in cache)


def shares(attentions, question, latents):
    """Attention share per part for the newest query: rows are layers, columns PARTS."""
    a = torch.stack([layer[0, :, -1, :] for layer in attentions]).mean(1).float().cpu().double()  # [layers, keys]
    edges = (0, 1, question, question + 1, question + 1 + TRAINED, question + 1 + latents, a.shape[1])
    return torch.stack([a[:, lo:hi].sum(1) for lo, hi in zip(edges, edges[1:])], 1)


def readout(cache, row, question, latents, position=None):
    step = embed(torch.tensor([[codi.eot_id]], device=device))
    tokens, per_query = [], []
    for offset in range(codi.max_new_tokens):
        where = None if position is None else torch.tensor([[position + offset]], device=device)
        output = model(inputs_embeds=step, past_key_values=cache, use_cache=True, position_ids=where, output_attentions=True)
        cache = output.past_key_values
        per_query.append(shares(output.attentions, question, latents))
        token = int(output.logits[0, -1, :codi.vocab_size - 1].argmax())
        tokens.append(token)
        if token == codi.tokenizer.eos_token_id:
            break
        step = embed(torch.tensor([[token]], device=device))
    text = codi.tokenizer.decode([t for t in tokens if t < len(codi.tokenizer)], skip_special_tokens=True)
    scored = score(text, row["answer"], row.get("answer_type", answer_type))
    answer = torch.stack(per_query[1:]).mean(0) if len(per_query) > 1 else per_query[0]
    keep = lambda x: [[round(float(v), 5) for v in layer] for layer in x]
    return {"text": text, "prediction": scored["prediction"], "correct": bool(scored["correct"]),
            "attn_eot": keep(per_query[0]), "attn_answer": keep(answer)}


indexed = list(enumerate(rows))[:args.limit] if args.limit else list(enumerate(rows))
indexed = indexed[args.shard::args.shards]
args.output.parent.mkdir(parents=True, exist_ok=True)
start = time.time()
with torch.no_grad(), args.output.open("w") as sink:
    for number, (index, row) in enumerate(indexed, 1):
        encoded = codi.tokenizer.encode(row["question"].strip().replace("  ", " ")) + [codi.bot_id]
        question = len(encoded) - 1  # <bot> sits at this index, latent step t at question + t
        output = model(input_ids=torch.tensor([encoded], device=device), use_cache=True, output_hidden_states=True)
        cache, latent = output.past_key_values, projection(output.hidden_states[-1][:, -1:, :])
        record = {"id": row["id"], "gold": row["answer"], "parts": PARTS, "plain": {}}
        for step in range(1, max(PLAIN) + 1):
            cache, latent = advance(latent, cache)
            if step == TRAINED:
                at_six = cache
            if step in PLAIN:
                record["plain"][str(step)] = readout(cache, row, question, step)
        for label, (entries, repeats) in COPIES.items():
            record[label] = {}
            for times in repeats:
                latents = TRAINED + times * len(entries)
                record[label][str(latents)] = readout(copied(at_six, question, entries, times), row, question, latents,
                                                      position=len(encoded) + TRAINED)
        sink.write(json.dumps(record) + "\n")
        sink.flush()
        if number % 25 == 0:
            print(f"shard {args.shard}: {number}/{len(indexed)} {time.time() - start:.0f}s", flush=True)
print(f"shard {args.shard} finished {len(indexed)} in {time.time() - start:.0f}s", flush=True)
