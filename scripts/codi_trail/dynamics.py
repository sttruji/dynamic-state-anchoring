"""Post hoc, no training: CODI at every depth, where drift comes from, and whether its trail is question-specific.

Plan: results/codi_trail/DYNAMICS_PLAN.md. Needs the latents written by probe.py; dynamics.sh runs it in shards.
"""
import argparse, json, sys, time
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[2])
parser.add_argument("--device", default="mps", help="the committed results ran on mps")
parser.add_argument("--dataset", required=True)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--latents", type=Path, required=True)
parser.add_argument("--content-only", action="store_true")
parser.add_argument("--limit", type=int)
parser.add_argument("--shard", type=int, default=0)
parser.add_argument("--shards", type=int, default=1)
args = parser.parse_args()
sys.path.insert(0, str(args.repo))

import torch
from dsa.bench import data as bench_data
from dsa.bench.models.codi import CODIGPT2
from dsa.bench.scoring import score

TRAINED, PLAIN, TESTED = 6, (2, 4, 5, 6, 7, 8, 9, 10, 12), (12, 24)
rows, _ = bench_data.load_eval_rows(args.dataset, args.repo / "data")
answer_type = bench_data.EVAL_DATASETS[args.dataset]["answer_type"]
saved = torch.load(args.latents, map_location="cpu", weights_only=True)
assert saved["ids"] == [row["id"] for row in rows]
mean_trail = saved["z"].mean(0)
codi = CODIGPT2(device=args.device)
model, projection, device = codi.model, codi.projection, codi.device
embed = model.get_input_embeddings()

def advance(latent, cache):
    output = model(inputs_embeds=latent, past_key_values=cache, use_cache=True, output_hidden_states=True)
    return output.past_key_values, projection(output.hidden_states[-1][:, -1:, :])

def feed(latents, cache):
    for latent in latents:
        cache, _ = advance(latent, cache)
    return cache

def readout(cache, row, position=None):
    step = embed(torch.tensor([[codi.eot_id]], device=device))
    tokens = []
    for offset in range(codi.max_new_tokens):
        where = None if position is None else torch.tensor([[position + offset]], device=device)
        output = model(inputs_embeds=step, past_key_values=cache, use_cache=True, position_ids=where)
        cache = output.past_key_values
        token = int(output.logits[0, -1, :codi.vocab_size - 1].argmax())
        tokens.append(token)
        if token == codi.tokenizer.eos_token_id:
            break
        step = embed(torch.tensor([[token]], device=device))
    text = codi.tokenizer.decode([t for t in tokens if t < len(codi.tokenizer)], skip_special_tokens=True)
    scored = score(text, row["answer"], row.get("answer_type", answer_type))
    return {"text": text, "prediction": scored["prediction"], "correct": bool(scored["correct"])}

stored = lambda trail: [trail[t].to(device)[None, None, :] for t in range(TRAINED)]
indexed = list(enumerate(rows))[:args.limit] if args.limit else list(enumerate(rows))
indexed = indexed[args.shard::args.shards]
args.output.parent.mkdir(parents=True, exist_ok=True)
start = time.time()
with torch.no_grad(), args.output.open("w") as sink:
    for number, (index, row) in enumerate(indexed, 1):
        encoded = codi.tokenizer.encode(row["question"].strip().replace("  ", " ")) + [codi.bot_id]
        output = model(input_ids=torch.tensor([encoded], device=device), use_cache=True, output_hidden_states=True)
        asked, latent = output.past_key_values, projection(output.hidden_states[-1][:, -1:, :])
        record = {"id": row["id"], "gold": row["answer"], "donor": (index + 1) % len(rows)}
        record["swap"] = readout(feed(stored(saved["z"][record["donor"]]), asked), row)
        record["mean"] = readout(feed(stored(mean_trail), asked), row)
        if not args.content_only:
            plain, reset, late = {}, {}, {}
            cache, fed = asked, []
            for step in range(1, max(TESTED) + 1):
                fed.append(latent)
                cache, latent = advance(latent, cache)
                if step in PLAIN:
                    plain[str(step)] = readout(cache, row)
                if step in TESTED:
                    reset[str(step)] = readout(cache, row, position=len(encoded) + TRAINED)
                    late[str(step)] = readout(feed(fed[-TRAINED:], asked), row)
            record.update(plain=plain, pos_reset=reset, late_six=late)
        sink.write(json.dumps(record) + "\n")
        sink.flush()
        if number % 25 == 0:
            print(f"{args.dataset} shard {args.shard}: {number}/{len(indexed)} {time.time() - start:.0f}s", flush=True)
print(f"{args.dataset} shard {args.shard} finished {len(indexed)} in {time.time() - start:.0f}s", flush=True)
