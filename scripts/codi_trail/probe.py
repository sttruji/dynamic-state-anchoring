"""One CODI pass per question: readouts at several depths, token confidences, saved latents and hold variants.

    python scripts/codi_trail/probe.py --dataset gsm8k_valid --output results/codi_trail/probe/gsm8k_valid.jsonl --hold
    python scripts/codi_trail/probe.py --dataset svamp --output results/codi_trail/probe/svamp.jsonl

Latents go to <output>.latents.pt (about 77 MB per 500 questions, not committed).
"""
import argparse, json, sys, time
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[2])
parser.add_argument("--device", default="mps", help="the committed results ran on mps")
parser.add_argument("--dataset", required=True)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--hold", action="store_true")
parser.add_argument("--limit", type=int)
args = parser.parse_args()
sys.path.insert(0, str(args.repo))

import torch
from dsa.bench import data as bench_data
from dsa.bench.models.codi import CODIGPT2
from dsa.bench.scoring import score

DEPTH, READOUTS, HOLD_FROM, HOLD_READOUTS = 24, (3, 6, 12, 24), 6, (12, 24)
rows, _ = bench_data.load_eval_rows(args.dataset, args.repo / "data")
rows = rows[:args.limit] if args.limit else rows
answer_type = bench_data.EVAL_DATASETS[args.dataset]["answer_type"]
codi = CODIGPT2(device=args.device)
model, projection, device = codi.model, codi.projection, codi.device
embed = model.get_input_embeddings()

def advance(latent, cache):
    output = model(inputs_embeds=latent, past_key_values=cache, use_cache=True, output_hidden_states=True)
    assert isinstance(output.past_key_values, tuple), "branching relies on an immutable cache"
    hidden = output.hidden_states[-1][:, -1:, :]
    return output.past_key_values, hidden, projection(hidden)

def readout(cache, gold, row):
    step = embed(torch.tensor([[codi.eot_id]], device=device))
    tokens, logprobs = [], []
    for _ in range(codi.max_new_tokens):
        output = model(inputs_embeds=step, past_key_values=cache, use_cache=True)
        cache = output.past_key_values
        logits = output.logits[0, -1, :codi.vocab_size - 1]
        token = int(logits.argmax())
        tokens.append(token)
        logprobs.append(round(float(torch.log_softmax(logits, -1)[token]), 6))
        if token == codi.tokenizer.eos_token_id:
            break
        step = embed(torch.tensor([[token]], device=device))
    text = codi.tokenizer.decode([t for t in tokens if t < len(codi.tokenizer)], skip_special_tokens=True)
    scored = score(text, gold, row.get("answer_type", answer_type))
    return {"text": text, "prediction": scored["prediction"], "correct": bool(scored["correct"]), "logprobs": logprobs}

args.output.parent.mkdir(parents=True, exist_ok=True)
zs, hs, ids = [], [], []
start = time.time()
with torch.no_grad(), args.output.open("w") as sink:
    for number, row in enumerate(rows, 1):
        began = time.time()
        encoded = codi.tokenizer.encode(row["question"].strip().replace("  ", " ")) + [codi.bot_id]
        output = model(input_ids=torch.tensor([encoded], device=device), use_cache=True, output_hidden_states=True)
        cache, hidden = output.past_key_values, output.hidden_states[-1][:, -1:, :]
        latent = projection(hidden)
        trail_z, trail_h = [latent], [hidden]
        plain, branch = {}, None
        for step in range(1, DEPTH + 1):
            fed = latent
            cache, hidden, latent = advance(fed, cache)
            trail_z.append(latent)
            trail_h.append(hidden)
            if step in READOUTS:
                plain[str(step)] = readout(cache, row["answer"], row)
            if step == HOLD_FROM:
                branch = (cache, fed, latent)
        record = {"id": row["id"], "gold": row["answer"], "question_tokens": len(encoded) - 1, "plain": plain}
        if args.hold:
            at_six, held_fed, held_next = branch
            for label, held in (("hold_next", held_next), ("hold_fed", held_fed)):
                cache = at_six
                record[label] = {}
                for step in range(HOLD_FROM + 1, DEPTH + 1):
                    cache, _, _ = advance(held, cache)
                    if step in HOLD_READOUTS:
                        record[label][str(step)] = readout(cache, row["answer"], row)
        record["seconds"] = round(time.time() - began, 3)
        sink.write(json.dumps(record) + "\n")
        sink.flush()
        ids.append(row["id"])
        zs.append(torch.cat(trail_z, 1).squeeze(0).float().cpu())
        hs.append(torch.cat(trail_h, 1).squeeze(0).float().cpu())
        if number % 50 == 0:
            print(f"{args.dataset} {number}/{len(rows)} {time.time() - start:.0f}s", flush=True)
torch.save({"ids": ids, "z": torch.stack(zs), "h": torch.stack(hs)}, args.output.with_suffix(".latents.pt"))
print(f"{args.dataset} finished {len(rows)} in {time.time() - start:.0f}s", flush=True)
