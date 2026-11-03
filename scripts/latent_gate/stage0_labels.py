"""Stage 0 label check, no training: does a latent whose logit lens decodes to a gold intermediate result matter causally?

For each question, run the plain loop to `--depth` steps, then remove one latent entry at a time from the cache and
teacher-force the gold answer (`<eot>The answer is: N<eos>`). An entry's effect is the drop in the answer's summed
log-probability when it is removed. The decode label is the one the Stage 1 gate is trained on (and its spread-1 version,
which also labels the entry just before a labelled one).

    python scripts/latent_gate/stage0_labels.py --dataset gsm8k_valid --output results/latent_gate/stage0/labels-codi.jsonl
"""
import argparse, json, sys, time
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[2])
parser.add_argument("--device", default="mps")
parser.add_argument("--dataset", required=True)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--checkpoint")
parser.add_argument("--depth", type=int, default=12)
parser.add_argument("--limit", type=int)
args = parser.parse_args()
sys.path.insert(0, str(args.repo))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch
import torch.nn.functional as F
from dsa.bench import data as bench_data
from dsa.bench.models.codi import CODIGPT2
from gate import gold_results, numeric

rows, _ = bench_data.load_eval_rows(args.dataset, args.repo / "data")
rows = rows[:args.limit] if args.limit else rows
codi = CODIGPT2(checkpoint=args.checkpoint, device=args.device)
model, projection, device, tokenizer = codi.model, codi.projection, codi.device, codi.tokenizer
embed = model.get_input_embeddings()


def answer_logprob(cache, answer, position):
    """Summed log-probability of the answer tokens after <eot>, teacher-forced, at natural positions from `position`."""
    ids = torch.tensor([answer], device=device)
    out = model(inputs_embeds=embed(ids), past_key_values=cache, use_cache=False,
                position_ids=position + torch.arange(len(answer), device=device)[None])
    logp = F.log_softmax(out.logits[0, :-1].float(), -1)
    return float(logp.gather(1, ids[0, 1:, None]).sum())


args.output.parent.mkdir(parents=True, exist_ok=True)
start = time.time()
with torch.no_grad(), args.output.open("w") as sink:
    for number, row in enumerate(rows, 1):
        results = gold_results(row.get("rationale", ""))
        answer = [codi.eot_id] + tokenizer.encode(f"The answer is: {str(row['answer']).strip()}") + [tokenizer.eos_token_id]
        encoded = tokenizer.encode(row["question"].strip().replace("  ", " ")) + [codi.bot_id]
        question = len(encoded) - 1
        output = model(input_ids=torch.tensor([encoded], device=device), use_cache=True, output_hidden_states=True)
        cache, latent = output.past_key_values, projection(output.hidden_states[-1][:, -1:, :])
        decoded = []
        for step in range(1, args.depth + 1):
            output = model(inputs_embeds=latent, past_key_values=cache, use_cache=True, output_hidden_states=True)
            cache = output.past_key_values
            decoded.append(numeric(tokenizer.decode([int(output.logits[0, -1].argmax())])))
            latent = projection(output.hidden_states[-1][:, -1:, :])
        position = question + args.depth + 1
        full = answer_logprob(cache, answer, position)
        effects = []
        for j in range(1, args.depth + 1):
            keep = torch.tensor([i for i in range(cache[0][0].shape[2]) if i != question + j], device=device)
            pruned = tuple((k[:, :, keep], v[:, :, keep]) for k, v in cache)
            effects.append(round(full - answer_logprob(pruned, answer, position), 5))
        label = [v is not None and v in results for v in decoded]
        sink.write(json.dumps({"id": row["id"], "rationale_steps": len(results), "full_logprob": round(full, 5),
                               "decoded": decoded, "label": label, "drop_effect": effects}) + "\n")
        sink.flush()
        if number % 100 == 0:
            print(f"{number}/{len(rows)} {time.time() - start:.0f}s", flush=True)
print(f"finished {len(rows)} in {time.time() - start:.0f}s", flush=True)
