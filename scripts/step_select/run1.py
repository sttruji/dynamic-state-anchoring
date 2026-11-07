"""Run 1 of step selection (results/step_select/run1/PLAN.md): greedy plus dropout reruns at steps 1-3, with each
candidate's hidden state and latent saved for the probe. Evaluation only.

For each question, greedy runs the six latent steps with dropout off. For each step s in 1..--steps, --reruns dropout
reruns of step s start from greedy's state and finish with dropout off. Candidate 0 at step s is greedy's own step. Each
candidate's fp16 last-layer hidden state (after ln_f) and its projected next latent are saved.

Outputs in --output:
    records.jsonl          one line per question: greedy (with its match against --reference) and every rerun's answer
    features/c<start>.npz  hidden and latent [questions, steps, 1 + reruns, width] for 50 questions at a time
    status.json            questions done, evaluation seconds, whether the cap stopped the run

The run resumes from records.jsonl. Torch is reseeded per question (seed * 100003 + index), so a resume changes no draw.

    python scripts/step_select/run1.py --checkpoint <simcot dir> --dataset gsm8k_test --output <dir> \
        --reference results/latent_gate/heldout_scale/simcot-gpt2/gsm8k_test.jsonl --max-minutes 20
"""
from __future__ import annotations

import argparse
import contextlib
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CHUNK = 50
parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", required=True, help="a directory holding pytorch_model.bin (fetch_simcot.py)")
parser.add_argument("--dataset", required=True)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--reference", type=Path, help="heldout_scale-format records: greedy must match plain at 6 steps")
parser.add_argument("--p", type=float, default=0.2)
parser.add_argument("--seed", type=int, default=3)
parser.add_argument("--reruns", type=int, default=7)
parser.add_argument("--steps", type=int, default=3, help="rerun steps 1..steps")
parser.add_argument("--depth", type=int, default=6)
parser.add_argument("--answer-tokens", type=int, default=32, help="answer length cap for reruns")
parser.add_argument("--max-minutes", type=float, help="stop after this much evaluation time")
parser.add_argument("--limit", type=int)
parser.add_argument("--device", default="cuda")
parser.add_argument("--data-dir", type=Path, default=None, help="default: $DSA_DATA_DIR or <repo>/data")
args = parser.parse_args()
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import torch
from dsa.bench import data as bench_data
from dsa.bench.models.codi import CODIGPT2
from dsa.bench.scoring import score
from stepsel import expand, mc_dropout, readout
from looped import latent_step

data_dir = args.data_dir or Path(os.environ.get("DSA_DATA_DIR", ROOT / "data"))
rows, _ = bench_data.load_eval_rows(args.dataset, data_dir)
rows = rows[:args.limit] if args.limit else rows
answer_type = bench_data.EVAL_DATASETS[args.dataset]["answer_type"]
reference = None
if args.reference:
    reference = {r["id"]: r["plain"]["6"]["text"] for r in map(json.loads, args.reference.read_text().splitlines())}
codi = CODIGPT2(checkpoint=args.checkpoint, device=args.device)
model, projection, tokenizer, device = codi.model, codi.projection, codi.tokenizer, codi.device
model.eval()
projection.eval()
depth = args.depth


def answer(tokens, row) -> dict:
    text = tokenizer.decode([t for t in tokens if t < len(tokenizer)], skip_special_tokens=True)
    scored = score(text, row["answer"], row.get("answer_type", answer_type))
    return {"text": text, "prediction": scored["prediction"], "correct": bool(scored["correct"])}


def run_step(latent, cache, position: int, p: float | None = None):
    """One latent step for every row: the hidden state after ln_f, the projected next latent and the cache."""
    with mc_dropout((model, projection), p) if p else contextlib.nullcontext():
        hidden, cache = latent_step(model, None, latent, cache,
                                    torch.full((latent.shape[0], 1), position, device=latent.device))
        return hidden, projection(hidden), cache


args.output.mkdir(parents=True, exist_ok=True)
(args.output / "features").mkdir(exist_ok=True)
records_path = args.output / "records.jsonl"
done = len(records_path.read_text().splitlines()) if records_path.exists() else 0
print(f"{args.dataset}: {len(rows)} questions, {done} already done", flush=True)

lines, hidden_buffer, latent_buffer = [], [], []
start = time.time()
capped = False


def flush(first: int) -> None:
    """Features first, then the records, so a crash between them only repeats a chunk."""
    if not lines:
        return
    np.savez(args.output / "features" / f"c{first:05d}.npz", ids=np.array([json.loads(x)["id"] for x in lines]),
             hidden=np.stack(hidden_buffer), latent=np.stack(latent_buffer))
    with records_path.open("a") as sink:
        sink.write("".join(x + "\n" for x in lines))
    lines.clear()
    hidden_buffer.clear()
    latent_buffer.clear()


with torch.no_grad():
    chunk_start = done
    for index in range(done, len(rows)):
        if args.max_minutes is not None and time.time() - start > 60 * args.max_minutes:
            capped = True
            break
        row = rows[index]
        torch.manual_seed(args.seed * 100003 + index)
        encoded = tokenizer.encode(row["question"].strip().replace("  ", " ")) + [codi.bot_id]
        question = len(encoded) - 1  # <bot> sits at this cache index, latent step s at question + s
        out = model(input_ids=torch.tensor([encoded], device=device), use_cache=True, output_hidden_states=True)
        caches, latents, hiddens = [out.past_key_values], [projection(out.hidden_states[-1][:, -1:, :])], [None]
        for s in range(1, depth + 1):
            hidden, latent, cache = run_step(latents[-1], caches[-1], question + s)
            hiddens.append(hidden)
            latents.append(latent)
            caches.append(cache)
        greedy = answer(readout(model, codi, caches[-1], question + depth + 1, 1)[0], row)
        if reference is not None:
            greedy["matches_reference"] = reference.get(row["id"]) == greedy["text"]

        branches, step_hidden, step_latent = [], [], []
        for s in range(1, args.steps + 1):
            hidden, latent, cache = run_step(latents[s - 1].expand(args.reruns, -1, -1),
                                             expand(caches[s - 1], args.reruns), question + s, args.p)
            step_hidden.append(torch.cat([hiddens[s], hidden], 0)[:, 0, :])
            step_latent.append(torch.cat([latents[s], latent], 0)[:, 0, :])
            current, current_cache = latent, cache
            for t in range(s + 1, depth + 1):
                _, current, current_cache = run_step(current, current_cache, question + t)
            tokens = readout(model, codi, current_cache, question + depth + 1, args.reruns, args.answer_tokens)
            branches.append([{k: v for k, v in answer(t, row).items() if k != "text"} for t in tokens])

        lines.append(json.dumps({"id": row["id"], "gold": row["answer"], "greedy": greedy, "branches": branches}))
        hidden_buffer.append(torch.stack(step_hidden).half().cpu().numpy())
        latent_buffer.append(torch.stack(step_latent).half().cpu().numpy())
        if len(lines) == CHUNK:
            flush(chunk_start)
            chunk_start = index + 1
        if (index + 1) % 100 == 0:
            print(f"{args.dataset}: {index + 1}/{len(rows)} {time.time() - start:.0f}s", flush=True)
    flush(chunk_start)

records = [json.loads(x) for x in records_path.read_text().splitlines()] if records_path.exists() else []
status_path = args.output / "status.json"
previous = json.loads(status_path.read_text()) if status_path.exists() else {"seconds": 0.0, "sessions": 0}
status = {"dataset": args.dataset, "questions": len(rows), "done": len(records), "capped": capped,
          "seconds": previous["seconds"] + time.time() - start, "sessions": previous["sessions"] + 1,
          "reruns": args.reruns, "steps": args.steps, "p": args.p, "seed": args.seed,
          "identity": sum(r["greedy"].get("matches_reference", False) for r in records) if reference else None}
status_path.write_text(json.dumps(status, indent=1) + "\n")
print(f"{args.dataset}: {len(records)}/{len(rows)} done in {status['seconds']:.0f}s over {status['sessions']} "
      f"session(s); greedy {sum(r['greedy']['correct'] for r in records)}; identity {status['identity']}; "
      f"capped {capped}", flush=True)
