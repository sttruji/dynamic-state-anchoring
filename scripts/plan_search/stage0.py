"""Stage 0 of H, answer-conditioned plan search (results/plan_search/stage0/PLAN.md). Evaluation only.

Phase 1, greedy: pool questions from gsm_train.json, in random.Random(seed) order, run greedy until --wrong are greedy-
wrong or --greedy-minutes pass. Every greedy z1 is kept for the PCA. Each greedy-wrong question also gets --samples
MC-dropout samples of z1, whose largest distance from greedy z1 is its radius r.

Phase 2, search: for each greedy-wrong question, two targets: the gold answer, and the wrong integer within --window of
gold (not greedy's) whose base log-probability is closest to gold's. Each target gets --steps Adam steps on c, with
z1 = greedy z1 + U(sigma * c) in the top --components PCA directions. The objective is the target answer's
log-probability after the model's own latent steps, and c is projected back onto the radius-r ball after every step. A
plan is accepted if a deterministic rollout from the final z1 gives exactly the target.

Outputs in --output: greedy.jsonl, z1.npy, phase1.json, search.jsonl (resumable) and status.json.

    python scripts/plan_search/stage0.py --checkpoint <simcot dir> --output <dir> --device cuda
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", required=True, help="a directory holding pytorch_model.bin (fetch_simcot.py)")
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--train-file", type=Path,
                    default=Path(os.environ.get("DSA_DATA_DIR", ROOT / "data")) / "gsm_train.json")
parser.add_argument("--wrong", type=int, default=1000, help="greedy-wrong questions to collect")
parser.add_argument("--greedy-minutes", type=float, default=8.0)
parser.add_argument("--search-minutes", type=float, default=12.0)
parser.add_argument("--steps", type=int, default=20)
parser.add_argument("--lr", type=float, default=0.1)
parser.add_argument("--batch", type=int, default=16, help="questions searched together")
parser.add_argument("--components", type=int, default=64)
parser.add_argument("--samples", type=int, default=8, help="MC-dropout samples of z1 for the radius")
parser.add_argument("--p", type=float, default=0.2)
parser.add_argument("--window", type=int, default=50, help="wrong targets lie within this distance of gold")
parser.add_argument("--seed", type=int, default=11)
parser.add_argument("--check", type=int, default=20, help="questions checked against GPT-2's own forward")
parser.add_argument("--depth", type=int, default=6)
parser.add_argument("--max-question-tokens", type=int, default=256)
parser.add_argument("--device", default="cuda")
args = parser.parse_args()
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts/step_select"))

import numpy as np
import torch
import torch.nn.functional as F
from dsa.bench.models.codi import CODIGPT2
from dsa.bench.scoring import score
from stepsel import expand, mc_dropout, readout
from looped import latent_step

codi = CODIGPT2(checkpoint=args.checkpoint, device=args.device)
model, projection, tokenizer, device = codi.model, codi.projection, codi.tokenizer, codi.device
model.eval()
projection.eval()
for parameter in list(model.parameters()) + list(projection.parameters()):
    parameter.requires_grad_(False)
embed = model.get_input_embeddings()
depth = args.depth
args.output.mkdir(parents=True, exist_ok=True)


def gold_of(answer):
    try:
        value = float(str(answer).replace(",", ""))
    except ValueError:
        return None
    return int(value) if value >= 0 and value == int(value) else None


def chain(z1, cache, question: int):
    """The model's own latent steps 1..depth from z1 [rows, 1, width], dropout off: the cache after the last step."""
    latent = z1
    for s in range(1, depth + 1):
        hidden, cache = latent_step(model, None, latent, cache,
                                    torch.full((latent.shape[0], 1), question + s, device=device))
        latent = projection(hidden)
    return cache


def answer_tokens(value) -> list[int]:
    return [codi.eot_id] + tokenizer.encode(f"The answer is: {value}") + [tokenizer.eos_token_id]


def answer_logp(cache, question: int, values):
    """log p(the answer tokens of each value) after a cache with one row per value: a tensor [len(values)]."""
    ids = [answer_tokens(v) for v in values]
    width = max(len(x) for x in ids)
    tokens = torch.tensor([x + [tokenizer.eos_token_id] * (width - len(x)) for x in ids], device=device)
    known = torch.tensor([[1] * len(x) + [0] * (width - len(x)) for x in ids], device=device)
    positions = question + depth + 1 + torch.arange(width, device=device)[None].expand(len(ids), -1)
    out = model(inputs_embeds=embed(tokens), past_key_values=cache, position_ids=positions, use_cache=False)
    labels = tokens[:, 1:].masked_fill(known[:, 1:] == 0, -100)
    loss = F.cross_entropy(out.logits[:, :-1].float().transpose(1, 2), labels, ignore_index=-100, reduction="none")
    return -loss.sum(1)


def decode(cache, question: int, rows: int, max_tokens=None) -> list[str]:
    tokens = readout(model, codi, cache, question + depth + 1, rows, max_tokens)
    return [tokenizer.decode([t for t in seq if t < len(tokenizer)], skip_special_tokens=True) for seq in tokens]


def encode(row):
    ids = tokenizer.encode(row["question"].strip().replace("  ", " ")) + [codi.bot_id]
    return ids if len(ids) <= args.max_question_tokens else None


source = json.loads(args.train_file.read_text())
order = random.Random(args.seed).sample(range(len(source)), len(source))

# Phase 1: greedy, the radii, and the z1 basis.
phase1 = args.output / "phase1.json"
if not phase1.exists():
    greedy_records, z1s, start, checked = [], [], time.time(), 0
    with torch.no_grad():
        for index in order:
            wrong_so_far = sum(not r["correct"] for r in greedy_records)
            if wrong_so_far >= args.wrong or time.time() - start > 60 * args.greedy_minutes:
                break
            row = source[index]
            gold, ids = gold_of(row["answer"]), encode(row)
            if gold is None or ids is None:
                continue
            question = len(ids) - 1
            out = model(input_ids=torch.tensor([ids], device=device), use_cache=True, output_hidden_states=True)
            cache, z1 = out.past_key_values, projection(out.hidden_states[-1][:, -1:, :])
            text = decode(chain(z1, cache, question), question, 1)[0]
            if checked < args.check:  # the same chain through GPT-2's own forward must give the same text
                latent, hf_cache = z1, cache
                for s in range(1, depth + 1):
                    step = model(inputs_embeds=latent, past_key_values=hf_cache, use_cache=True,
                                 output_hidden_states=True, position_ids=torch.tensor([[question + s]], device=device))
                    hf_cache, latent = step.past_key_values, projection(step.hidden_states[-1])
                if decode(hf_cache, question, 1)[0] != text:
                    raise SystemExit(f"identity check failed on pool index {index}: latent_step and GPT-2 differ")
                checked += 1
            scored = score(text, str(gold), "numeric")
            record = {"index": index, "gold": gold, "text": text, "prediction": scored["prediction"],
                      "correct": bool(scored["correct"])}
            if not record["correct"]:
                torch.manual_seed(args.seed * 100003 + index)
                with mc_dropout((model, projection), args.p):
                    noisy = model(input_ids=torch.tensor([ids] * args.samples, device=device),
                                  output_hidden_states=True)
                    samples = projection(noisy.hidden_states[-1][:, -1:, :])
                record["radius"] = float((samples - z1).flatten(1).norm(dim=1).max())
            greedy_records.append(record)
            z1s.append(z1[0, 0].float().cpu().numpy())
    np.save(args.output / "z1.npy", np.stack(z1s))
    (args.output / "greedy.jsonl").write_text("".join(json.dumps(r) + "\n" for r in greedy_records))
    phase1.write_text(json.dumps({"processed": len(greedy_records), "wrong": sum(not r["correct"] for r in greedy_records),
                                  "identity_checked": checked, "seconds": time.time() - start}, indent=1) + "\n")
summary = json.loads(phase1.read_text())
greedy_records = [json.loads(x) for x in (args.output / "greedy.jsonl").read_text().splitlines()]
print(f"phase 1: {summary['processed']} questions, {summary['wrong']} greedy-wrong, identity checked on "
      f"{summary['identity_checked']}, {summary['seconds']:.0f}s", flush=True)

# The basis: top principal directions of greedy z1, fixed before any search.
z1_all = np.load(args.output / "z1.npy")
centered = z1_all - z1_all.mean(0)
values, vectors = np.linalg.eigh(centered.T @ centered / len(centered))
top = np.argsort(values)[::-1][:args.components]
U = torch.tensor(vectors[:, top].copy(), dtype=torch.float32, device=device)
sigma = torch.tensor(np.sqrt(np.clip(values[top], 0, None)).copy(), dtype=torch.float32, device=device)

# Phase 2: the search, --batch questions at a time and resumable from search.jsonl. Every row's search is independent.
# The batch is left-padded as in scripts/inner_loop/train.py; base scoring and the accepting rollout run one question
# at a time, unpadded.


def padded(id_lists):
    """A left-padded question batch: its cache, the first latents, the mask and each row's step-1 position."""
    width = max(len(x) for x in id_lists)
    ids = torch.tensor([[tokenizer.eos_token_id] * (width - len(x)) + x for x in id_lists], device=device)
    mask = torch.tensor([[0] * (width - len(x)) + [1] * len(x) for x in id_lists], device=device)
    positions = (mask.cumsum(1) - 1).clamp(min=0)
    out = model(input_ids=ids, attention_mask=mask, position_ids=positions, use_cache=True, output_hidden_states=True)
    return out.past_key_values, projection(out.hidden_states[-1][:, -1:, :]), mask, positions[:, -1] + 1


def padded_logp(z1, cache, mask, first, values):
    """log p(each row's answer) after the model's own latent steps from z1, for a padded batch."""
    latent = z1
    for s in range(1, depth + 1):
        mask = torch.cat([mask, torch.ones_like(mask[:, :1])], 1)
        hidden, cache = latent_step(model, None, latent, cache, (first + s - 1)[:, None], mask=mask)
        latent = projection(hidden)
    ids = [answer_tokens(v) for v in values]
    width = max(len(x) for x in ids)
    tokens = torch.tensor([x + [tokenizer.eos_token_id] * (width - len(x)) for x in ids], device=device)
    known = torch.tensor([[1] * len(x) + [0] * (width - len(x)) for x in ids], device=device)
    out = model(inputs_embeds=embed(tokens), past_key_values=cache, attention_mask=torch.cat([mask, known], 1),
                position_ids=first[:, None] + depth + torch.arange(width, device=device)[None], use_cache=False)
    labels = tokens[:, 1:].masked_fill(known[:, 1:] == 0, -100)
    loss = F.cross_entropy(out.logits[:, :-1].float().transpose(1, 2), labels, ignore_index=-100, reduction="none")
    return -loss.sum(1)


search_path = args.output / "search.jsonl"
done = {json.loads(x)["index"] for x in search_path.read_text().splitlines()} if search_path.exists() else set()
status_path = args.output / "status.json"
spent = json.loads(status_path.read_text())["search_seconds"] if status_path.exists() else 0.0  # earlier sessions
start, capped = time.time(), False
pending = [r for r in greedy_records if not r["correct"] and r["index"] not in done]
with search_path.open("a") as sink:
    for first_row in range(0, len(pending), args.batch):
        if spent + time.time() - start > 60 * args.search_minutes:
            capped = True
            break
        group = pending[first_row:first_row + args.batch]
        id_lists = [encode(source[r["index"]]) for r in group]
        targets, bases = [], []
        with torch.no_grad():  # the wrong target, chosen per question on its unpadded greedy chain
            for record, ids in zip(group, id_lists):
                gold, question = record["gold"], len(ids) - 1
                out = model(input_ids=torch.tensor([ids], device=device), use_cache=True, output_hidden_states=True)
                cache, z1 = out.past_key_values, projection(out.hidden_states[-1][:, -1:, :])
                greedy_value = gold_of(record["prediction"]) if record["prediction"] is not None else None
                candidates = [v for v in range(max(0, gold - args.window), gold + args.window + 1)
                              if v != gold and v != greedy_value]
                base = answer_logp(expand(chain(z1, cache, question), len(candidates) + 1), question,
                                   [gold] + candidates).cpu()
                wrong = min((abs(float(base[i + 1] - base[0])), abs(v - gold), v) for i, v in enumerate(candidates))[2]
                targets += [gold, wrong]
                bases.append((float(base[0]), float(base[1 + candidates.index(wrong)])))
            cache, z1, mask, first = padded(id_lists)
            rows = torch.arange(len(group), device=device).repeat_interleave(2)  # gold and wrong for each question
            cache = tuple((k[rows], v[rows]) for k, v in cache)
            z1, mask, first = z1[rows], mask[rows], first[rows]
        radius = torch.tensor([r["radius"] for r in group], device=device).repeat_interleave(2)[:, None]
        c = torch.zeros(len(rows), args.components, device=device, requires_grad=True)
        optimizer = torch.optim.Adam([c], lr=args.lr)
        for step in range(args.steps):
            delta = (c * sigma) @ U.T
            logp = padded_logp(z1 + delta[:, None, :], cache, mask, first, targets)
            if step == 0:  # before any move, the padded batch must give the unpadded base log-probabilities
                expected = torch.tensor([x for pair in bases for x in pair], device=device)
                if float((logp.detach() - expected).abs().max()) > 1e-2:
                    raise SystemExit(f"padded and unpadded log-probabilities differ by "
                                     f"{float((logp.detach() - expected).abs().max()):.4f}; the padding is wrong")
            loss = -logp.sum()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            with torch.no_grad():  # stay within the question's own dropout spread
                norm = ((c * sigma) @ U.T).norm(dim=1, keepdim=True)
                c.mul_(torch.clamp(radius / norm.clamp(min=1e-12), max=1.0))
        with torch.no_grad():
            delta = (c * sigma) @ U.T
            final_logp = padded_logp(z1 + delta[:, None, :], cache, mask, first, targets).cpu()
            for q, (record, ids) in enumerate(zip(group, id_lists)):  # the accepting rollout: unpadded, deterministic
                question = len(ids) - 1
                out = model(input_ids=torch.tensor([ids], device=device), use_cache=True, output_hidden_states=True)
                own, own_z1 = out.past_key_values, projection(out.hidden_states[-1][:, -1:, :])
                plans = own_z1.expand(2, -1, -1) + delta[2 * q:2 * q + 2, None, :]
                texts = decode(chain(plans, expand(own, 2), question), question, 2, 32)
                outcome = {}
                for k, name in enumerate(("gold", "wrong")):
                    value = targets[2 * q + k]
                    scored = score(texts[k], str(value), "numeric")
                    distance = float(delta[2 * q + k].norm())
                    outcome[name] = {"target": value, "accepted": bool(scored["correct"]),
                                     "prediction": scored["prediction"], "final_logp": float(final_logp[2 * q + k]),
                                     "distance": distance, "at_boundary": distance >= 0.999 * record["radius"]}
                sink.write(json.dumps({"index": record["index"], "gold_answer": record["gold"],
                                       "greedy_prediction": record["prediction"], "radius": record["radius"],
                                       "base_logp_gold": bases[q][0], "base_logp_wrong": bases[q][1],
                                       **outcome}) + "\n")
        sink.flush()

searched = [json.loads(x) for x in search_path.read_text().splitlines()]
status = {"phase1": summary, "searched": len(searched), "capped": capped,
          "search_seconds": spent + time.time() - start,
          "gold_accepted": sum(r["gold"]["accepted"] for r in searched),
          "wrong_accepted": sum(r["wrong"]["accepted"] for r in searched)}
status_path.write_text(json.dumps(status, indent=1) + "\n")
print(f"phase 2: searched {len(searched)} questions; gold accepted {status['gold_accepted']}, wrong accepted "
      f"{status['wrong_accepted']}; capped {capped}", flush=True)
