"""Stage 0, no training: run the latent loop to 12 and 24 steps while the cache holds at most 6 latent entries.

Plan: results/latent_gate/stage0/PLAN.md. The six eviction rules run together as one batch from step 7 on.

    python scripts/latent_gate/stage0_evict.py --dataset gsm8k_valid --output results/latent_gate/stage0/codi.jsonl
    python scripts/latent_gate/stage0_evict.py --dataset gsm8k_valid --checkpoint <SIM-CoT dir> \
        --output results/latent_gate/stage0/simcot.jsonl
"""
import argparse, json, re, sys, time
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[2])
parser.add_argument("--device", default="mps")
parser.add_argument("--dataset", required=True)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--checkpoint", help="a directory holding pytorch_model.bin; the published CODI weights if omitted")
parser.add_argument("--limit", type=int)
parser.add_argument("--shard", type=int, default=0)
parser.add_argument("--shards", type=int, default=1)
args = parser.parse_args()
sys.path.insert(0, str(args.repo))

import torch
import torch.nn.functional as F
from dsa.bench import data as bench_data
from dsa.bench.models.codi import CODIGPT2
from dsa.bench.scoring import score

TRAINED, BUDGET, DEPTH, READ = 6, 6, 24, (12, 24)
RULES = ("keep_first", "recency", "dedup", "attention", "random", "oracle")
rows, _ = bench_data.load_eval_rows(args.dataset, args.repo / "data")
answer_type = bench_data.EVAL_DATASETS[args.dataset]["answer_type"]
codi = CODIGPT2(checkpoint=args.checkpoint, device=args.device)
model, projection, device = codi.model, codi.projection, codi.device
embed = model.get_input_embeddings()


def numeric(text):
    try:
        return float(text.replace(",", "").strip())
    except ValueError:
        return None


def gold_results(row):
    return [x for x in (numeric(m) for m in re.findall(r"=\s*([-\d.,]+)\s*>>", row.get("rationale", ""))) if x is not None]


def labels(logits, results):
    """Logit-lens label per row: (decoded value, whether it is one of the gold intermediate results)."""
    values = [numeric(codi.tokenizer.decode([int(i)])) for i in logits.argmax(-1)]
    return [(v, v is not None and v in results) for v in values]


def received(attentions, question, count):
    """Mean attention (layers and heads) the newest query gives each of the `count` latent entries before it: [rows, count]."""
    a = torch.stack([layer[:, :, -1, :] for layer in attentions]).mean((0, 2)).float().cpu()
    return a[:, question + 1:question + 1 + count]


def gather(cache, keep):
    """Keep cache positions `keep` ([rows, length]) in every layer."""
    index = keep.to(device)[:, None, :, None]
    return tuple((k.gather(2, index.expand(-1, k.shape[1], -1, k.shape[3])),
                  v.gather(2, index.expand(-1, v.shape[1], -1, v.shape[3]))) for k, v in cache)


def readout(cache, row, position=None):
    width = cache[0][0].shape[0]
    step = embed(torch.full((width, 1), codi.eot_id, device=device))
    tokens, done = [[] for _ in range(width)], [False] * width
    for offset in range(codi.max_new_tokens):
        where = None if position is None else torch.full((width, 1), position + offset, device=device)
        output = model(inputs_embeds=step, past_key_values=cache, use_cache=True, position_ids=where)
        cache = output.past_key_values
        chosen = output.logits[:, -1, :codi.vocab_size - 1].argmax(-1)
        for r in range(width):
            if not done[r]:
                tokens[r].append(int(chosen[r]))
                done[r] = int(chosen[r]) == codi.tokenizer.eos_token_id
        if all(done):
            break
        step = embed(chosen[:, None])
    out = []
    for t in tokens:
        text = codi.tokenizer.decode([i for i in t if i < len(codi.tokenizer)], skip_special_tokens=True)
        scored = score(text, row["answer"], row.get("answer_type", answer_type))
        out.append({"text": text, "prediction": scored["prediction"], "correct": bool(scored["correct"])})
    return out


def victim(rule, entries, fed, label, seen, generator):
    """The entry (a step number) to evict from `entries`, oldest first, newest last."""
    if rule == "keep_first":
        return entries[-1]
    if rule == "recency":
        return entries[0]
    if rule == "random":
        return entries[int(torch.randint(len(entries), (1,), generator=generator))]
    if rule == "dedup":
        vectors = F.normalize(torch.stack([fed[e] for e in entries]), dim=-1)
        similar = vectors @ vectors.T
        similar.fill_diagonal_(-2)
        i, j = divmod(int(similar.argmax()), len(entries))
        return entries[min(i, j)]
    if rule == "attention":
        return min(entries[:-1], key=lambda e: seen[e][0] / seen[e][1])
    unlabelled = [e for e in entries if not label[e][1]]
    if unlabelled:
        return unlabelled[0]
    for i, e in enumerate(entries):
        if any(label[later][0] == label[e][0] for later in entries[i + 1:]):
            return e
    return entries[0]


def oracle_pick(label, depth):
    """Readout-only oracle: labelled entries with distinct values, most recent first, then the earliest others."""
    picked, values = [], set()
    for e in range(depth, 0, -1):
        if label[e][1] and label[e][0] not in values and len(picked) < BUDGET:
            picked.append(e)
            values.add(label[e][0])
    for e in range(1, depth + 1):
        if len(picked) < BUDGET and e not in picked:
            picked.append(e)
    return sorted(picked)


indexed = list(enumerate(rows))[:args.limit] if args.limit else list(enumerate(rows))
indexed = indexed[args.shard::args.shards]
args.output.parent.mkdir(parents=True, exist_ok=True)
done = set()
if args.output.exists():  # resume: keep finished questions, drop a partly written last line
    kept = []
    for line in args.output.read_text().splitlines():
        try:
            kept.append(json.loads(line))
        except json.JSONDecodeError:
            break
    args.output.write_text("".join(json.dumps(r) + "\n" for r in kept))
    done = {r["id"] for r in kept}
    print(f"shard {args.shard}: resuming after {len(done)} finished questions", flush=True)
indexed = [(i, row) for i, row in indexed if row["id"] not in done]
start = time.time()
with torch.no_grad(), args.output.open("a") as sink:
    for number, (index, row) in enumerate(indexed, 1):
        results = gold_results(row)
        encoded = codi.tokenizer.encode(row["question"].strip().replace("  ", " ")) + [codi.bot_id]
        question = len(encoded) - 1  # <bot> sits at this cache index, latent step t at question + t
        output = model(input_ids=torch.tensor([encoded], device=device), use_cache=True, output_hidden_states=True)
        cache, latent = output.past_key_values, projection(output.hidden_states[-1][:, -1:, :])
        record = {"id": row["id"], "gold": row["answer"], "rationale_steps": len(results), "plain": {}, "oracle_readout": {}}
        fed, label, seen = {}, {}, {}

        # Plain loop, unbatched as in the probe: the controls, the shared steps 1-6 and the readout-only oracle.
        for step in range(1, DEPTH + 1):
            output = model(inputs_embeds=latent, past_key_values=cache, use_cache=True, output_hidden_states=True,
                           output_attentions=step <= TRAINED)
            cache = output.past_key_values
            fed[step], label[step] = latent[0, 0], labels(output.logits[:, -1, :], results)[0]
            if step <= TRAINED:
                for e, share in zip(range(1, step), received(output.attentions, question, step - 1)[0].tolist()):
                    seen[e] = (seen.get(e, (0.0, 0))[0] + share, seen.get(e, (0.0, 0))[1] + 1)
            latent = projection(output.hidden_states[-1][:, -1:, :])
            if step == TRAINED:
                at_six, next_six = cache, latent
            if step in (TRAINED,) + READ:
                record["plain"][str(step)] = readout(cache, row)[0]
            if step in READ:
                kept = oracle_pick(label, step)
                keep = torch.tensor([list(range(question + 1)) + [question + e for e in kept]])
                record["oracle_readout"][str(step)] = readout(gather(cache, keep), row, question + step + 1)[0] | {"kept": kept}
        record["labels"] = [[label[e][0], label[e][1]] for e in range(1, DEPTH + 1)]

        # The six rules as one batch from step 7, each holding at most BUDGET latent entries.
        width = len(RULES)
        cache = tuple((k.expand(width, -1, -1, -1).contiguous(), v.expand(width, -1, -1, -1).contiguous()) for k, v in at_six)
        latent = next_six.expand(width, -1, -1)
        entries = [list(range(1, TRAINED + 1)) for _ in RULES]
        fed_r = [dict((e, fed[e]) for e in range(1, TRAINED + 1)) for _ in RULES]
        label_r = [dict((e, label[e]) for e in range(1, TRAINED + 1)) for _ in RULES]
        seen_r = [dict(seen) for _ in RULES]
        generator = torch.Generator().manual_seed(index)
        record["rules"] = {rule: {} for rule in RULES}
        for step in range(TRAINED + 1, DEPTH + 1):
            output = model(inputs_embeds=latent, past_key_values=cache, use_cache=True, output_hidden_states=True,
                           output_attentions=True, position_ids=torch.full((width, 1), question + step, device=device))
            cache = output.past_key_values
            new_labels = labels(output.logits[:, -1, :], results)
            shares = received(output.attentions, question, BUDGET)
            keep = []
            for r, rule in enumerate(RULES):
                for e, share in zip(entries[r], shares[r].tolist()):
                    total, count = seen_r[r].get(e, (0.0, 0))
                    seen_r[r][e] = (total + share, count + 1)
                current = entries[r] + [step]  # cache order, newest last
                fed_r[r][step], label_r[r][step] = latent[r, 0], new_labels[r]
                out = current.index(victim(rule, current, fed_r[r], label_r[r], seen_r[r], generator))
                entries[r] = current[:out] + current[out + 1:]
                keep.append(list(range(question + 1)) + [question + 1 + i for i in range(len(current)) if i != out])
            cache = gather(cache, torch.tensor(keep))
            latent = projection(output.hidden_states[-1][:, -1:, :])
            if step in READ:
                for r, (rule, result) in enumerate(zip(RULES, readout(cache, row, question + step + 1))):
                    record["rules"][rule][str(step)] = result | {"kept": list(entries[r])}
        record["seconds"] = round(time.time() - start, 1)
        sink.write(json.dumps(record) + "\n")
        sink.flush()
        if number % 25 == 0:
            print(f"shard {args.shard}: {number}/{len(indexed)} {time.time() - start:.0f}s", flush=True)
print(f"shard {args.shard} finished {len(indexed)} in {time.time() - start:.0f}s", flush=True)
