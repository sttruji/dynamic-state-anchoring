#!/usr/bin/env python3
"""Fine-tune GPT-2 on explicit chain of thought: the SFT baseline and explicit-only control.

    python scripts/train_sft.py --train overfit32 --output runs/sft-overfit32
    python scripts/evaluate.py --model gpt2-sft --arg checkpoint=runs/sft-overfit32 \\
        --name sft-overfit32 --suite overfit,dev

Defaults match the DSA pilot protocol (seed 7, one seeded shuffle repeated in
order, single-example AdamW updates at 3e-5, weight decay 0.01, clip 1.0, 200
steps), so the same command is the explicit-only control for a DSA run on the
same data. The text format is LOTUS's CoT format, which the pinned checkpoint
was trained on: question, one step per line, "### answer", EOS. Only rationale
and answer tokens receive loss.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import random
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def format_example(row: dict, tokenizer) -> tuple[list[int], list[int]]:
    """Prompt ids (question + newline, as at inference) and completion ids."""
    rationale = row["rationale"]
    # GSM8K-Aug rationales are LOTUS steps joined by spaces; LOTUS trains one step per line.
    steps = rationale.split(" ") if row["id"].startswith("gsm8k_aug/") else [rationale]
    prompt = tokenizer.encode(row["question"].strip() + "\n", add_special_tokens=False)
    completion = tokenizer.encode("\n".join(steps) + "\n### " + row["answer"], add_special_tokens=False)
    return prompt, completion + [tokenizer.eos_token_id]


def main(argv=None) -> int:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from dsa.bench import data as bench_data
    from dsa.data.pilot_prepare import DEFAULT_BACKBONE, DEFAULT_REVISION
    from dsa.devices import resolve_device

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--train", required=True, choices=bench_data.TRAIN_CORPORA)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--init", default=DEFAULT_BACKBONE, help="HF id or local dir; 'gpt2' for vanilla")
    parser.add_argument("--revision", default=None, help=f"default: pinned {DEFAULT_REVISION[:12]} for the CoT model")
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--learning-rate", type=float, default=3e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", default="auto", choices=("auto", "cuda", "mps", "cpu"))
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument("--data-dir", type=Path, default=None)
    args = parser.parse_args(argv)
    if args.output.exists() and any(args.output.iterdir()):
        parser.error(f"{args.output} is not empty; choose a new --output")
    revision = args.revision or (DEFAULT_REVISION if args.init == DEFAULT_BACKBONE else None)

    rows, data_sha = bench_data.load_train_rows(args.train, args.data_dir)
    device = resolve_device(args.device)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.init, revision=revision, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(args.init, revision=revision, torch_dtype=torch.float32,
                                                 attn_implementation="eager").to(device).train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    order = list(range(len(rows)))
    random.Random(args.seed).shuffle(order)
    sample_ids = [rows[order[step % len(order)]]["id"] for step in range(args.steps)]

    args.output.mkdir(parents=True, exist_ok=True)
    log = (args.output / "train_log.jsonl").open("w", encoding="utf-8")
    started = time.perf_counter()
    for step, sample_id in enumerate(sample_ids, start=1):
        prompt, completion = format_example(rows[order[(step - 1) % len(order)]], tokenizer)
        ids = torch.tensor([prompt + completion], device=device)
        labels = torch.tensor([[-100] * len(prompt) + completion], device=device)
        loss = model(input_ids=ids, labels=labels).loss
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm, error_if_nonfinite=True)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        log.write(json.dumps({"step": step, "id": sample_id, "loss": loss.item(), "grad_norm": norm.item()}) + "\n")
        if step % args.log_every == 0 or step == args.steps:
            print(f"step {step}/{args.steps} loss {loss.item():.4f} ({time.perf_counter() - started:.0f}s)")
    log.close()
    model.save_pretrained(args.output, safe_serialization=True)
    tokenizer.save_pretrained(args.output)
    config = {**{key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
              "revision": revision, "train_sha256": data_sha, "device": str(device),
              "sample_order_sha256": hashlib.sha256(json.dumps(sample_ids).encode()).hexdigest()}
    (args.output / "sft_config.json").write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")
    print(f"Saved {args.output}. Evaluate with: python scripts/evaluate.py --model gpt2-sft "
          f"--arg checkpoint={args.output} --name <name> --suite overfit,dev")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
