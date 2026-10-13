#!/usr/bin/env python3
"""Local synthetic implementation smoke test; never a benchmark result."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import copy
from dataclasses import replace
import hashlib
import json
import time

import torch

from dsa.data.milestones import annotate_record
from dsa.data.tokenize import TinyTokenizer, tokenize_record
from dsa.models.backbone import BackboneAdapter
from dsa.models.anchored_loop import AnchorConfig, AnchoredLoop
from dsa.training.trainer import seed_everything, train_steps, save_training_state, runtime_manifest
from dsa.evaluation.evaluate import predict_question, score_answer


SYNTHETIC = [
    {"question": "Start with 2, add 3, then double. What is the result?",
     "answer": "<<2+3=5>>\n<<5*2=10>>\n#### 10"},
    {"question": "Start with 3, add 4, then double. What is the result?",
     "answer": "<<3+4=7>>\n<<7*2=14>>\n#### 14"},
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--loops", type=int, default=4)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output", type=Path, default=Path("artifacts/tiny-smoke"))
    args = parser.parse_args()
    if args.steps < 1 or args.loops < 1:
        parser.error("steps and loops must be positive")
    torch.set_num_threads(1)
    seed_everything(args.seed)
    tokenizer = TinyTokenizer()
    records = [tokenize_record(annotate_record(row), tokenizer, args.loops) for row in SYNTHETIC]
    prefix = tokenizer.encode("\n### ")
    initial = AnchoredLoop(BackboneAdapter.tiny(len(tokenizer)), AnchorConfig(
        loop_budget=args.loops, marker_token_id=tokenizer.marker_token_id))
    initial_state = copy.deepcopy(initial.state_dict())
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    report = {"purpose": "synthetic correctness smoke; not generalization evidence",
              "seed": args.seed, "training_steps_per_arm": args.steps,
              "fixed_loop_budget": args.loops, "workspace_size": 4,
              "same_initialization": True, "backbone": "random tiny GPT-2, 2 layers, width 32",
              "model_revision": "random initialization; no downloaded checkpoint",
              "dataset_revision": "inline synthetic records, v1",
              "dataset_sha256": hashlib.sha256(json.dumps(SYNTHETIC, sort_keys=True).encode()).hexdigest(),
              "runtime": runtime_manifest(), "arms": {}}
    for mode in ("static", "learned"):
        seed_everything(args.seed)
        model = copy.deepcopy(initial)
        model.config = replace(model.config, write_mode=mode)
        model.load_state_dict(initial_state)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.003, weight_decay=0.0)
        started = time.perf_counter()
        metrics = train_steps(model, records, prefix, optimizer, args.steps)
        training_seconds = time.perf_counter() - started
        model.eval()
        sample = records[0]
        question_ids = torch.tensor([sample["question_input_ids"]])
        inference_started = time.perf_counter()
        generated = model.generate(question_ids, torch.ones_like(question_ids),
                                   answer_prefix_ids=prefix, max_new_tokens=8)
        inference_seconds = time.perf_counter() - inference_started
        checkpoint = output / mode
        model.save_pretrained(checkpoint, tokenizer)
        save_training_state(checkpoint / "training_state.pt", model, optimizer, args.steps, records=records)
        reloaded = AnchoredLoop.from_pretrained(checkpoint).eval()
        check = reloaded.generate(question_ids, torch.ones_like(question_ids),
                                  answer_prefix_ids=prefix, max_new_tokens=8)
        if not torch.equal(generated.token_ids, check.token_ids):
            raise RuntimeError("Checkpoint reload changed greedy output")
        predictions = []
        for row in SYNTHETIC:
            prediction = predict_question(model, tokenizer, row["question"], max_new_tokens=8)
            gold = row["answer"].split("####")[1].strip()
            predictions.append({"question": row["question"], **prediction,
                                "gold": gold, "correct": score_answer(prediction["prediction"], gold)})
        arm = {"parameters": sum(p.numel() for p in model.parameters()),
               "first_loss": metrics[0]["total"], "last_loss": metrics[-1]["total"],
               "training_seconds": training_seconds, "inference_seconds": inference_seconds,
               "controller_supervised_records": sum(r["controller_supervised"] for r in records),
               "question_only_generation": tokenizer.decode(generated.token_ids[0].tolist(), skip_special_tokens=True),
               "gate_probabilities": generated.loop.gate_logits.sigmoid()[0].tolist(),
               "writes": generated.loop.writes[0].tolist(), "checkpoint_reload_identical": True}
        arm["training_set_predictions"] = predictions
        report["arms"][mode] = arm
        (checkpoint / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
        print(json.dumps({"arm": mode, **arm}), flush=True)
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"Report: {output / 'report.json'}")


if __name__ == "__main__":
    main()
