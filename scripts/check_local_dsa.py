#!/usr/bin/env python3
"""Bounded local DSA optimizer check; saves measurements, not large checkpoints.

Use train_dsa.py for resumable research training. This check measures feasibility
on training records only and never evaluates validation or held-out test answers.
"""
import argparse
from dataclasses import replace
import gc
import json
import math
from pathlib import Path
import random
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
from dsa.devices import mps_memory_snapshot, synchronize
from dsa.training import pilot
from dsa.training.objectives import LossConfig, joint_objective
from scripts.smoke_pretrained_dsa import atomic_json


def check_local(prepared_dir, output_dir, *, smoke_report=None, device="auto", steps=4,
                recurrence_mode="flt_memory", seed=7):
    if not 1 <= steps <= 16:
        raise ValueError("Local feasibility check is bounded to 1–16 updates per policy")
    prepared_dir, output_dir = Path(prepared_dir), Path(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError("Use an empty output directory to preserve previous measurements")
    manifest = json.loads((prepared_dir / "manifest.json").read_text())
    if not 1 <= manifest["loop_budget"] <= 16:
        raise ValueError("Local feasibility check is bounded to 1–16 recurrent loops")
    config = pilot.PilotConfig(
        prepared_dir, output_dir, backbone=manifest["backbone"], revision=manifest["revision"],
        device=device, loops=manifest["loop_budget"], tiny=manifest["tiny"],
        steps=steps, seed=seed, recurrence_mode=recurrence_mode, smoke_report=smoke_report)
    dev = pilot._device(device)
    pilot._seed(seed, dev)
    torch.set_num_threads(1)
    _, manifest_sha, tokenizer, records = pilot._prepared(config)
    order = list(range(len(records)))
    random.Random(seed).shuffle(order)
    selected = [records[order[index % len(order)]] for index in range(steps)]
    runtime = pilot._runtime(dev)
    report = {
        "schema_version": "dsa_local_optimizer_check_v1", "complete": False,
        "runtime": runtime, "prepared_manifest_sha256": manifest_sha,
        "script_sha256": pilot.sha256_file(__file__),
        "backbone": config.backbone, "revision": config.revision,
        "tiny": config.tiny, "recurrence_mode": recurrence_mode,
        "loop_budget": config.loops, "steps_per_policy": steps, "seed": seed,
        "sample_ids": [row["sample_id"] for row in selected], "cases": {},
        "validation_or_test_evaluated": False,
        "interpretation": "Training-device feasibility only; no reasoning, generalization, or scaling-readiness claim.",
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    atomic_json(output_dir / "report.json", report)
    prefix = tokenizer.encode("\n### ", add_special_tokens=False)
    initial_digest = None
    started = time.perf_counter()
    try:
        for policy in ("static", "every_loop", "periodic", "learned"):
            pilot._seed(seed, dev)
            model = pilot._initial_model(replace(config, write_mode=policy), tokenizer, dev)
            digest = pilot._state_digest(model)
            if initial_digest is not None and digest != initial_digest:
                raise AssertionError("Write policies must start from identical model parameters")
            initial_digest = digest
            model.train()
            optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate,
                                         weight_decay=config.weight_decay, foreach=False)
            metrics = []
            with (output_dir / f"{policy}.jsonl").open("w") as journal:
                for index, record in enumerate(selected, start=1):
                    synchronize(dev)
                    tick = time.perf_counter()
                    optimizer.zero_grad(set_to_none=True)
                    losses = joint_objective(model, record, prefix, LossConfig())
                    if not bool(torch.isfinite(losses["total"])):
                        raise FloatingPointError(f"Nonfinite loss in {policy} step {index}")
                    losses["total"].backward()
                    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm,
                                                         error_if_nonfinite=True).item()
                    gate_norm = model.gate_head.weight.grad.norm().item()
                    optimizer.step()
                    synchronize(dev)
                    row = {
                        "step": index, "sample_id": record["sample_id"],
                        **{key: losses[key].detach().item() for key in
                           ("total", "answer", "cot", "gate", "anchor")},
                        "gradient_norm_before_clipping": norm, "gate_gradient_norm": gate_norm,
                        "writes": losses["loop"].writes.detach().sum().item(),
                        "gate_probability": losses["loop"].gate_logits.detach().sigmoid().mean().item(),
                        "seconds": time.perf_counter() - tick, **mps_memory_snapshot(dev),
                    }
                    if not all(math.isfinite(row[key]) for key in
                               ("total", "answer", "cot", "gate", "anchor", "gate_gradient_norm")):
                        raise FloatingPointError("Nonfinite local check measurements")
                    journal.write(json.dumps(row, allow_nan=False) + "\n")
                    journal.flush()
                    metrics.append(row)
                    print(json.dumps({"policy": policy, **row}), flush=True)
                    del losses
            if not all(bool(torch.isfinite(parameter).all()) for parameter in model.parameters()):
                raise FloatingPointError(f"Nonfinite parameters after {policy} optimizer updates")
            final_digest = pilot._state_digest(model)
            if final_digest == digest:
                raise AssertionError("Optimizer check did not update any model parameters")
            report["cases"][policy] = {
                "initial_state_sha256": digest, "parameter_count": sum(p.numel() for p in model.parameters()),
                "final_state_sha256": final_digest, "parameters_updated": True,
                "optimizer_updates": len(metrics), "finite": True,
                "median_update_seconds": statistics.median(row["seconds"] for row in metrics),
                "median_warm_update_seconds": statistics.median(row["seconds"] for row in metrics[1:])
                                              if len(metrics) > 1 else None,
                "median_gradient_norm_before_clipping": statistics.median(
                    row["gradient_norm_before_clipping"] for row in metrics),
                "clipped_update_fraction": sum(row["gradient_norm_before_clipping"] > config.max_grad_norm
                                               for row in metrics) / len(metrics),
                "max_observed_mps_driver_bytes": max(row.get("mps_driver_allocated_bytes", 0) for row in metrics),
                "metrics_sha256": pilot.sha256_file(output_dir / f"{policy}.jsonl"),
            }
            atomic_json(output_dir / "report.json", report)
            del optimizer, model
            gc.collect()
            if dev.type == "mps":
                torch.mps.empty_cache()
        report.update(complete=True, elapsed_seconds=time.perf_counter() - started,
                      identical_initial_parameters=True)
    except Exception as exc:
        report.update(error=f"{type(exc).__name__}: {exc}", elapsed_seconds=time.perf_counter() - started)
        atomic_json(output_dir / "report.json", report)
        raise
    atomic_json(output_dir / "report.json", report)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--smoke-report", type=Path)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--recurrence-mode", choices=("standard", "flt_memory"), default="flt_memory")
    result = check_local(**vars(parser.parse_args()))
    print(json.dumps({"complete": result["complete"], "cases": result["cases"]}, indent=2))
