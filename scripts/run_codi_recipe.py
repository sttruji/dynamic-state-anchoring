#!/usr/bin/env python3
"""Train and evaluate one CODI-recipe DSA arm, resumably, for a Colab GPU session.

1. Plan: pick the largest batch whose worst case (the prepared set's longest
   records) fits in GPU memory, then fix steps, warmup and the code commit in
   ``BACKUP/plan.json``. Reruns reuse the plan, so a resumed run is the same run.
2. Train on fast local disk with ``train_dsa.py``. Print progress, and copy each
   new checkpoint to ``BACKUP/backup``. A rerun restores that backup and resumes.
3. Evaluate the final model with the harness: 8 loops, 1 loop and the explicit
   route. Print paired 8-versus-1 counts, the first check that the loop helps.

Recipe and options: docs/CODI_RECIPE.md. Notebook: notebooks/DSA_CODI_Recipe_Colab.ipynb.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from math import comb
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
# CODI's GPT-2 GSM8K-Aug settings (scripts/train_gpt2_gsm8k-aug.sh), with DSA's K=8 FLT loop.
RECIPE = {"recurrence-mode": "flt_memory", "loops": 8, "workspace": 4, "seed": 7,
          "lora-rank": 128, "lora-alpha": 32, "lora-dropout": 0.1, "learning-rate": 3e-3,
          "weight-decay": 0.1, "max-grad-norm": 2, "answer-distill-weight": 1}
RECORDS_PER_UPDATE = 128  # CODI: 64 per device x 2 accumulation
WARMUP_FRACTION = 0.03


def _flags(values: dict) -> list[str]:
    return [item for key, value in values.items() for item in (f"--{key}", str(value))]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _metrics(run: Path) -> list[dict]:
    rows = []
    path = run / "metrics.jsonl"
    for line in path.read_text(encoding="utf-8").splitlines() if path.exists() else []:
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:  # the trainer may be writing the last line
            break
    return rows


def train_command(args, prepared: Path, output: Path, **settings) -> list[str]:
    command = [sys.executable, str(ROOT / "scripts/train_dsa.py"), "--prepared-dir", str(prepared),
               "--output-dir", str(output), "--device", args.device, "--write-mode", args.write_mode,
               *_flags(RECIPE), *_flags(settings)]
    if args.new_module_learning_rate is not None:
        command += ["--new-module-learning-rate", str(args.new_module_learning_rate)]
    return command + (["--tiny"] if args.tiny else ["--smoke-report", str(args.smoke_report)])


def longest_records_set(prepared: Path, size: int, output: Path) -> Path:
    """A copy of ``prepared`` whose training split is its ``size`` longest records."""
    if (output / "manifest.json").exists():
        return output
    lines = (prepared / "train.tokenized.jsonl").read_text(encoding="utf-8").splitlines()
    def width(line):
        record = json.loads(line)
        return (len(record["explicit_input_ids"]),
                len(record["question_input_ids"]) + len(record["latent_suffix_input_ids"]))
    longest = sorted(lines, key=width, reverse=True)[:size]
    staging = output.with_name(output.name + ".tmp")
    shutil.rmtree(staging, ignore_errors=True)
    shutil.copytree(prepared / "tokenizer", staging / "tokenizer")
    manifest = json.loads((prepared / "manifest.json").read_text(encoding="utf-8"))
    for split in ("valid", "test"):
        shutil.copy2(prepared / manifest["splits"][split]["path"], staging / manifest["splits"][split]["path"])
    (staging / "train.tokenized.jsonl").write_text("\n".join(longest) + "\n", encoding="utf-8")
    manifest["splits"]["train"] = {"path": "train.tokenized.jsonl", "records": len(longest),
                                   "sha256": _sha256(staging / "train.tokenized.jsonl")}
    (staging / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    staging.rename(output)
    return output


def stress_trial(args, prepared: Path, batch: int) -> dict | None:
    """Two updates on the ``batch`` longest records; None when they run out of memory."""
    root = args.work_dir.parent / f"{args.work_dir.name}-stress"
    output = root / f"run-{batch}"
    shutil.rmtree(output, ignore_errors=True)
    command = train_command(args, longest_records_set(prepared, batch, root / f"longest-{batch}"), output,
                            **{"steps": 2, "batch-size": batch, "gradient-accumulation": 1, "save-every": 2})
    result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
    try:
        if result.returncode:
            if "out of memory" in (result.stdout + result.stderr).lower():
                return None
            raise RuntimeError(f"Calibration run failed:\n{(result.stdout + result.stderr)[-4000:]}")
        rows = _metrics(output)
        return {"peak_bytes": max(row.get("peak_cuda_allocated_bytes") or 0 for row in rows),
                "seconds": rows[-1]["seconds"]}
    finally:
        shutil.rmtree(output, ignore_errors=True)


def _device_memory(device: str) -> tuple[str, int | None]:
    if device != "cuda":
        return device, None
    import torch
    return torch.cuda.get_device_name(0), torch.cuda.get_device_properties(0).total_memory


def _commit() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()


def choose_plan(args, prepared: Path) -> dict:
    manifest = json.loads((prepared / "manifest.json").read_text(encoding="utf-8"))
    records = manifest["splits"]["train"]["records"]
    steps = records // args.records_per_update
    if steps < 1:
        raise ValueError(f"{records} prepared records are fewer than one {args.records_per_update}-record update")
    name, memory = _device_memory(args.device)
    for batch in args.batch_candidates:
        if args.records_per_update % batch:
            continue
        trial = stress_trial(args, prepared, batch)
        if trial is None:
            print(f"batch {batch}: out of memory", flush=True)
        elif memory is not None and trial["peak_bytes"] > args.memory_fraction * memory:
            print(f"batch {batch}: {trial['peak_bytes'] / 2**30:.1f} GiB peak, over "
                  f"{args.memory_fraction:.0%} of {memory / 2**30:.1f} GiB", flush=True)
        else:
            break
    else:
        raise RuntimeError(f"No batch size in {args.batch_candidates} fits on {name}")
    # The longest records are the slowest, so this bounds the time per update from above.
    update_seconds = trial["seconds"] * args.records_per_update / batch
    return {"schema_version": 1, "commit": _commit(), "gpu": name, "gpu_memory_bytes": memory,
            "prepared_manifest_sha256": _sha256(prepared / "manifest.json"), "train_records": records,
            "write_mode": args.write_mode, "recipe": RECIPE, "new_module_learning_rate": args.new_module_learning_rate,
            "batch_size": batch, "gradient_accumulation": args.records_per_update // batch, "steps": steps,
            "warmup_steps": max(1, round(WARMUP_FRACTION * steps)),
            "save_every": max(1, min(steps, round(args.backup_minutes * 60 / update_seconds))),
            "calibration": {"peak_bytes": trial["peak_bytes"], "longest_batch_seconds": trial["seconds"],
                            "upper_bound_hours": steps * update_seconds / 3600}}


def load_or_choose_plan(args, prepared: Path) -> dict:
    path, backup = args.backup_dir / "plan.json", args.backup_dir / "backup"
    plan = json.loads(path.read_text(encoding="utf-8")) if path.exists() else None
    started = (backup / "latest.json").exists() or (args.work_dir / "latest.json").exists()
    if plan is not None:
        if plan["prepared_manifest_sha256"] != _sha256(prepared / "manifest.json"):
            raise RuntimeError("Prepared data differ from this run's plan; use a new run name for new data.")
        if plan["write_mode"] != args.write_mode or plan["new_module_learning_rate"] != args.new_module_learning_rate:
            raise RuntimeError("Settings differ from this run's plan; use a new run name for new settings.")
        gpu, commit = _device_memory(args.device)[0], _commit()
        if started and plan["commit"] != commit:
            raise RuntimeError(f"This run started at commit {plan['commit']}; check it out to resume.")
        if started and plan["gpu"] != gpu:
            raise RuntimeError(f"This run started on {plan['gpu']} and can only resume there, not on {gpu}. "
                               "Reconnect to that GPU type, or start a new run name here.")
        if plan["gpu"] == gpu and plan["commit"] == commit:  # an unstarted plan is otherwise remade
            print("Reusing plan:", json.dumps({key: plan[key] for key in
                  ("gpu", "batch_size", "gradient_accumulation", "steps", "save_every")}), flush=True)
            return plan
    plan = choose_plan(args, prepared)
    args.backup_dir.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")
    print("New plan:", json.dumps({key: plan[key] for key in ("gpu", "batch_size", "gradient_accumulation", "steps",
          "warmup_steps", "save_every")}), flush=True)
    print(f"Training should take at most about {plan['calibration']['upper_bound_hours']:.1f} h "
          "(estimated from the longest records).", flush=True)
    return plan


def backup_checkpoint(run: Path, backup: Path) -> str | None:
    """Copy the run's latest complete checkpoint and journal to ``backup``, replacing the old copy."""
    if not (run / "latest.json").exists():
        return None
    latest = json.loads((run / "latest.json").read_text(encoding="utf-8"))
    if (backup / "latest.json").exists() and json.loads((backup / "latest.json").read_text())["path"] == latest["path"]:
        return latest["path"]
    staging, old = backup.with_name(backup.name + ".staging"), backup.with_name(backup.name + ".old")
    shutil.rmtree(staging, ignore_errors=True)
    try:
        shutil.copytree(run / latest["path"], staging / latest["path"])
    except (FileNotFoundError, shutil.Error):  # superseded and removed while copying; next poll retries
        shutil.rmtree(staging, ignore_errors=True)
        return None
    for name in ("metrics.jsonl", "run_manifest.json"):  # the journal may run ahead; resume trims it
        if (run / name).exists():
            shutil.copy2(run / name, staging / name)
    (staging / "latest.json").write_text(json.dumps(latest) + "\n", encoding="utf-8")
    shutil.rmtree(old, ignore_errors=True)
    if backup.exists():
        backup.rename(old)
    staging.rename(backup)
    shutil.rmtree(old, ignore_errors=True)
    return latest["path"]


def restore_checkpoint(backup: Path, run: Path) -> bool:
    if (run / "latest.json").exists() or not (backup / "latest.json").exists():
        return False
    shutil.copytree(backup, run, dirs_exist_ok=True)
    print(f"Restored {json.loads((backup / 'latest.json').read_text())['path']} from {backup}", flush=True)
    return True


def train(args, prepared: Path, plan: dict) -> None:
    run, backup = args.work_dir, args.backup_dir / "backup"
    restore_checkpoint(backup, run)
    run.mkdir(parents=True, exist_ok=True)
    command = train_command(args, prepared, run, **{
        "steps": plan["steps"], "warmup-steps": plan["warmup_steps"], "lr-schedule": "cosine",
        "batch-size": plan["batch_size"], "gradient-accumulation": plan["gradient_accumulation"],
        "save-every": plan["save_every"]})
    log_path = run.parent / f"{run.name}-train.log"
    print(f"Training {plan['steps']} updates of {plan['batch_size']} x {plan['gradient_accumulation']} records. "
          f"Log: {log_path}", flush=True)
    with log_path.open("a", encoding="utf-8") as log:
        process = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                                   env={**os.environ, "PYTHONUNBUFFERED": "1"})
        reported = time.monotonic()
        while process.poll() is None:
            time.sleep(args.poll_seconds)
            backup_checkpoint(run, backup)
            if time.monotonic() - reported >= args.report_minutes * 60:
                reported = time.monotonic()
                rows = _metrics(run)
                if rows:
                    row, recent = rows[-1], statistics.median(r["seconds"] for r in rows[-20:])
                    print(f"update {row['step']}/{plan['steps']}  loss {row['total']:.3f}  answer {row['answer']:.3f}  "
                          f"distill {row['distill']:.3f}  {recent:.1f} s/update  "
                          f"about {(plan['steps'] - row['step']) * recent / 3600:.1f} h left", flush=True)
    backup_checkpoint(run, backup)
    if process.returncode:
        tail = log_path.read_text(encoding="utf-8", errors="replace")[-4000:]
        hint = ("\nThe saved run's settings or software differ from this session (for example a new Colab "
                "PyTorch). Start a new run name to train from scratch here." if "settings differ" in tail else "")
        raise RuntimeError(f"Training failed (exit {process.returncode}). Log tail:\n{tail}{hint}")
    status = json.loads((run / "run_manifest.json").read_text(encoding="utf-8"))["status"]
    if status != "complete":
        raise RuntimeError(f"Training stopped with status {status}")


def _run_without_counters(command: list[str]) -> None:
    """Run and echo output, keeping only the last carriage-return update of each line."""
    process = subprocess.Popen(command, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                               env={**os.environ, "PYTHONUNBUFFERED": "1"})
    for line in process.stdout:
        sys.stdout.write(line.decode("utf-8", errors="replace").split("\r")[-1])
        sys.stdout.flush()
    if process.wait():
        raise subprocess.CalledProcessError(process.returncode, command)


def evaluate(args) -> dict[str, list[str]]:
    rows = {f"{args.name}-k8": [], f"{args.name}-k1": ["--arg", "loops=1"]}
    if not args.tiny:  # the explicit route needs the Hugging Face tokenizer
        rows[f"{args.name}-explicit"] = ["--arg", "route=explicit"]
    for name, extra in rows.items():
        print(f"Evaluating {name} on {args.suite}", flush=True)
        _run_without_counters([sys.executable, str(ROOT / "scripts/evaluate.py"), "--model", "dsa",
                               "--arg", f"checkpoint={args.work_dir}", "--name", name, *extra,
                               "--suite", args.suite, "--device", args.device,
                               "--results-dir", str(args.results_dir)])
    return rows


def _correct(results: Path, name: str, dataset: str) -> dict[str, bool] | None:
    path = results / "predictions" / name / f"{dataset}.jsonl"
    if not path.exists():
        return None
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    return {row["id"]: bool(row["correct"]) for row in rows if "id" in row}


def paired_counts(left: dict[str, bool], right: dict[str, bool]) -> dict:
    """Questions only ``left`` gets right, only ``right`` does, and the exact two-sided sign-test p."""
    if left.keys() != right.keys():
        raise ValueError("Paired predictions must cover the same questions")
    only_left = sum(left[key] and not right[key] for key in left)
    only_right = sum(right[key] and not left[key] for key in left)
    n, k = only_left + only_right, min(only_left, only_right)
    p = min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n) if n else 1.0
    return {"only_left": only_left, "only_right": only_right, "p": p}


def summarize(args) -> None:
    leaderboard = args.results_dir / "leaderboard.csv"
    board = {row["model"]: row for row in csv.DictReader(leaderboard.open())} if leaderboard.exists() else {}
    for dataset in ("dev64", "gsm8k_valid"):
        k8, k1 = (_correct(args.results_dir, f"{args.name}-{k}", dataset) for k in ("k8", "k1"))
        if k8 is None or k1 is None:
            continue
        counts = paired_counts(k8, k1)
        print(f"\n{dataset}: 8 loops {100 * sum(k8.values()) / len(k8):.1f}%, 1 loop {100 * sum(k1.values()) / len(k1):.1f}% "
              f"(only 8 loops right: {counts['only_left']}, only 1 loop: {counts['only_right']}, "
              f"exact paired p = {counts['p']:.3f})")
        for model in (f"{args.name}-explicit", "gpt2-sft", "codi", "dsa-data512-static-k8"):
            if board.get(model, {}).get(f"{dataset}_acc"):
                print(f"  {model}: {board[model][f'{dataset}_acc']}%")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--prepared-dir", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True, help="run directory on fast local disk")
    parser.add_argument("--backup-dir", type=Path, required=True, help="persistent plan and checkpoint copy (Drive)")
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--name", required=True, help="leaderboard row prefix")
    parser.add_argument("--smoke-report", type=Path)
    parser.add_argument("--device", default="cuda", choices=("cuda", "mps", "cpu"))
    parser.add_argument("--write-mode", default="static", choices=("static", "every_loop", "periodic", "learned"))
    parser.add_argument("--new-module-learning-rate", type=float)
    parser.add_argument("--records-per-update", type=int, default=RECORDS_PER_UPDATE)
    parser.add_argument("--batch-candidates", default="32,16,8,4,2,1",
                        type=lambda text: [int(value) for value in text.split(",")])
    parser.add_argument("--memory-fraction", type=float, default=0.8,
                        help="largest share of GPU memory the worst-case batch may use")
    parser.add_argument("--backup-minutes", type=float, default=20, help="target time between checkpoints")
    parser.add_argument("--report-minutes", type=float, default=5)
    parser.add_argument("--poll-seconds", type=float, default=30)
    parser.add_argument("--suite", default="dev,valid")
    parser.add_argument("--tiny", action="store_true", help="download-free tiny model, for testing this script")
    args = parser.parse_args(argv)
    if not args.tiny and args.smoke_report is None:
        parser.error("--smoke-report is required for the pretrained backbone")
    args.work_dir, args.backup_dir = args.work_dir.resolve(), args.backup_dir.resolve()
    plan = load_or_choose_plan(args, args.prepared_dir.resolve())
    train(args, args.prepared_dir.resolve(), plan)
    evaluate(args)
    summarize(args)


if __name__ == "__main__":
    main()
