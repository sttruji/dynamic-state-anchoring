#!/usr/bin/env python3
"""Download, normalize, tag and split every dataset, then check it against the team lock.

    python scripts/setup_data.py            # everything (about 2 minutes, ~1 GB)
    python scripts/setup_data.py --eval-only  # just the evaluation sets (fast)
    python scripts/setup_data.py --verify   # check existing files, no downloads

Outputs go to $DSA_DATA_DIR (default: <repo>/data, which git ignores):
    data/bench/train/*.jsonl   training corpora with deterministic <SA> tags
    data/bench/eval/*.jsonl    fixed evaluation sets used by scripts/evaluate.py
    data/gsm_*.json            pinned LOTUS arrays for scripts/prepare_dsa_pilot.py

Maintainers only: --write-lock records hashes after adding or changing a source;
commit provenance/bench_data_lock.json together with the code change.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dsa.bench import data as bench_data


def _verify(data_dir: Path, lock: dict, keys) -> list[str]:
    problems = []
    for key in keys:
        expected = lock["files"].get(key)
        path = data_dir / "bench" / key
        if expected is None:
            problems.append(f"{key}: not in the lock (maintainer: run --write-lock and commit)")
        elif not path.is_file():
            problems.append(f"{key}: missing")
        elif bench_data.file_sha256(path) != expected["sha256"]:
            problems.append(f"{key}: differs from the team copy")
    return problems


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=Path, default=None, help="default: $DSA_DATA_DIR or <repo>/data")
    parser.add_argument("--eval-only", action="store_true", help="skip the large training corpora")
    parser.add_argument("--verify", action="store_true", help="only check existing files against the lock")
    parser.add_argument("--write-lock", action="store_true", help="maintainers: record current hashes")
    args = parser.parse_args(argv)
    data_dir = (args.data_dir or bench_data.default_data_dir()).resolve()
    lock = bench_data.load_lock()

    if args.verify:
        keys = [key for key in lock["files"] if not (args.eval_only and key.startswith("train/"))]
        problems = _verify(data_dir, lock, keys)
        for problem in problems:
            print("FAIL", problem)
        print("OK: all files match the lock" if not problems else f"{len(problems)} problem(s)")
        return 1 if problems else 0

    print(f"Data directory: {data_dir}")
    outputs = bench_data.build_all(data_dir, with_train=not args.eval_only)
    written = {}
    for key in sorted(outputs):
        written[key] = bench_data.write_jsonl(data_dir / "bench" / key, outputs[key])
        print(f"  {key:34s} {written[key]['rows']:>8,d} rows  {written[key]['sha256'][:12]}")

    overlap = bench_data.contamination_report(outputs)
    leaked = {key: count for key, count in overlap.items() if count}
    if overlap:
        print("Train/eval question overlap:", leaked or "none")

    if args.write_lock:
        lock = {"schema": bench_data.LOCK_SCHEMA, "files": {**lock.get("files", {}), **written},
                "train_eval_overlap": {**lock.get("train_eval_overlap", {}), **overlap}}
        bench_data.LOCK_PATH.write_text(json.dumps(lock, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"Wrote {bench_data.LOCK_PATH.relative_to(bench_data.REPOSITORY)}; commit it.")
        return 0

    problems = _verify(data_dir, lock, written)
    for problem in problems:
        print("FAIL", problem)
    if problems:
        print("Data does not match the team lock. Do not evaluate; pull the latest repo or ask a maintainer.")
        return 1
    print("OK: every file matches provenance/bench_data_lock.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
