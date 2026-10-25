#!/usr/bin/env python3
"""Evaluate any model on fixed datasets and print it against the persistent leaderboard.

    python scripts/evaluate.py --model gpt2-sft --suite dev
    python scripts/evaluate.py --model dsa --arg checkpoint=runs/learned --arg loops=16 \\
        --name dsa-learned-k16 --suite dev,valid
    python scripts/evaluate.py --model my_package.my_module:MyModel --suite dev
    python scripts/evaluate.py --table          # print the leaderboard only

Models: gpt2-vanilla, gpt2-sft, codi, lotus, dsa, or any "module:Class" that
subclasses dsa.bench.models.BenchModel. --arg key=value is passed to its
constructor (values parse as JSON when possible, e.g. 8, true, null).
Suites: overfit, dev, valid, ood, test. Test-split datasets need --test.
--limit N runs a quick smoke check that is printed but never saved.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def _parse_arg(text: str) -> tuple[str, object]:
    if "=" not in text:
        raise argparse.ArgumentTypeError(f"--arg expects key=value, got {text!r}")
    key, value = text.split("=", 1)
    try:
        return key, json.loads(value)
    except json.JSONDecodeError:
        return key, value


def main(argv=None) -> int:
    from dsa.bench import data as bench_data
    from dsa.bench import harness
    from dsa.bench.models import build_model

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", help="registered name or module:Class")
    parser.add_argument("--arg", action="append", type=_parse_arg, default=[], metavar="KEY=VALUE")
    parser.add_argument("--name", help="leaderboard row name (default: chosen by the model)")
    parser.add_argument("--suite", default="", help=f"comma list of {', '.join(bench_data.SUITES)}")
    parser.add_argument("--datasets", default="", help=f"comma list of {', '.join(bench_data.EVAL_DATASETS)}")
    parser.add_argument("--device", default="auto", choices=("auto", "cuda", "mps", "cpu"))
    parser.add_argument("--test", action="store_true", help="allow held-out test datasets")
    parser.add_argument("--limit", type=int, help="smoke check on the first N rows; not saved")
    parser.add_argument("--overwrite", action="store_true", help="replace a result whose identity changed")
    parser.add_argument("--table", action="store_true", help="print the leaderboard and exit")
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--results-dir", type=Path, default=harness.DEFAULT_RESULTS_DIR)
    args = parser.parse_args(argv)

    if args.table:
        harness.write_leaderboard(args.results_dir)
        print(harness.format_table(args.results_dir))
        return 0
    if not args.model:
        parser.error("--model is required (or use --table)")
    datasets = []
    for suite in filter(None, args.suite.split(",")):
        if suite not in bench_data.SUITES:
            parser.error(f"unknown suite {suite!r}; choose from {', '.join(bench_data.SUITES)}")
        datasets += bench_data.SUITES[suite]
    for dataset in filter(None, args.datasets.split(",")):
        if dataset not in bench_data.EVAL_DATASETS:
            parser.error(f"unknown dataset {dataset!r}; choose from {', '.join(bench_data.EVAL_DATASETS)}")
        datasets.append(dataset)
    datasets = list(dict.fromkeys(datasets))
    if not datasets:
        parser.error("choose --suite and/or --datasets")
    held_out = [name for name in datasets if bench_data.EVAL_DATASETS[name]["test"]]
    if held_out and not args.test:
        parser.error(f"{', '.join(held_out)} are held-out test sets; pass --test only for final numbers")

    kwargs = dict(args.arg)
    kwargs.setdefault("device", args.device)
    data_dir = (args.data_dir or bench_data.default_data_dir()).resolve()
    try:
        for dataset in datasets:  # Fail on missing/stale data before loading any weights.
            bench_data.load_eval_rows(dataset, data_dir)
        model = build_model(args.model, **kwargs)
        summaries = harness.evaluate(model, args.model, datasets, data_dir=data_dir, results_dir=args.results_dir,
                                     name=args.name, overwrite=args.overwrite, limit=args.limit)
    except (ValueError, FileNotFoundError, TypeError) as exc:
        parser.exit(2, f"evaluation stopped: {exc}\n")
    name = args.name or model.name
    for summary in summaries:
        print(f"{name} / {summary['dataset']}: {summary['correct']}/{summary['n']} "
              f"({100 * summary['accuracy']:.1f}%), strict {100 * summary['strict_accuracy']:.1f}%")
    if args.limit is None:
        print()
        print(harness.format_table(args.results_dir, highlight=name))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
