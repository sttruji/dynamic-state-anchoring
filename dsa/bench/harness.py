"""Run a BenchModel over fixed datasets and keep a persistent, comparable leaderboard.

Layout under ``results/`` (commit all of it; it is the team's shared record):

    runs/<model>/<dataset>.json          one finished result, the source of truth
    predictions/<model>/<dataset>.jsonl  every raw output, for paired analysis
    leaderboard.csv                      rebuilt from runs/ after every evaluation

A finished result is reused only when model identity, dataset hash and scorer
version all match. Anything else is refused instead of silently overwritten.
Interrupted runs resume from their predictions file.
"""
from __future__ import annotations

import csv
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import re
import subprocess
import tempfile
import time
from datetime import datetime, timezone

from dsa.bench import data as bench_data
from dsa.bench.scoring import SCORER_VERSION, score

REPOSITORY = Path(__file__).resolve().parents[2]
DEFAULT_RESULTS_DIR = REPOSITORY / "results"
_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._+-]*\Z")
BASELINE_ORDER = ("gpt2-vanilla", "gpt2-sft", "codi", "lotus")


def _atomic_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        handle.write(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n")
    os.chmod(temporary, 0o644)  # mkstemp creates 0600; results are shared files.
    os.replace(temporary, path)


def _git_commit() -> str | None:
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPOSITORY, text=True,
                                         stderr=subprocess.DEVNULL).strip()
        dirty = subprocess.check_output(["git", "status", "--porcelain", "--", "dsa", "scripts"],
                                        cwd=REPOSITORY, text=True, stderr=subprocess.DEVNULL).strip()
        return commit + ("-dirty" if dirty else "")
    except (OSError, subprocess.CalledProcessError):
        return None


def _runtime(model) -> dict:
    versions = {}
    for package in ("torch", "transformers"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    device = getattr(model, "device", None)
    return {"device": str(device) if device is not None else None, "platform": platform.platform(),
            "python": platform.python_version(), **versions, "git_commit": _git_commit()}


def _read_predictions(path: Path, identity: dict, rows: list[dict], overwrite: bool) -> list[dict]:
    if not path.exists():
        return []
    lines = path.read_text(encoding="utf-8").split("\n")
    try:
        header = json.loads(lines[0])["_identity"]
    except (json.JSONDecodeError, KeyError, TypeError):
        header = None
    if header != identity:
        if overwrite:
            path.unlink()
            return []
        raise ValueError(f"{path} was produced by a different model/dataset identity; "
                         "use a new --name or pass --overwrite")
    predictions = []
    for line in lines[1:]:
        if not line.strip():
            continue
        try:
            predictions.append(json.loads(line))
        except json.JSONDecodeError:
            break  # A partial trailing line from an interrupted write; redo it.
    for index, prediction in enumerate(predictions):
        if index >= len(rows) or prediction["id"] != rows[index]["id"]:
            raise ValueError(f"{path} does not follow the dataset order; delete it or pass --overwrite")
    # Rewrite without any partial tail so appends start on a clean line.
    with path.open("w", encoding="utf-8") as handle:
        handle.write(json.dumps({"_identity": identity}, sort_keys=True, ensure_ascii=False) + "\n")
        for prediction in predictions:
            handle.write(json.dumps(prediction, sort_keys=True, ensure_ascii=False) + "\n")
    return predictions


def _summarize(predictions: list[dict]) -> dict:
    count = len(predictions)
    correct = sum(item["correct"] for item in predictions)
    strict = sum(item["strict_correct"] for item in predictions)
    valid = sum(item["valid_answer"] for item in predictions)
    seconds = sum(item["seconds"] for item in predictions)
    return {"n": count, "correct": correct, "accuracy": correct / count if count else 0.0,
            "strict_correct": strict, "strict_accuracy": strict / count if count else 0.0,
            "valid_answer_rate": valid / count if count else 0.0,
            "seconds_per_question": seconds / count if count else 0.0}


def _run_rows(model, rows: list[dict], handle=None, progress: str = "") -> list[dict]:
    results = []
    batch = max(1, int(getattr(model, "batch_size", 1)))
    for start in range(0, len(rows), batch):
        chunk = rows[start:start + batch]
        started = time.perf_counter()
        texts = model.generate([row["question"] for row in chunk])
        elapsed = (time.perf_counter() - started) / len(chunk)
        if len(texts) != len(chunk) or not all(isinstance(text, str) for text in texts):
            raise ValueError("model.generate must return one string per question")
        for row, text in zip(chunk, texts):
            item = {"id": row["id"], "gold": row["answer"], "generated_text": text, "seconds": elapsed,
                    **score(text, row["answer"], row.get("answer_type", "numeric"))}
            results.append(item)
            if handle is not None:
                handle.write(json.dumps(item, sort_keys=True, ensure_ascii=False) + "\n")
        if handle is not None:
            handle.flush()
            os.fsync(handle.fileno())
        if progress:
            done = start + len(chunk)
            print(f"\r  {progress}: {done}/{len(rows)}", end="", flush=True)
    if progress:
        print()
    return results


def evaluate(model, model_spec: str, datasets: list[str], *, data_dir: Path,
             results_dir: Path = DEFAULT_RESULTS_DIR, name: str | None = None,
             overwrite: bool = False, limit: int | None = None) -> list[dict]:
    """Evaluate one model on each dataset; ``limit`` runs an unsaved smoke check."""
    name = name or model.name
    if not _NAME.match(name):
        raise ValueError(f"model name {name!r} may only use letters, digits and . _ + -")
    model_identity = model.identity()
    json.dumps(model_identity)  # Fail before any work if the identity is not serializable.
    summaries = []
    for dataset in datasets:
        rows, dataset_sha = bench_data.load_eval_rows(dataset, data_dir)
        if limit is not None:
            rows = rows[:limit]
            print(f"[{name} / {dataset}] smoke check on {len(rows)} rows (not saved)")
            predictions = _run_rows(model, rows, progress=dataset)
            for item in predictions[:3]:
                print(f"  gold={item['gold']!r} pred={item['prediction']!r} text={item['generated_text'][:120]!r}")
            summaries.append({"model": name, "dataset": dataset, **_summarize(predictions), "saved": False})
            continue
        identity = {"model": name, "model_spec": model_spec, "model_identity": model_identity,
                    "dataset": dataset, "dataset_sha256": dataset_sha, "rows": len(rows),
                    "scorer_version": SCORER_VERSION}
        result_path = results_dir / "runs" / name / f"{dataset}.json"
        prediction_path = results_dir / "predictions" / name / f"{dataset}.jsonl"
        if result_path.exists():
            existing = json.loads(result_path.read_text(encoding="utf-8"))
            if existing.get("identity") == identity:
                print(f"[{name} / {dataset}] already evaluated with this exact identity; reusing")
                summaries.append(existing)
                continue
            if not overwrite:
                changed = sorted(key for key in identity if existing.get("identity", {}).get(key) != identity[key])
                raise ValueError(f"results/runs/{name}/{dataset}.json exists with different {changed}; "
                                 "choose a new --name (recommended) or pass --overwrite")
            result_path.unlink()
        prediction_path.parent.mkdir(parents=True, exist_ok=True)
        predictions = _read_predictions(prediction_path, identity, rows, overwrite)
        if predictions:
            print(f"[{name} / {dataset}] resuming at {len(predictions)}/{len(rows)}")
        else:
            prediction_path.write_text(json.dumps({"_identity": identity}, sort_keys=True,
                                                  ensure_ascii=False) + "\n", encoding="utf-8")
        started = time.perf_counter()
        with prediction_path.open("a", encoding="utf-8") as handle:
            predictions += _run_rows(model, rows[len(predictions):], handle, progress=f"{name} / {dataset}")
        result = {"identity": identity, "model": name, "dataset": dataset, **_summarize(predictions),
                  "session_seconds": time.perf_counter() - started, "runtime": _runtime(model),
                  "finished_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}
        _atomic_json(result_path, result)
        summaries.append(result)
    if limit is None:
        write_leaderboard(results_dir)
    return summaries


def collect_results(results_dir: Path) -> tuple[list[dict], list[str]]:
    """Return comparable results plus warnings for stale ones (old scorer or data)."""
    current, stale = [], []
    lock = bench_data.load_lock()
    for path in sorted((results_dir / "runs").glob("*/*.json")):
        result = json.loads(path.read_text(encoding="utf-8"))
        identity = result.get("identity", {})
        expected = lock.get("files", {}).get(bench_data.eval_file_key(result["dataset"]), {}).get("sha256")
        if identity.get("scorer_version") != SCORER_VERSION:
            stale.append(f"{path.relative_to(results_dir)}: scorer {identity.get('scorer_version')}")
        elif expected is not None and identity.get("dataset_sha256") != expected:
            stale.append(f"{path.relative_to(results_dir)}: dataset changed since evaluation")
        else:
            current.append(result)
    return current, stale


def _model_order(names) -> list[str]:
    known = [name for name in BASELINE_ORDER if name in names]
    return known + sorted(name for name in names if name not in BASELINE_ORDER)


def write_leaderboard(results_dir: Path = DEFAULT_RESULTS_DIR) -> Path:
    """One row per model; accuracy, strict accuracy and n for each dataset."""
    results, _ = collect_results(results_dir)
    datasets = [name for name in bench_data.EVAL_DATASETS if any(r["dataset"] == name for r in results)]
    table = {}
    for result in results:
        table.setdefault(result["model"], {})[result["dataset"]] = result
    header = ["model"] + [f"{dataset}_{field}" for dataset in datasets for field in ("acc", "strict_acc", "n")]
    path = results_dir / "leaderboard.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(header)
        for model in _model_order(table):
            row = [model]
            for dataset in datasets:
                result = table[model].get(dataset)
                row += (["", "", ""] if result is None else
                        [f"{100 * result['accuracy']:.2f}", f"{100 * result['strict_accuracy']:.2f}", result["n"]])
            writer.writerow(row)
    return path


def format_table(results_dir: Path = DEFAULT_RESULTS_DIR, highlight: str | None = None,
                 datasets: list[str] | None = None) -> str:
    """Accuracy % (strict %) per dataset, for printing in a terminal or notebook."""
    results, stale = collect_results(results_dir)
    present = [name for name in bench_data.EVAL_DATASETS if any(r["dataset"] == name for r in results)]
    datasets = [name for name in (datasets or present) if name in present]
    table = {}
    for result in results:
        table.setdefault(result["model"], {})[result["dataset"]] = result
    if not table:
        return "No results yet. Run: python scripts/evaluate.py --model gpt2-sft --suite dev"
    cells = [["model"] + [f"{d} (n={next(r['n'] for r in results if r['dataset'] == d)})" for d in datasets]]
    for model in _model_order(table):
        row = [("-> " if model == highlight else "   ") + model]
        for dataset in datasets:
            result = table[model].get(dataset)
            row.append("" if result is None else
                       f"{100 * result['accuracy']:5.1f} ({100 * result['strict_accuracy']:5.1f})")
        cells.append(row)
    widths = [max(len(row[i]) for row in cells) for i in range(len(cells[0]))]
    lines = ["  ".join(cell.ljust(width) for cell, width in zip(row, widths)) for row in cells]
    lines.insert(1, "  ".join("-" * width for width in widths))
    lines.append("accuracy % (strict %); strict requires one clean scalar after the answer marker.")
    if stale:
        lines.append("Excluded stale results (re-run them): " + "; ".join(stale))
    return "\n".join(lines)
