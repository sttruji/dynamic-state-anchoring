#!/usr/bin/env python3
"""Question-only validation/final-test evaluation with persistent resume."""
from pathlib import Path
import argparse
import importlib.metadata
import json
import math
import os
import platform
import re
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dsa.data.adapters import canonical_record, stable_json_hash
from dsa.devices import mps_memory_snapshot, resolve_device, synchronize
from dsa.provenance import file_sha256
from dsa.training.pilot import tokenizer_directory_sha256
from scripts.smoke_pretrained_dsa import atomic_json


def _json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _contained(directory, relative):
    directory = Path(directory).resolve()
    path = (directory / relative).resolve()
    if not path.is_relative_to(directory):
        raise ValueError("Artifact path escapes its declared directory")
    return path


def resolve_checkpoint(run_dir):
    run_dir = Path(run_dir).resolve()
    latest = _json(run_dir / "latest.json")
    checkpoint = _contained(run_dir, latest["path"])
    if checkpoint.parent != run_dir or not re.fullmatch(r"checkpoint-[0-9]{8}", checkpoint.name):
        raise ValueError("Latest checkpoint is outside the run directory or has an invalid name")
    if not (checkpoint / "COMPLETE.json").is_file():
        raise ValueError("Latest checkpoint is incomplete")
    completion = _json(checkpoint / "COMPLETE.json")
    step = int(checkpoint.name.removeprefix("checkpoint-"))
    if (completion.get("schema_version") != 1 or completion.get("step") != step
            or latest.get("step") != step):
        raise ValueError("Checkpoint completion marker and latest step disagree")
    required = ("training_state.pt", "model/anchor.pt", "model/anchor_config.json",
                "model/backbone/config.json")
    if any(not (checkpoint / name).is_file() for name in required):
        raise ValueError("Completed checkpoint is missing required model or training state")
    if not (checkpoint / "model/tokenizer").is_dir():
        raise ValueError("Completed checkpoint is missing its tokenizer")
    return checkpoint


def _checkpoint_files(checkpoint):
    hashes = {}
    for path in sorted(checkpoint.rglob("*")):
        if path.is_file():
            if not path.resolve().is_relative_to(checkpoint.resolve()):
                raise ValueError("Checkpoint contains an external file link")
            hashes[str(path.relative_to(checkpoint))] = file_sha256(path)
    completion = _json(checkpoint / "COMPLETE.json")
    # Current writers mark an atomic directory complete. The hashes below bind
    # its weights and optimizer files to this evaluation and future resume.
    for name, expected in completion.get("files_sha256", {}).items():
        if hashes.get(name) != expected:
            raise ValueError("Checkpoint file checksum differs from completion marker")
    return hashes


def _validate_training_identity(run_dir, checkpoint, manifest, manifest_sha256):
    import torch
    run = _json(run_dir / "run_manifest.json")
    contract = run["contract"]
    if (contract.get("prepared_manifest_sha256") != manifest_sha256
            or run.get("prepared_manifest") != manifest):
        raise ValueError("Prepared dataset identity differs from the training run")
    expected_identity = {key: manifest[key] for key in
                         ("backbone", "revision", "tiny", "tokenizer_sha256", "splits")}
    if contract.get("prepared_identity") != expected_identity:
        raise ValueError("Prepared source/model/tokenizer identity differs from training")
    # mmap reads metadata without materializing the full Adam tensors in RAM.
    # This is a trusted local artifact containing Python and NumPy RNG tuples.
    state = torch.load(checkpoint / "training_state.pt", map_location="cpu",
                       weights_only=False, mmap=True)
    completion = _json(checkpoint / "COMPLETE.json")
    if (state.get("schema_version") != 1 or state.get("step") != completion["step"]
            or state.get("contract") != contract
            or state.get("initial_state_sha256") != run.get("initial_state_sha256")
            or state.get("records_consumed") != state["step"] * contract["settings"]["gradient_accumulation"]):
        raise ValueError("Checkpoint state identity differs from its completed training run")
    anchor = _json(checkpoint / "model/anchor_config.json")
    settings = contract["settings"]
    checks = {"workspace_size": settings["workspace"], "loop_budget": settings["loops"],
              "gate_threshold": settings["gate_threshold"], "write_mode": settings["write_mode"],
              "periodic_interval": settings["periodic_interval"]}
    if any(anchor.get(key) != value for key, value in checks.items()):
        raise ValueError("Saved anchor configuration differs from the training contract")
    if anchor.get("recurrence_mode", "standard") != settings.get("recurrence_mode", "standard"):
        raise ValueError("Saved recurrence differs from the training contract")
    for name in ("dsa/models/backbone.py", "dsa/models/anchored_loop.py", "dsa/models/flt_memory.py", "dsa/evaluation/evaluate.py"):
        expected = contract.get("runtime", {}).get("implementation_sha256", {}).get(name)
        if expected is not None and file_sha256(ROOT / name) != expected:
            raise ValueError("Inference implementation differs from the training run")
    return anchor, contract


def _load_model(directory):
    import torch
    from transformers import AutoModelForCausalLM
    from dsa.models.anchored_loop import AnchorConfig, AnchoredLoop
    from dsa.models.backbone import BackboneAdapter
    # Select eager before construction: changing the config afterwards leaves
    # already-created SDPA attention modules in place.
    backbone = BackboneAdapter(AutoModelForCausalLM.from_pretrained(
        directory / "backbone", attn_implementation="eager", torch_dtype=torch.float32,
        local_files_only=True))
    model = AnchoredLoop(backbone, AnchorConfig(**_json(directory / "anchor_config.json")))
    memory = torch.load(directory / "anchor.pt", map_location="cpu", weights_only=True)
    missing, unexpected = model.load_state_dict(memory, strict=False)
    if unexpected or any(not name.startswith("backbone.") for name in missing):
        raise ValueError("Incomplete or incompatible anchor checkpoint")
    return model


def _load_tokenizer(directory, tiny):
    if tiny:
        from dsa.data.tokenize import TinyTokenizer
        return TinyTokenizer.from_pretrained(directory)
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(directory, use_fast=True, local_files_only=True)


def _validate_tokenizers(prepared_dir, checkpoint, manifest, anchor):
    if tokenizer_directory_sha256(prepared_dir / "tokenizer") != manifest["tokenizer_sha256"]:
        raise ValueError("Prepared tokenizer checksum mismatch")
    prepared = _load_tokenizer(prepared_dir / "tokenizer", manifest["tiny"])
    saved = _load_tokenizer(checkpoint / "model/tokenizer", manifest["tiny"])
    if (prepared.get_vocab() != saved.get_vocab()
            or prepared.pad_token_id != saved.pad_token_id
            or prepared.eos_token_id != saved.eos_token_id
            or prepared.encode("\n### <SA>", add_special_tokens=False)
            != saved.encode("\n### <SA>", add_special_tokens=False)):
        raise ValueError("Checkpoint tokenizer differs from the prepared training tokenizer")
    marker = saved.encode("<SA>", add_special_tokens=False)
    if (len(marker) != 1 or anchor.get("marker_token_id") != marker[0]
            or anchor.get("pad_token_id") != saved.pad_token_id
            or anchor.get("eos_token_id") != saved.eos_token_id):
        raise ValueError("Checkpoint control token identities disagree")
    return saved


def _validate_prediction(record, index, rows, budget):
    from dsa.evaluation.evaluate import extract_answer, score_answer
    if index >= len(rows):
        raise ValueError("Saved prediction log exceeds the evaluation denominator")
    row = rows[index]
    gold = row["final_answer"].replace(",", "").strip()
    required = {"index", "sample_id", "question_hash", "gold", "generated_text", "prediction",
                "valid_numeric_answer", "latency_seconds", "recurrent_forwards",
                "gate_probabilities", "writes", "correct"}
    if not isinstance(record, dict) or not required <= record.keys():
        raise ValueError("Saved prediction is missing required fields")
    if (record["index"] != index or record["sample_id"] != row["sample_id"]
            or record["question_hash"] != row["question_hash"] or record["gold"] != gold):
        raise ValueError("Saved prediction order/source differs")
    if (not isinstance(record["generated_text"], str) or not isinstance(record["prediction"], str)
            or record["prediction"] != extract_answer(record["generated_text"])
            or record["correct"] != score_answer(record["prediction"], gold)):
        raise ValueError("Saved prediction/scoring fields disagree")
    try:
        valid = math.isfinite(float(record["prediction"]))
    except ValueError:
        valid = False
    if record["valid_numeric_answer"] != valid:
        raise ValueError("Saved answer-validity fields disagree")
    if (not isinstance(record["latency_seconds"], (int, float))
            or not math.isfinite(record["latency_seconds"]) or record["latency_seconds"] < 0
            or record["recurrent_forwards"] != budget
            or len(record["gate_probabilities"]) != budget or len(record["writes"]) != budget
            or any(not isinstance(value, (float, int)) or not 0 <= value <= 1
                   for value in record["gate_probabilities"])
            or any(value not in (0, 1) for value in record["writes"])):
        raise ValueError("Saved prediction compute/latency fields disagree")


def _recover_predictions(path, rows, budget):
    if not path.exists():
        return []
    payload = path.read_bytes()
    lines = payload.splitlines(keepends=True)
    predictions = []
    valid_bytes = 0
    for index, line in enumerate(lines):
        try:
            record = json.loads(line)
        except (json.JSONDecodeError, UnicodeDecodeError):
            if index != len(lines) - 1 or line.endswith(b"\n"):
                raise ValueError("Corrupt saved prediction log") from None
            break
        _validate_prediction(record, index, rows, budget)
        predictions.append(record)
        valid_bytes += len(line)
    recovered = payload[:valid_bytes]
    # Valid JSON at EOF without its newline is complete and must be retained.
    if recovered and not recovered.endswith(b"\n"):
        recovered += b"\n"
    if recovered != payload:
        temporary = path.with_name("." + path.name + ".tmp")
        temporary.write_bytes(recovered)
        os.replace(temporary, path)
    return predictions


def _cached_report(report_path, predictions_path, identity, predictions, expected_total):
    if not report_path.exists():
        return None
    report = _json(report_path)
    if not report.get("complete"):
        return None
    if (report.get("identity") != identity or len(predictions) != expected_total
            or report.get("total") != expected_total
            or not predictions_path.is_file()
            or report.get("predictions_sha256") != file_sha256(predictions_path)
            or report.get("correct") != sum(row["correct"] for row in predictions)
            or report.get("ordered_evaluation_id_sha256")
            != stable_json_hash([row["sample_id"] for row in predictions])):
        raise ValueError("Completed report does not match its prediction log or denominator")
    return {**report, "reused_completed_evaluation": True}


def evaluate_run(prepared_dir, run_dir, output_dir, *, split="valid", loop_budget=None,
                 limit=0, max_new_tokens=64, device="auto", final_test=False):
    import torch
    from dsa.evaluation.evaluate import predict_question, score_answer
    prepared_dir, run_dir, output_dir = [Path(path).resolve() for path in
                                        (prepared_dir, run_dir, output_dir)]
    if split not in {"valid", "test"}:
        raise ValueError("Evaluation split must be valid or test")
    if split == "test" and (not final_test or limit):
        raise ValueError("Final test requires --final-test and the complete test denominator (--limit 0)")
    if limit < 0 or max_new_tokens < 1 or (loop_budget is not None and loop_budget < 1):
        raise ValueError("Invalid evaluation limits or loop budget")
    device = str(resolve_device(device))
    manifest_path = prepared_dir / "manifest.json"
    manifest = _json(manifest_path)
    info = manifest["splits"][split]
    data_path = _contained(prepared_dir, info["path"])
    if file_sha256(data_path) != info["sha256"]:
        raise ValueError("Prepared evaluation source checksum mismatch")
    checkpoint = resolve_checkpoint(run_dir)
    anchor, contract = _validate_training_identity(
        run_dir, checkpoint, manifest, file_sha256(manifest_path))
    tokenizer = _validate_tokenizers(prepared_dir, checkpoint, manifest, anchor)
    checkpoint_hashes = _checkpoint_files(checkpoint)
    budget = anchor["loop_budget"] if loop_budget is None else loop_budget
    with data_path.open(encoding="utf-8") as handle:
        all_rows = [canonical_record(json.loads(line)) for line in handle if line.strip()]
    if not all_rows or ("records" in info and len(all_rows) != info["records"]):
        raise ValueError("Prepared evaluation record count differs from its declared denominator")
    if ("ordered_sample_ids_sha256" in info and stable_json_hash([row["sample_id"] for row in all_rows])
            != info["ordered_sample_ids_sha256"]):
        raise ValueError("Prepared ordered evaluation identities differ from manifest")
    official = manifest.get("source_provenance", {}).get("official_test_comparison", {})
    if split == "test" and official.get("status") == "compared":
        if (not official.get("ordered_question_answer_match")
                or len(all_rows) != official.get("official_rows")
                or len(all_rows) != official.get("supplied_rows")):
            raise ValueError("Final test denominator differs from the verified official test")
    rows = all_rows[:limit] if limit else all_rows
    identity = {"schema_version": "dsa_evaluation_identity_v1", "checkpoint": str(checkpoint),
                "checkpoint_files_sha256": checkpoint_hashes,
                "source_sha256": info["sha256"], "prepared_manifest_sha256": file_sha256(manifest_path),
                "split": split, "loop_budget": budget, "limit": limit,
                "expected_total": len(rows), "max_new_tokens": max_new_tokens, "device": device,
                "runtime": {"torch": str(torch.__version__),
                            "transformers": importlib.metadata.version("transformers"),
                            "cuda": torch.version.cuda,
                            "device_name": torch.cuda.get_device_name() if device == "cuda"
                            else platform.processor() or platform.machine()},
                "scoring": "LOTUS eb77e2f extraction/float equality/string fallback",
                "evaluator_sha256": file_sha256(Path(__file__)),
                "device_helpers_sha256": file_sha256(ROOT / "dsa/devices.py"),
                "evaluation_helpers_sha256": file_sha256(ROOT / "dsa/evaluation/evaluate.py")}
    output_dir.mkdir(parents=True, exist_ok=True)
    identity_path = output_dir / "identity.json"
    if identity_path.exists() and _json(identity_path) != identity:
        raise ValueError("Existing evaluation has different settings or checkpoint bytes; choose a new output directory")
    if not identity_path.exists() and ((output_dir / "predictions.jsonl").exists()
                                       or (output_dir / "report.json").exists()):
        raise ValueError("Existing evaluation artifacts are missing their identity")
    atomic_json(identity_path, identity)
    report_path = output_dir / "report.json"
    predictions_path = output_dir / "predictions.jsonl"
    # Check the original journal before recovery can change a completed log.
    if report_path.exists() and _json(report_path).get("complete"):
        if (not predictions_path.exists()
                or _json(report_path).get("predictions_sha256") != file_sha256(predictions_path)):
            raise ValueError("Completed report prediction checksum mismatch")
    predictions = _recover_predictions(predictions_path, rows, budget)
    cached = _cached_report(report_path, predictions_path, identity, predictions, len(rows))
    if cached is not None:
        return cached
    model = _load_model(checkpoint / "model").to(device).eval()
    if model.backbone.embeddings.num_embeddings != len(tokenizer):
        raise ValueError("Saved tokenizer vocabulary does not match model embeddings")
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    with predictions_path.open("a", encoding="utf-8") as handle:
        for index in range(len(predictions), len(rows)):
            row = rows[index]
            synchronize(device)
            tick = time.perf_counter()
            prediction = predict_question(model, tokenizer, row["question"],
                                          loop_budget=budget, max_new_tokens=max_new_tokens)
            synchronize(device)
            prediction["latency_seconds"] = time.perf_counter() - tick
            gold = row["final_answer"].replace(",", "").strip()
            result = {"index": index, "sample_id": row["sample_id"],
                      "question_hash": row["question_hash"], "gold": gold, **prediction,
                      "correct": score_answer(prediction["prediction"], gold)}
            _validate_prediction(result, index, rows, budget)
            handle.write(json.dumps(result, ensure_ascii=False) + "\n")
            handle.flush()
            predictions.append(result)
            if (index + 1) % 25 == 0:
                print(json.dumps({"evaluated": index + 1, "total": len(rows)}), flush=True)
    total = len(predictions)
    if total != len(rows):
        raise AssertionError("Evaluation denominator incomplete")
    correct = sum(row["correct"] for row in predictions)
    invalid = sum(not row["valid_numeric_answer"] for row in predictions)
    report = {"schema_version": "dsa_evaluation_report_v1", "complete": True,
              "identity": identity, "split": split, "total": total, "correct": correct,
              "accuracy": correct / total, "invalid_answer_count": invalid,
              "invalid_answer_rate": invalid / total,
              "ordered_evaluation_id_sha256": stable_json_hash([row["sample_id"] for row in predictions]),
              "predictions_sha256": file_sha256(predictions_path), "loop_budget": budget,
              "max_new_tokens": max_new_tokens, "executed_recurrent_forwards": total * budget,
              "device": device, "average_latency_seconds": sum(row["latency_seconds"] for row in predictions) / total,
              "total_measured_prediction_seconds": sum(row["latency_seconds"] for row in predictions),
              "current_session_seconds": time.perf_counter() - started,
              "peak_cuda_memory_bytes": torch.cuda.max_memory_allocated() if device == "cuda" else None,
              **mps_memory_snapshot(device),
              "all_source_examples_retained": len(rows) == len(all_rows),
              "purpose": "final held-out test" if split == "test" else "implementation-pilot validation",
              "question_only_inference": True, "annotation_filtering": False,
              "checkpoint": str(checkpoint), "tiny": manifest["tiny"],
              "source_regime": manifest.get("source_provenance", {}).get("regime", "unspecified"),
              "official_test_denominator_verified": split == "test" and official.get("status") == "compared"}
    atomic_json(report_path, report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared-dir", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split", choices=["valid", "test"], default="valid")
    parser.add_argument("--loop-budget", type=int)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    parser.add_argument("--final-test", action="store_true")
    args = parser.parse_args()
    print(json.dumps(evaluate_run(args.prepared_dir, args.run_dir, args.output_dir,
                                 split=args.split, loop_budget=args.loop_budget,
                                 limit=args.limit, max_new_tokens=args.max_new_tokens,
                                 device=args.device, final_test=args.final_test), indent=2))


if __name__ == "__main__":
    main()
