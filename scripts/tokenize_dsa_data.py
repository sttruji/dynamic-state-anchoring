#!/usr/bin/env python3
"""Build tokenized training artifacts from a complete annotated JSONL corpus."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dsa.data.adapters import stable_json_hash
from dsa.data.milestones import SCHEMA_VERSION
from dsa.data.tokenize import TinyTokenizer, register_marker_token, tokenize_record


def _reject_constant(value: str) -> None:
    raise ValueError(f"nonstandard JSON constant: {value}")


def _json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n"


def _stage(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        Path(name).unlink(missing_ok=True)
        raise
    return Path(name)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--tokenizer-dir", type=Path, required=True, help="save tokenizer and provenance identity here")
    parser.add_argument("--tokenizer", type=Path, help="existing local Hugging Face fast tokenizer directory; default: TinyTokenizer")
    parser.add_argument("--loops", type=int, default=4)
    args = parser.parse_args(argv)
    if args.loops < 1:
        parser.error("--loops must be a positive integer")
    files = [args.input.resolve(), args.output.resolve(), args.report.resolve()]
    if len(set(files)) != len(files):
        parser.error("input, output and report must be distinct files")
    tokenizer_dir = args.tokenizer_dir.resolve()
    if any(path == tokenizer_dir or tokenizer_dir in path.parents for path in files):
        parser.error("tokenizer directory must not contain input, output or report")
    if args.tokenizer is not None and not args.tokenizer.is_dir():
        parser.error("--tokenizer must name an existing local directory; network downloads are disabled")
    if tokenizer_dir.exists() and not tokenizer_dir.is_dir():
        parser.error("--tokenizer-dir must be a directory")
    if tokenizer_dir.exists() and any(tokenizer_dir.iterdir()):
        try:
            prior = json.loads((tokenizer_dir / "tokenizer_identity.json").read_text(encoding="utf-8"))
            expected = set(prior["tokenizer_file_sha256s"]) | {"tokenizer_identity.json"}
            actual = {str(path.relative_to(tokenizer_dir)) for path in tokenizer_dir.rglob("*")}
            if actual != expected:
                raise ValueError("directory contains files outside the saved tokenizer")
        except (OSError, ValueError, KeyError, TypeError) as exc:
            parser.error(f"choose an empty tokenizer directory or a previous DSA tokenizer output: {exc}")
    try:
        original_bytes = args.input.read_bytes()
        # Unicode U+2028/U+2029 are legal inside JSON strings, not JSONL dividers.
        lines = original_bytes.decode("utf-8").split("\n")
        if lines and not lines[-1]:
            lines.pop()
    except (OSError, UnicodeError) as exc:
        parser.error(str(exc))
    records = []
    policy = None
    for line_number, line in enumerate(lines, 1):
        try:
            record = json.loads(line, parse_constant=_reject_constant)
            if not isinstance(record, dict) or record.get("schema_version") != SCHEMA_VERSION:
                raise ValueError(f"expected annotated {SCHEMA_VERSION} record")
            record_policy = record.get("policy")
            if not isinstance(record_policy, dict) or not isinstance(record_policy.get("marker_token"), str):
                raise ValueError("annotation policy must declare a marker_token")
            if policy is None:
                policy = record_policy
            elif policy != record_policy:
                raise ValueError("all input records must share an annotation policy")
            records.append(record)
        except (ValueError, TypeError) as exc:
            parser.error(f"{args.input}: line {line_number}: {exc}; no output files were replaced")
    if not records:
        parser.error("input corpus is empty; no output files were replaced")
    try:
        if args.tokenizer is None:
            tokenizer = TinyTokenizer(marker_token=policy["marker_token"])
        else:
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(str(args.tokenizer.resolve()), use_fast=True, local_files_only=True)
        marker_id = register_marker_token(tokenizer, marker_token=policy["marker_token"])
    except (ValueError, OSError, ImportError, TypeError) as exc:
        parser.error(f"could not initialize local tokenizer: {exc}; no output files were replaced")
    tokens = []
    for line_number, record in enumerate(records, 1):
        try:
            tokenized = tokenize_record(record, tokenizer, args.loops)
            tokens.append({"schema_version": "dsa_tokenized_v1", "question_hash": record["question_hash"],
                           "annotation_policy_sha256": stable_json_hash(policy), **tokenized})
        except (ValueError, TypeError, KeyError, OverflowError) as exc:
            parser.error(f"{args.input}: line {line_number}: {exc}; no output files were replaced")
    reasons = Counter(record["controller_mask_reason"] for record in tokens if not record["controller_supervised"])
    output_text = "".join(json.dumps(record, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n" for record in tokens)
    identity = {
        "schema_version": "dsa_tokenization_identity_v1",
        "input_sha256": hashlib.sha256(original_bytes).hexdigest(),
        "tokenized_output_sha256": hashlib.sha256(output_text.encode("utf-8")).hexdigest(),
        "annotation_policy": policy,
        "annotation_policy_sha256": stable_json_hash(policy),
        "marker_token": policy["marker_token"], "marker_token_id": marker_id,
        "tokenizer_class": type(tokenizer).__name__,
        "tokenizer_source": str(args.tokenizer.resolve()) if args.tokenizer else "dsa_tiny_utf8_v1",
        "tokenizer_vocab_sha256": stable_json_hash(tokenizer.get_vocab()),
        "loop_budget": args.loops,
        "truncation": False,
        "latent_answer_boundary": "fixed_inference_prefix_then_answer_tokens",
    }
    report = {
        "schema_version": "dsa_tokenization_report_v1", "record_count": len(tokens),
        "event_count": sum(record["event_count"] for record in records),
        "no_event_count": sum(record["event_count"] == 0 for record in records),
        "controller_supervised_count": sum(record["controller_supervised"] for record in tokens),
        "controller_masked_count": sum(reasons.values()),
        "controller_mask_reasons": dict(sorted(reasons.items())),
        "controller_label_overflow": reasons["event_overflow"],
        "tokenizer_mapping_failures": 0,
        "max_explicit_length": max(len(record["explicit_input_ids"]) for record in tokens),
        "max_question_length": max(len(record["question_input_ids"]) for record in tokens),
        "loop_budget": args.loops, "truncation": False,
        "input_sha256": identity["input_sha256"],
        "annotation_policy_sha256": identity["annotation_policy_sha256"],
        "tokenizer_vocab_sha256": identity["tokenizer_vocab_sha256"],
    }
    report["no_event_rate"] = report["no_event_count"] / len(tokens)
    staged_files = []
    tokenizer_stage = None
    tokenizer_backup = None
    try:
        # All lines and both training sequences are validated before staging.
        tokenizer_dir.parent.mkdir(parents=True, exist_ok=True)
        tokenizer_stage = Path(tempfile.mkdtemp(prefix=f".{tokenizer_dir.name}.", dir=tokenizer_dir.parent))
        tokenizer.save_pretrained(tokenizer_stage)
        saved_hashes = {str(path.relative_to(tokenizer_stage)): hashlib.sha256(path.read_bytes()).hexdigest()
                        for path in sorted(tokenizer_stage.rglob("*")) if path.is_file()}
        identity["tokenizer_file_sha256s"] = saved_hashes
        identity["tokenizer_config_sha256"] = stable_json_hash(saved_hashes)
        identity["cache_identity_sha256"] = stable_json_hash(identity)
        (tokenizer_stage / "tokenizer_identity.json").write_text(_json(identity), encoding="utf-8")
        report["tokenizer_config_sha256"] = identity["tokenizer_config_sha256"]
        report["cache_identity_sha256"] = identity["cache_identity_sha256"]
        staged_files.append((_stage(args.output, output_text), args.output))
        staged_files.append((_stage(args.report, _json(report)), args.report))
        if tokenizer_dir.exists():
            tokenizer_backup = Path(tempfile.mkdtemp(prefix=f".{tokenizer_dir.name}.backup.", dir=tokenizer_dir.parent))
            tokenizer_backup.rmdir()
            os.replace(tokenizer_dir, tokenizer_backup)
        try:
            os.replace(tokenizer_stage, tokenizer_dir)
        except OSError:
            if tokenizer_backup is not None:
                os.replace(tokenizer_backup, tokenizer_dir)
                tokenizer_backup = None
            raise
        for temporary, destination in staged_files:
            os.replace(temporary, destination)
    except OSError as exc:
        parser.error(f"could not write tokenization artifacts: {exc}")
    finally:
        for temporary, _ in staged_files:
            temporary.unlink(missing_ok=True)
        if tokenizer_stage is not None and tokenizer_stage.exists():
            shutil.rmtree(tokenizer_stage)
        if tokenizer_backup is not None and tokenizer_backup.exists():
            shutil.rmtree(tokenizer_backup)
    print(json.dumps({"record_count": report["record_count"], "controller_label_overflow": report["controller_label_overflow"],
                      "controller_masked_count": report["controller_masked_count"], "max_explicit_length": report["max_explicit_length"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
