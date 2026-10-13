#!/usr/bin/env python3
"""Descriptively audit local LOTUS JSON arrays without writing training data.

The held-out test split is inspected for format and literal control-token
collisions only. Accepted milestone mapping is measured on train and valid.
Every source record is counted, including no-event, overflow and failed rows.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import sys
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dsa.data.adapters import canonical_record, stable_json_hash
from dsa.data.milestones import annotate_record, map_events_to_loops
from dsa.data.tokenize import register_marker_token, tokenize_record


LOOP_BUDGETS = (4, 8, 16, 32)


class FrozenVocabularyTokenizer:
    """Audit-only view: snapshot vocabulary after final marker registration.

    HF get_vocab constructs a 50k-entry dictionary on every invocation. The
    production record mapper checks that vocabulary per record; reusing its
    immutable snapshot here preserves that check without rebuilding it 770k
    times. Encoding/offset/decode calls still use the original tokenizer.
    """
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.vocabulary = tokenizer.get_vocab()

    def get_vocab(self):
        return self.vocabulary

    def __getattr__(self, name):
        return getattr(self.tokenizer, name)

    def __call__(self, *args, **kwargs):
        return self.tokenizer(*args, **kwargs)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def length_summary(values: Counter[int], context_limit: int) -> dict[str, Any]:
    """Use exact nearest-rank quantiles from a bounded length histogram."""
    count = sum(values.values())
    if not count:
        return {"record_count": 0, "max": None, "quantiles": {}, "over_context": 0}
    ordered = sorted(values.items())
    quantiles = {}
    for quantile in (0.5, 0.9, 0.95, 0.99, 1.0):
        target, cumulative = max(1, math.ceil(quantile * count)), 0
        for value, frequency in ordered:
            cumulative += frequency
            if cumulative >= target:
                quantiles[str(quantile)] = value
                break
    return {"record_count": count, "max": max(values), "quantiles": quantiles,
            "over_context": sum(frequency for value, frequency in values.items() if value > context_limit),
            "at_context": values[context_limit], "context_limit": context_limit,
            "quantile_method": "nearest_rank"}


def audit_records(records: list[dict[str, Any]], tokenizer: Any, *, split: str,
                  context_limit: int = 1024, modes: tuple[str, ...] = ("conclusive", "verified_only"),
                  sample_limit: int = 3, progress: bool = False) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Audit every row while retaining only bounded, deterministic examples."""
    tokenizer = FrozenVocabularyTokenizer(tokenizer)
    held_out = split == "test"
    controls = sorted(set(tokenizer.all_special_tokens) | {"<SA>"})
    counts: Counter[str] = Counter()
    occurrences: Counter[str] = Counter()
    collision_records: Counter[str] = Counter()
    format_reasons: Counter[str] = Counter()
    policy_stats = {}
    examples = []
    example_counts: Counter[str] = Counter()

    def sample(category: str, index: int, item: dict[str, Any]) -> None:
        if not held_out and example_counts[category] < sample_limit:
            examples.append({"split": split, "row_index": index, "category": category, **item})
            example_counts[category] += 1

    for mode in (() if held_out else modes):
        policy_stats[mode] = {
            "counts": Counter(), "failures": Counter(), "event_count_histogram": Counter(),
            "lengths": {name: Counter() for name in ("explicit_including_eos", "question_including_newline",
                                                     "raw_rationale", "latent_suffix_including_eos",
                                                     "question_plus_latent_suffix_excluding_memory")},
            "loop_budgets": {budget: Counter() for budget in LOOP_BUDGETS},
        }

    for index, source in enumerate(records):
        counts["source_records"] += 1
        try:
            canonical = canonical_record(source, steps_join="space")
        except (TypeError, ValueError, KeyError) as exc:
            counts["format_failures"] += 1
            format_reasons[f"{type(exc).__name__}: {exc}"] += 1
            sample("format_failure", index, {"error": str(exc)})
            continue
        counts["format_successes"] += 1
        has_collision = False
        for control in controls:
            token_count = sum(canonical[field].count(control) for field in ("question", "rationale_raw", "final_answer"))
            if token_count:
                has_collision = True
                occurrences[control] += token_count
                collision_records[control] += 1
        counts["records_with_literal_control_tokens"] += has_collision
        # Only policy can differ between these frozen annotations. Identical
        # marked text/event endpoints yield exactly the same tokenizer inputs;
        # reuse that per-row mapping while retaining separate policy counters.
        tokenization_cache = {}
        for mode, stats in policy_stats.items():
            try:
                annotated = annotate_record(source, mode=mode, exclude_terminal=True, steps_join="space")
            except (TypeError, ValueError, KeyError, OverflowError) as exc:
                stats["counts"]["annotation_failures"] += 1
                stats["failures"][f"annotation/{type(exc).__name__}: {exc}"] += 1
                sample(f"{mode}/annotation_failure", index, {"error": str(exc), "sample_id": canonical["sample_id"]})
                continue
            stats["counts"]["annotation_successes"] += 1
            stats["counts"]["accepted_events"] += annotated["event_count"]
            stats["event_count_histogram"][annotated["event_count"]] += 1
            mapping_key = (annotated["rationale_marked"], tuple(event["end"] for event in annotated["events"]))
            try:
                # Sequence and offset mapping are independent of loop budget.
                # Tokenize exactly once, then call the same production scheduler.
                if mapping_key not in tokenization_cache:
                    tokenized = tokenize_record(annotated, tokenizer, max(LOOP_BUDGETS))
                    text = annotated["question"] + "\n" + annotated["rationale_marked"] + "\n### " + annotated["final_answer"]
                    restored = tokenizer.decode(tokenized["explicit_input_ids"][:-1], skip_special_tokens=False,
                                                clean_up_tokenization_spaces=False)
                    raw_offsets = tokenizer(annotated["rationale_raw"], add_special_tokens=False,
                                            return_offsets_mapping=True)["offset_mapping"]
                    tokenization_cache[mapping_key] = (tokenized, text, restored, raw_offsets)
                tokenized, text, restored, raw_offsets = tokenization_cache[mapping_key]
            except (TypeError, ValueError, KeyError, OverflowError) as exc:
                stats["counts"]["tokenizer_mapping_failures"] += 1
                stats["failures"][f"tokenize/{type(exc).__name__}: {exc}"] += 1
                sample(f"{mode}/tokenizer_failure", index, {"error": str(exc), "sample_id": canonical["sample_id"]})
                continue
            stats["counts"]["tokenized_records"] += 1
            stats["counts"]["mapped_events"] += len(tokenized["raw_event_token_ends"])
            if restored != text:
                stats["counts"]["explicit_roundtrip_failures"] += 1
                sample(f"{mode}/roundtrip_failure", index, {"sample_id": canonical["sample_id"]})
            for event, token_end in zip(annotated["events"], tokenized["raw_event_token_ends"]):
                start, end = raw_offsets[token_end - 1]
                stats["counts"]["events_ending_inside_bpe_token"] += end != event["end"]
                stats["counts"]["events_mapped_to_exact_token_end"] += end == event["end"]
                if not start < event["end"] <= end:
                    raise AssertionError("production mapping returned an uncovered event boundary")
            stats["counts"]["events_sharing_raw_token_end"] += len(tokenized["raw_event_token_ends"]) - len(set(tokenized["raw_event_token_ends"]))
            lengths = {
                "explicit_including_eos": len(tokenized["explicit_input_ids"]),
                "question_including_newline": len(tokenized["question_input_ids"]),
                "raw_rationale": tokenized["raw_rationale_token_length"],
                "latent_suffix_including_eos": len(tokenized["latent_suffix_input_ids"]),
                "question_plus_latent_suffix_excluding_memory": len(tokenized["question_input_ids"]) + len(tokenized["latent_suffix_input_ids"]),
            }
            for key, value in lengths.items():
                stats["lengths"][key][value] += 1
            for budget in LOOP_BUDGETS:
                schedule = map_events_to_loops(tokenized["raw_event_token_ends"], tokenized["raw_rationale_token_length"], budget)
                stats["loop_budgets"][budget]["controller_supervised" if schedule["supervised"] else schedule["reason"]] += 1
                if budget == max(LOOP_BUDGETS):
                    for key in ("gate_targets", "event_loops"):
                        if tokenized[key] != schedule[key]:
                            raise AssertionError("production tokenizer and scheduler disagree")
            category = "no_events" if not annotated["events"] else "event_overflow_at_4" if annotated["event_count"] > 4 else "mapped"
            sample(f"{mode}/{category}", index, {
                "sample_id": canonical["sample_id"], "question_hash": canonical["question_hash"],
                "event_count": annotated["event_count"], "event_char_ends": [event["end"] for event in annotated["events"]],
                "raw_event_token_ends": tokenized["raw_event_token_ends"], "lengths": lengths,
                "tokenized_record_sha256": stable_json_hash(tokenized),
            })
        if progress and (index + 1) % 10000 == 0:
            print(f"{split}: inspected {index + 1} source records", file=sys.stderr, flush=True)

    result = {
        "scope": "held_out_membership_format_and_literal_controls_only" if held_out else "all_augmented_traces_descriptive_mapping",
        "record_count": counts["source_records"], "format_successes": counts["format_successes"],
        "format_failures": counts["format_failures"], "format_failure_reasons": dict(sorted(format_reasons.items())),
        "records_with_literal_control_tokens": counts["records_with_literal_control_tokens"],
        "literal_control_token_occurrences": dict(sorted(occurrences.items())),
        "literal_control_token_record_counts": dict(sorted(collision_records.items())), "policies": {},
    }
    for mode, stats in policy_stats.items():
        result["policies"][mode] = {
            **{key: stats["counts"][key] for key in (
                "annotation_successes", "annotation_failures", "accepted_events", "tokenized_records", "mapped_events",
                "tokenizer_mapping_failures", "explicit_roundtrip_failures", "events_ending_inside_bpe_token",
                "events_mapped_to_exact_token_end", "events_sharing_raw_token_end")},
            "failure_reasons": dict(sorted(stats["failures"].items())),
            "event_count_histogram": {str(key): value for key, value in sorted(stats["event_count_histogram"].items())},
            "lengths": {key: length_summary(values, context_limit) for key, values in stats["lengths"].items()},
            "loop_budgets": {str(budget): {key: histogram[key] for key in ("controller_supervised", "no_events", "event_overflow")}
                             for budget, histogram in stats["loop_budgets"].items()},
        }
    return result, examples


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", required=True, type=Path)
    parser.add_argument("--tokenizer", required=True, type=Path)
    parser.add_argument("--source-metadata", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--samples", required=True, type=Path)
    args = parser.parse_args(argv)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(args.tokenizer), use_fast=True, local_files_only=True)
    original_vocab_size = len(tokenizer)
    original_controls = list(tokenizer.all_special_tokens)
    marker_id = register_marker_token(tokenizer)
    config = json.loads((args.tokenizer / "config.json").read_text())
    context_limit = config["n_positions"]
    source_metadata = json.loads(args.source_metadata.read_text())
    for name, identity in source_metadata["files"].items():
        if file_sha256(args.tokenizer / name) != identity["sha256"]:
            raise ValueError(f"source tokenizer hash mismatch: {name}")
    report = {
        "schema_version": "dsa_lotus_tokenizer_audit_v1",
        "scope": "descriptive audit; no training data emitted and no benchmark selection performed",
        "training_approval": "not_granted_by_this_audit; consult independent split audit",
        "no_filtering": True, "truncation": False, "test_used_for_annotation_or_selection": False,
        "tokenizer_source": source_metadata,
        "tokenizer": {"class": type(tokenizer).__name__, "is_fast": tokenizer.is_fast,
                      "original_vocab_size": original_vocab_size, "registered_vocab_size": len(tokenizer),
                      "marker_token": "<SA>", "marker_token_id": marker_id,
                      "original_special_tokens": original_controls, "eos_token_id": tokenizer.eos_token_id,
                      "pad_token_id": tokenizer.pad_token_id, "context_limit": context_limit,
                      "registered_vocab_sha256": stable_json_hash(tokenizer.get_vocab())},
        "runtime": {name: importlib.metadata.version(name) for name in ("transformers", "tokenizers")},
        "implementation_file_sha256s": {name: file_sha256(Path(__file__).resolve().parents[1] / name)
                                       for name in ("dsa/data/adapters.py", "dsa/data/milestones.py", "dsa/data/tokenize.py", "scripts/audit_lotus_tokenizer.py")},
        "limitations": ["Mapping success is not a measurement of milestone semantic precision.",
                        "No-event and event-overflow rows are retained in all denominators; their controller and anchor supervision is masked.",
                        "Question-plus-suffix lengths omit anchor/workspace slots; downstream context checks must include configured memory and generation allowance.",
                        "The held-out test split receives no annotation, tokenization, examples, or parameter selection.",
                        "No model weights were downloaded or validated; tokenizer pinning is not a checkpoint smoke test.",
                        "Canonical steps_join=space follows the upstream training text convention; no parser changes were made."],
        "splits": {},
    }
    samples = []
    for split in ("train", "valid", "test"):
        path = args.input_dir / f"gsm_{split}.json"
        with path.open(encoding="utf-8") as handle:
            records = json.load(handle)
        if not isinstance(records, list):
            raise ValueError(f"{path}: expected JSON array")
        result, split_samples = audit_records(records, tokenizer, split=split, context_limit=context_limit, progress=True)
        result["input_file"] = str(path)
        result["input_sha256"] = file_sha256(path)
        result["input_bytes"] = path.stat().st_size
        report["splits"][split] = result
        samples.extend(split_samples)
        del records
        print(f"{split}: audit complete ({result['record_count']} records)", file=sys.stderr, flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.samples.parent.mkdir(parents=True, exist_ok=True)
    sample_text = "".join(json.dumps(sample, sort_keys=True) + "\n" for sample in samples)
    args.samples.write_text(sample_text, encoding="utf-8")
    report["deterministic_samples"] = {"file": str(args.samples), "sha256": hashlib.sha256(sample_text.encode()).hexdigest(),
                                       "count": len(samples), "selection": "first three per train/valid mode and category in source order; test excluded"}
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"report": str(args.output), "record_counts": {key: value["record_count"] for key, value in report["splits"].items()}}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
