#!/usr/bin/env python3
"""Describe the unchanged supplied LOTUS corpus, including blocking defects.

This audit is descriptive: it never partitions, drops, repairs, annotates, or
trains on records. The primary preparation pipeline must still accept the split
manifest before a benchmark comparison may proceed.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import sys
from collections import Counter, defaultdict
from decimal import Decimal, InvalidOperation
from itertools import combinations
from pathlib import Path

REPOSITORY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY))

from dsa.data.adapters import canonical_record, stable_json_hash
from dsa.provenance import DATA_SOURCE_COMMIT, LFS_PREFIX, SPLITS, file_sha256

CONTROL_TOKENS = ("<SA>", "<|start-latent|>", "<|end-latent|>", "<|latent|>")
EXAMPLE_LIMIT = 10


def answer_key(value):
    """Mirror LOTUS's reference-answer normalization without interpreting text."""
    return value.replace(",", "").strip()


def numeric_key(value):
    try:
        number = Decimal(answer_key(value))
        return ("0" if number.is_zero() else str(number.normalize())) if number.is_finite() else None
    except InvalidOperation:
        return None


def _histogram(values):
    return {str(key): count for key, count in sorted(Counter(values).items())}


def _load_json(path):
    if path.read_bytes()[:len(LFS_PREFIX)].startswith(LFS_PREFIX):
        raise ValueError(f"{path}: Git LFS pointer is not corpus data")
    rows = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        raise ValueError(f"{path}: expected JSON array")
    return rows


def inspect_rows(rows, *, name="split"):
    """Return summary plus lightweight row indices for independent comparisons."""
    by_question = defaultdict(list)
    by_group = defaultdict(set)
    raw_hash_counts = Counter()
    sample_id_counts = Counter()
    rationale_hashes = defaultdict(set)
    normalized_answers = defaultdict(set)
    numeric_answers = defaultdict(set)
    invalid = []
    nonnumeric_rows = []
    tokens = {token: {"rows": 0, "occurrences": 0, "fields": Counter(), "examples": []}
              for token in CONTROL_TOKENS}
    entries = []
    nonstandard_keys = Counter()
    empty_fields = Counter()
    steps_counts = []
    explicit_problem_id_rows = 0
    for index, row in enumerate(rows):
        row_number = index + 1
        try:
            if not isinstance(row, dict):
                raise ValueError("row must be an object")
            if not isinstance(row.get("question"), str) or not isinstance(row.get("answer"), str):
                raise ValueError("question and answer must be strings")
            if not isinstance(row.get("steps"), list) or not all(isinstance(step, str) for step in row["steps"]):
                raise ValueError("steps must be a list of strings")
            record = canonical_record(row, steps_join="space")
        except (TypeError, ValueError) as exc:
            invalid.append({"row_number": row_number, "error": str(exc)})
            continue
        qhash = record["question_hash"]
        group = record["source_problem_id"]
        sample = record["sample_id"]
        answer = answer_key(record["final_answer"])
        by_question[qhash].append(row_number)
        by_group[group].add(qhash)
        rationale_hashes[qhash].add(hashlib.sha256(record["rationale_raw"].encode("utf-8")).hexdigest())
        normalized_answers[qhash].add(answer)
        numeric = numeric_key(record["final_answer"])
        numeric_answers[qhash].add(numeric if numeric is not None else f"nonnumeric:{answer}")
        if numeric is None:
            nonnumeric_rows.append(row_number)
        raw_hash_counts[record["original_sha256"]] += 1
        sample_id_counts[sample] += 1
        explicit_problem_id_rows += any(row.get(key) is not None for key in ("source_problem_id", "problem_id", "id"))
        nonstandard_keys.update(set(row) - {"question", "steps", "answer"})
        steps_counts.append(len(row["steps"]))
        fields = {"question": record["question"], "rationale": record["rationale_raw"],
                  "answer": record["final_answer"]}
        for field, value in fields.items():
            empty_fields[field] += not value.strip()
        for token in CONTROL_TOKENS:
            occurrences = {field: value.count(token) for field, value in fields.items()}
            if not sum(occurrences.values()):
                continue
            detail = tokens[token]
            detail["rows"] += 1
            detail["occurrences"] += sum(occurrences.values())
            detail["fields"].update({field: count for field, count in occurrences.items() if count})
            if len(detail["examples"]) < EXAMPLE_LIMIT:
                detail["examples"].append({"row_number": row_number, "question_hash": qhash})
        entries.append({"row_number": row_number, "question_hash": qhash,
                        "source_problem_id": group, "sample_id": sample, "answer_normalized": answer})
    conflicts = sorted(key for key, values in normalized_answers.items() if len(values) > 1)
    numeric_conflicts = sorted(key for key, values in numeric_answers.items() if len(values) > 1)
    duplicate_questions = {key: value for key, value in by_question.items() if len(value) > 1}
    summary = {
        "split": name, "rows": len(rows), "valid_rows": len(entries),
        "malformed_rows": len(invalid), "malformed_examples": invalid[:EXAMPLE_LIMIT],
        "unique_normalized_questions": len(by_question), "unique_problem_groups": len(by_group),
        "explicit_problem_id_rows": explicit_problem_id_rows,
        "question_hash_fallback_rows": len(entries) - explicit_problem_id_rows,
        "unique_sample_ids": len(sample_id_counts),
        "repeated_sample_id_rows": sum(count - 1 for count in sample_id_counts.values()),
        "sample_ids_repeated": sum(count > 1 for count in sample_id_counts.values()),
        "unique_original_records": len(raw_hash_counts),
        "repeated_exact_original_record_rows": sum(count - 1 for count in raw_hash_counts.values()),
        "original_record_hashes_repeated": sum(count > 1 for count in raw_hash_counts.values()),
        "questions_with_multiple_rows": len(duplicate_questions),
        "rows_per_question_histogram": _histogram(map(len, by_question.values())),
        "distinct_rationales_per_question_histogram": _histogram(map(len, rationale_hashes.values())),
        "questions_with_multiple_distinct_rationales": sum(len(values) > 1 for values in rationale_hashes.values()),
        "distinct_question_rationale_pairs": sum(map(len, rationale_hashes.values())),
        "questions_with_answer_normalization_conflicts": len(conflicts),
        "questions_with_numeric_answer_conflicts": len(numeric_conflicts),
        "answer_conflict_examples": [{"question_hash": key, "row_numbers": by_question[key],
                                      "answers_normalized": sorted(normalized_answers[key])}
                                     for key in conflicts[:EXAMPLE_LIMIT]],
        "groups_with_multiple_normalized_questions": sum(len(values) > 1 for values in by_group.values()),
        "nonnumeric_answer_rows": len(nonnumeric_rows),
        "nonnumeric_answer_row_examples": nonnumeric_rows[:EXAMPLE_LIMIT],
        "empty_field_rows": dict(empty_fields),
        "additional_key_rows": dict(sorted(nonstandard_keys.items())),
        "steps_fragment_count_histogram": _histogram(steps_counts),
        "control_token_collisions": tokens,
        "duplicate_question_examples": [{"question_hash": key, "row_numbers": by_question[key],
                                         "distinct_rationales": len(rationale_hashes[key])}
                                        for key in sorted(duplicate_questions)[:EXAMPLE_LIMIT]],
    }
    return summary, entries


def _official_entries(path):
    entries = []
    for line_number, line in enumerate(io.StringIO(path.read_text(encoding="utf-8")), 1):
        if not line.strip():
            continue
        try:
            row = canonical_record(json.loads(line))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{path}: line {line_number}: {exc}") from exc
        entries.append({"row_number": line_number, "question_hash": row["question_hash"],
                        "answer_normalized": answer_key(row["final_answer"])})
    return entries


def compare_membership(supplied, official):
    left = defaultdict(set)
    right = defaultdict(set)
    for entry in supplied:
        left[entry["question_hash"]].add(entry["answer_normalized"])
    for entry in official:
        right[entry["question_hash"]].add(entry["answer_normalized"])
    shared = left.keys() & right.keys()
    mismatches = sorted(key for key in shared if left[key] != right[key])
    supplied_only = sorted(left.keys() - right.keys())
    official_only = sorted(right.keys() - left.keys())
    ordered_supplied = [(entry["question_hash"], entry["answer_normalized"]) for entry in supplied]
    ordered_official = [(entry["question_hash"], entry["answer_normalized"]) for entry in official]
    return {
        "supplied_rows": len(supplied), "official_rows": len(official),
        "supplied_unique_questions": len(left), "official_unique_questions": len(right),
        "matched_unique_questions": len(shared),
        "all_supplied_questions_in_official": not supplied_only,
        "question_membership_match": not supplied_only and not official_only,
        "ordered_question_answer_match": ordered_supplied == ordered_official,
        "supplied_only_question_hashes": supplied_only,
        "official_only_question_hashes": official_only,
        "answer_mismatch_question_hashes": mismatches,
        "answer_mismatch_examples": [{"question_hash": key, "supplied": sorted(left[key]),
                                     "official": sorted(right[key])} for key in mismatches[:EXAMPLE_LIMIT]],
        "ordered_question_answer_sha256": stable_json_hash(ordered_supplied),
        "official_ordered_question_answer_sha256": stable_json_hash(ordered_official),
    }


def audit_corpus(split_paths, *, official_train, official_test):
    summaries, indices = {}, {}
    hashes_before = {split: file_sha256(path) for split, path in split_paths.items()}
    for split in SPLITS:
        summary, entries = inspect_rows(_load_json(Path(split_paths[split])), name=split)
        summary.update({"path": str(Path(split_paths[split]).resolve()), "sha256": hashes_before[split]})
        summaries[split], indices[split] = summary, entries
    overlaps = []
    for left, right in combinations(SPLITS, 2):
        by_question = {name: defaultdict(list) for name in (left, right)}
        by_group = {name: defaultdict(list) for name in (left, right)}
        for name in (left, right):
            for entry in indices[name]:
                by_question[name][entry["question_hash"]].append(entry["row_number"])
                by_group[name][entry["source_problem_id"]].append(entry["row_number"])
        shared = sorted(by_question[left].keys() & by_question[right].keys())
        shared_groups = sorted(by_group[left].keys() & by_group[right].keys())
        overlaps.append({
            "splits": [left, right], "overlapping_questions": len(shared),
            "overlapping_problem_groups": len(shared_groups),
            "rows_from_overlapping_questions": {name: sum(len(by_question[name][key]) for key in shared)
                                                for name in (left, right)},
            "question_hashes": shared, "source_problem_ids": shared_groups,
            "examples": [{"question_hash": key, "row_numbers": {name: by_question[name][key]
                                                                  for name in (left, right)}}
                         for key in shared[:EXAMPLE_LIMIT]],
        })
    official = {"train": _official_entries(Path(official_train)), "test": _official_entries(Path(official_test))}
    comparisons = {
        "train_vs_official_train": compare_membership(indices["train"], official["train"]),
        "valid_vs_official_train": compare_membership(indices["valid"], official["train"]),
        "train_valid_union_vs_official_train": compare_membership(indices["train"] + indices["valid"], official["train"]),
        "test_vs_official_test": compare_membership(indices["test"], official["test"]),
    }
    reasons = []
    if any(item["overlapping_questions"] or item["overlapping_problem_groups"] for item in overlaps):
        reasons.append("supplied cross-partition problem overlap: stop primary comparison; do not silently resplit")
    if any(info["malformed_rows"] for info in summaries.values()):
        reasons.append("malformed original records require source discrepancy resolution")
    if any(info["control_token_collisions"][token]["rows"] for info in summaries.values() for token in CONTROL_TOKENS):
        reasons.append("literal reserved control token in original source")
    if any(info["questions_with_numeric_answer_conflicts"] for info in summaries.values()):
        reasons.append("conflicting answers for a normalized original problem")
    hashes_after = {split: file_sha256(path) for split, path in split_paths.items()}
    if hashes_before != hashes_after:
        raise ValueError("source JSON bytes changed during audit")
    return {
        "schema_version": "dsa_lotus_membership_audit_v1", "regime": "lotus_mirror",
        "data_source_commit": DATA_SOURCE_COMMIT,
        "script_sha256": file_sha256(Path(__file__)),
        "normalization": {"question": "NFKC, casefold, collapse whitespace",
                          "answer": "remove commas, strip whitespace (pinned LOTUS reference format)",
                          "numeric_diagnostic": "finite Decimal value, for distinguishing formatting-only conflicts",
                          "rationale": "join original steps with one space; no other rewrite"},
        "limitations": ["Normalized question hashes detect formatting duplicates, not arbitrary semantic paraphrases.",
                        "Rows are source occurrences; identical rows retain identical canonical sample IDs and remain in all denominators.",
                        "Official corpus comparison is provenance validation, not test-driven model selection."],
        "splits": summaries, "split_overlap_audit": {"passed": not any(item["overlapping_questions"] or item["overlapping_problem_groups"] for item in overlaps),
                                                      "pairs": overlaps},
        "official_sources": {name: {"path": str(Path(path).resolve()), "sha256": file_sha256(path)}
                             for name, path in (("train", official_train), ("test", official_test))},
        "official_membership_comparisons": comparisons,
        "primary_comparison_eligibility": {"eligible": not reasons, "stop_reasons": reasons},
        "source_files_unchanged": True,
        "evaluation_denominator": {"supplied_test_rows": summaries["test"]["rows"],
                                   "supplied_test_unique_questions": summaries["test"]["unique_normalized_questions"],
                                   "annotation_filtering_applied": False},
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for split in SPLITS:
        parser.add_argument(f"--{split}", type=Path, default=REPOSITORY / f"data/gsm_{split}.json")
    official_root = REPOSITORY.parent / "dsa-sources/grade-school-math/grade_school_math/data"
    parser.add_argument("--official-train", type=Path, default=official_root / "train.jsonl")
    parser.add_argument("--official-test", type=Path, default=official_root / "test.jsonl")
    parser.add_argument("--output", type=Path, default=REPOSITORY / "provenance/lotus_membership_audit.json")
    args = parser.parse_args(argv)
    report = audit_corpus({split: getattr(args, split) for split in SPLITS},
                          official_train=args.official_train, official_test=args.official_test)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"report": str(args.output.resolve()),
                      "splits": {split: {key: report["splits"][split][key]
                                         for key in ("rows", "unique_normalized_questions", "repeated_exact_original_record_rows")}
                                 for split in SPLITS},
                      "primary_comparison_eligibility": report["primary_comparison_eligibility"]}, indent=2))
    return 0 if report["primary_comparison_eligibility"]["eligible"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
