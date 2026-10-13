#!/usr/bin/env python3
"""Stream the frozen LOTUS splits through both existing milestone policies.

This audit measures the parser already in dsa.data.milestones; it never changes
rules in response to validation/test contents. A passed, byte-matching split
manifest is required before producing annotated derivatives or review samples.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Iterator

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dsa.data.adapters import stable_json_hash
from dsa.data.milestones import SCHEMA_VERSION, annotate_record
from dsa.provenance import SPLITS, file_sha256

MODES = ("conclusive", "verified_only")
LOOP_BUDGETS = (4, 8, 16, 32)
RULES = {"R1": "calculator_equation", "R2": "bare_numeric_equation", "R3": "isolated_variable"}


def iter_json_array(path: Path, chunk_chars: int = 1024 * 1024) -> Iterator[dict[str, Any]]:
    """Decode a top-level JSON array incrementally, retaining only one row."""
    def reject_constant(value: str) -> None:
        raise ValueError(f"nonstandard JSON constant: {value}")

    decoder = json.JSONDecoder(parse_constant=reject_constant)
    with path.open(encoding="utf-8") as handle:
        buffer, cursor, eof = "", 0, False

        def refill() -> None:
            nonlocal buffer, cursor, eof
            block = handle.read(chunk_chars)
            buffer = buffer[cursor:] + block
            cursor = 0
            eof = not block

        def peek() -> str:
            nonlocal cursor
            while True:
                while cursor < len(buffer) and buffer[cursor].isspace():
                    cursor += 1
                if cursor < len(buffer):
                    return buffer[cursor]
                if eof:
                    return ""
                refill()

        if peek() != "[":
            raise ValueError(f"{path}: expected a JSON array")
        cursor += 1
        if peek() == "]":
            cursor += 1
        else:
            while True:
                peek()
                while True:
                    try:
                        value, end = decoder.raw_decode(buffer, cursor)
                        break
                    except json.JSONDecodeError as exc:
                        if eof:
                            raise ValueError(f"{path}: invalid JSON array: {exc}") from exc
                        refill()
                cursor = end
                if not isinstance(value, dict):
                    raise ValueError(f"{path}: every array entry must be an object")
                yield value
                separator = peek()
                if separator == "]":
                    cursor += 1
                    break
                if separator != ",":
                    raise ValueError(f"{path}: expected comma or closing bracket")
                cursor += 1
                if peek() == "]":
                    raise ValueError(f"{path}: trailing comma")
        if peek():
            raise ValueError(f"{path}: trailing text after JSON array")


def fraction(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


class Coverage:
    """Keep small aggregate counters and per-question eligibility bitmasks."""

    def __init__(self) -> None:
        self.rows = 0
        self.histogram: Counter[int] = Counter()
        self.rules: Counter[str] = Counter()
        self.evidence: Counter[str] = Counter()
        self.reasons: Counter[str] = Counter()
        self.rejections_per_row: Counter[int] = Counter()
        self.terminal_per_row: Counter[int] = Counter()
        self.questions: dict[str, dict[str, Any]] = {}
        self.integrity: Counter[str] = Counter()

    def add(self, source: dict[str, Any], record: dict[str, Any], serialized: str) -> None:
        m = record["event_count"]
        self.rows += 1
        self.histogram[m] += 1
        self.rules.update(event["rule"] for event in record["events"])
        self.evidence.update(event["validation"] for event in record["events"])
        self.reasons.update(event["reason"] for event in record["rejected_candidates"])
        self.rejections_per_row[len(record["rejected_candidates"])] += 1
        self.terminal_per_row[sum(event["reason"] == "terminal_answer_match"
                                  for event in record["rejected_candidates"])] += 1
        group = self.questions.setdefault(record["question_hash"], {"rows": 0, "zero_rows": 0,
                                                                   "eligible_any": set(), "eligible_all": set(LOOP_BUDGETS)})
        group["rows"] += 1
        group["zero_rows"] += m == 0
        eligible = {budget for budget in LOOP_BUDGETS if 0 < m <= budget}
        group["eligible_any"].update(eligible)
        group["eligible_all"].intersection_update(eligible)

        checks = {
            "original_record_equal": record["original_record"] == source,
            "original_content_hash_equal": record["original_sha256"] == stable_json_hash(source),
            "question_unchanged": record["question"] == source["question"],
            "final_answer_unchanged": record["final_answer"] == source["answer"],
            "steps_space_join_equal": record["rationale_raw"] == " ".join(source["steps"]),
            "marker_removal_roundtrip": record["rationale_marked"].replace(record["policy"]["marker_token"], "")
                == record["rationale_raw"],
            "json_roundtrip": json.loads(serialized) == record,
            "event_count_equal": len(record["events"]) == m,
            "event_ordinals_equal": [event["ordinal"] for event in record["events"]] == list(range(m)),
            "event_ends_strict": all(a["end"] < b["end"] for a, b in zip(record["events"], record["events"][1:])),
        }
        for name, passed in checks.items():
            if not passed:
                raise ValueError(f"integrity check failed: {name} for {record['sample_id']}")
            self.integrity[name] += 1
        for collection in ("events", "rejected_candidates"):
            for event in record[collection]:
                if not (0 <= event["start"] < event["end"] <= len(record["rationale_raw"])):
                    raise ValueError(f"invalid {collection} span for {record['sample_id']}")
                if record["rationale_raw"][event["start"]:event["end"]] != event["text"]:
                    raise ValueError(f"nonmatching {collection} span for {record['sample_id']}")
                self.integrity[f"{collection}_span_checks"] += 1

    def report(self) -> dict[str, Any]:
        questions = len(self.questions)
        zero_all = sum(group["zero_rows"] == group["rows"] for group in self.questions.values())
        zero_any = sum(group["zero_rows"] > 0 for group in self.questions.values())
        controller = {}
        for budget in LOOP_BUDGETS:
            eligible = sum(count for m, count in self.histogram.items() if 0 < m <= budget)
            overflow = sum(count for m, count in self.histogram.items() if m > budget)
            controller[str(budget)] = {
                "eligible_rows": eligible, "eligible_row_rate": fraction(eligible, self.rows),
                "masked_M0_rows": self.histogram[0], "masked_MgtK_rows": overflow,
                "masked_MgtK_row_rate": fraction(overflow, self.rows),
                "questions_with_any_eligible_rationale": sum(budget in g["eligible_any"] for g in self.questions.values()),
                "questions_with_all_rationales_eligible": sum(budget in g["eligible_all"] for g in self.questions.values()),
            }
        return {
            "record_count": self.rows, "unique_questions": questions,
            "no_event_count": self.histogram[0], "no_event_rate": fraction(self.histogram[0], self.rows),
            "questions_with_no_events_in_any_rationale": zero_all,
            "questions_with_no_events_in_any_rationale_rate": fraction(zero_all, questions),
            "questions_with_at_least_one_no_event_rationale": zero_any,
            "questions_with_at_least_one_no_event_rationale_rate": fraction(zero_any, questions),
            "event_count": sum(m * count for m, count in self.histogram.items()),
            "event_count_histogram": {str(m): count for m, count in sorted(self.histogram.items())},
            "rule_counts": {rule: self.rules[rule] for rule in RULES.values()},
            "R1_R2_R3_counts": {key: self.rules[rule] for key, rule in RULES.items()},
            "evidence_counts": dict(sorted(self.evidence.items())),
            "terminal_exclusions": self.reasons["terminal_answer_match"],
            "terminal_exclusions_per_row_histogram": {str(k): v for k, v in sorted(self.terminal_per_row.items())},
            "rejection_reasons": dict(sorted(self.reasons.items())),
            "rejected_candidates_per_row_histogram": {str(k): v for k, v in sorted(self.rejections_per_row.items())},
            "controller_coverage": controller,
            "integrity_checks_passed": dict(sorted(self.integrity.items())),
        }


class ReviewPool:
    """Deterministic TRAIN-only strata, capped memory and unique question groups."""

    def __init__(self, seed: int, max_per_stratum: int = 48) -> None:
        self.seed, self.max_per_stratum = seed, max_per_stratum
        self.pools: dict[str, dict[str, tuple[str, dict[str, Any]]]] = {}

    def add(self, index: int, record: dict[str, Any]) -> None:
        labels = {f"accepted:{event['rule']}:{event['validation']}" for event in record["events"]}
        labels.update(f"rejected:{event['reason']}" for event in record["rejected_candidates"])
        if not record["events"]:
            labels.add("no_events:rejected_candidates" if record["rejected_candidates"] else "no_events:no_candidates")
        qhash = record["question_hash"]
        for label in labels:
            priority = stable_json_hash([self.seed, label, qhash])
            pool = self.pools.setdefault(label, {})
            previous = pool.get(qhash)
            if previous is not None and previous[1]["sample_id"] <= record["sample_id"]:
                continue
            if previous is None and len(pool) >= self.max_per_stratum:
                largest = max(pool, key=lambda key: pool[key][0])
                if pool[largest][0] <= priority:
                    continue
                del pool[largest]
            pool[qhash] = (priority, {"row_index": index, "sample_id": record["sample_id"],
                                    "question_hash": qhash, "original_record": record["original_record"]})

    def select(self, count: int = 32) -> list[dict[str, Any]]:
        ordered = {label: iter(sorted(pool.values(), key=lambda item: item[0]))
                   for label, pool in sorted(self.pools.items())}
        selected, seen = [], set()
        while ordered and len(selected) < count:
            for label in list(ordered):
                for _, item in ordered[label]:
                    if item["question_hash"] not in seen:
                        seen.add(item["question_hash"])
                        records = {mode: annotate_record(item["original_record"], mode=mode) for mode in MODES}
                        selected.append({"review_number": len(selected) + 1, "selection_stratum": label,
                                         **item, "annotations": records})
                        break
                else:
                    del ordered[label]
                if len(selected) >= count:
                    break
        return selected


def run_audit(split_paths: dict[str, Path], manifest_path: Path, output: Path,
              report_path: Path, *, seed: int = 7, sample_count: int = 32) -> dict[str, Any]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("split_overlap_audit", {}).get("status") != "passed":
        raise ValueError("a passed primary split-overlap audit is required")
    hashes = {split: file_sha256(path) for split, path in split_paths.items()}
    for split in SPLITS:
        if manifest["splits"][split]["original_file_sha256"] != hashes[split]:
            raise ValueError(f"{split}: source bytes differ from the passed split audit")
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"{output}: output must be empty")
    output.mkdir(parents=True, exist_ok=True)
    pool = ReviewPool(seed)
    policies: dict[str, dict[str, Any]] = {mode: {} for mode in MODES}
    artifacts = {}
    for split in SPLITS:
        counters = {mode: Coverage() for mode in MODES}
        handles = {mode: (output / f"{split}.{mode}.jsonl").open("w", encoding="utf-8", newline="\n")
                   for mode in MODES}
        original_path = output / f"{split}.original.jsonl"
        try:
            with original_path.open("w", encoding="utf-8", newline="\n") as originals:
                for index, source in enumerate(iter_json_array(split_paths[split])):
                    originals.write(json.dumps(source, ensure_ascii=False, allow_nan=False) + "\n")
                    for mode in MODES:
                        record = annotate_record(source, mode=mode, steps_join="space")
                        serialized = json.dumps(record, ensure_ascii=False, allow_nan=False)
                        counters[mode].add(source, record, serialized)
                        handles[mode].write(serialized + "\n")
                        if split == "train" and mode == "conclusive":
                            pool.add(index, record)
                    if (index + 1) % 10000 == 0:
                        print(json.dumps({"split": split, "rows_processed": index + 1}), flush=True)
        finally:
            for handle in handles.values():
                handle.close()
        for mode in MODES:
            policies[mode][split] = counters[mode].report()
            if counters[mode].rows != manifest["splits"][split]["rows"]:
                raise ValueError(f"{split}: row count changed since split audit")
            path = output / f"{split}.{mode}.jsonl"
            artifacts[path.name] = {"sha256": file_sha256(path), "bytes": path.stat().st_size}
        artifacts[original_path.name] = {"sha256": file_sha256(original_path), "bytes": original_path.stat().st_size}
        if file_sha256(split_paths[split]) != hashes[split]:
            raise ValueError(f"{split}: input changed during audit")
        print(json.dumps({"split": split, "completed_rows": counters[MODES[0]].rows}), flush=True)
    review = pool.select(sample_count)
    review_path = output / "train_review_sample.seed7.jsonl"
    review_path.write_text("".join(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n" for row in review), encoding="utf-8")
    artifacts[review_path.name] = {"sha256": file_sha256(review_path), "bytes": review_path.stat().st_size}
    report = {
        "schema_version": "dsa_lotus_annotation_audit_v1", "annotation_schema": SCHEMA_VERSION,
        "primary_split_manifest_path": str(manifest_path.resolve()),
        "primary_split_manifest_sha256": file_sha256(manifest_path),
        "input_sha256": hashes,
        "code_sha256": {name: file_sha256(ROOT / name) for name in
                        ("dsa/data/adapters.py", "dsa/data/milestones.py", "scripts/audit_lotus_annotations.py")},
        "policy": {"modes": list(MODES), "exclude_terminal": True, "steps_join": "space", "marker_token": "<SA>"},
        "rule_mapping": RULES,
        "controller_coverage_interpretation": "Count-only eligibility: M=0 and M>K mask BCE; token alignment is audited separately.",
        "policies": policies, "artifact_directory": str(output.resolve()), "artifacts": artifacts,
        "review_sample": {"path": str(review_path.resolve()), "seed": seed, "requested_count": sample_count,
                          "selected_count": len(review), "unique_question_groups": len({r["question_hash"] for r in review}),
                          "split": "train", "selection": "Round-robin across sorted observed conclusive-policy strata; SHA256([seed,stratum,question_hash]) ranks groups; minimum sample_id chooses rationale; one group globally.",
                          "strata_population_in_capped_pools": {label: len(items) for label, items in sorted(pool.pools.items())},
                          "review_status": "awaiting_agent_manual_review", "human_review_status": "not_performed"},
        "limitations": ["Local numeric consistency is not semantic correctness of a solution or milestone.",
                        "R3 is structural evidence only; verified_only filters it.",
                        "Terminal exclusion checks only the last accepted numeric value and can suppress a legitimate intermediate with the same value.",
                        "The TRAIN review is stratified and purposive; it does not estimate population semantic precision.",
                        "Validation/test contents are used only to measure frozen-rule coverage and integrity."],
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for split in SPLITS:
        parser.add_argument(f"--{split}", type=Path, default=ROOT / "data" / f"gsm_{split}.json")
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "artifacts/lotus-annotation-audit")
    parser.add_argument("--report", type=Path, default=ROOT / "provenance/lotus_annotation_audit.json")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--sample-count", type=int, default=32)
    args = parser.parse_args(argv)
    try:
        report = run_audit({split: getattr(args, split) for split in SPLITS}, args.split_manifest,
                           args.output_dir, args.report, seed=args.seed, sample_count=args.sample_count)
    except (OSError, ValueError, TypeError, KeyError) as exc:
        parser.exit(2, f"annotation audit stopped: {exc}\n")
    print(json.dumps({"report": str(args.report), "review_count": report["review_sample"]["selected_count"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
