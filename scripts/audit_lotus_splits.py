#!/usr/bin/env python3
"""Persist the complete source split manifest and a compact reviewable summary."""
from pathlib import Path
import argparse
import json
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dsa.provenance import build_split_manifest, SplitOverlapError, file_sha256, SPLITS


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full-output", type=Path, default=ROOT / "artifacts/lotus-corpus-audit/split_manifest.json")
    parser.add_argument("--summary-output", type=Path, default=ROOT / "provenance/lotus_split_manifest.json")
    parser.add_argument("--official-test", type=Path,
                        default=ROOT.parent / "dsa-sources/grade-school-math/grade_school_math/data/test.jsonl")
    args = parser.parse_args()
    try:
        report = build_split_manifest(
            {split: ROOT / "data" / f"gsm_{split}.json" for split in SPLITS},
            lotus_sha=subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip(),
            preprocessing_paths={name: ROOT / "preprocessing" / name for name in ("gsm_icot.bash", "gsm_icot.py")},
            official_test_path=args.official_test,
        )
    except SplitOverlapError as exc:
        failure = {"schema_version": "dsa_split_audit_failure_v1", "split_overlap_audit": {"status": "failed", "overlaps": exc.overlaps},
                   "comparison_stopped": True, "error": str(exc)}
        args.summary_output.parent.mkdir(parents=True, exist_ok=True)
        args.summary_output.write_text(json.dumps(failure, indent=2) + "\n")
        parser.exit(2, f"split audit stopped: {exc}\n")
    if args.full_output.resolve() == args.summary_output.resolve():
        parser.error("full manifest and summary must be different files")
    args.full_output.parent.mkdir(parents=True, exist_ok=True)
    args.full_output.write_text(json.dumps(report, indent=2) + "\n")
    summary = {key: value for key, value in report.items()
               if key not in ("splits", "ordered_evaluation_records", "ordered_evaluation_ids")}
    summary["schema_version"] = "dsa_split_manifest_summary_v1"
    summary["full_manifest"] = {"path": str(args.full_output.resolve()), "sha256": file_sha256(args.full_output),
                                "contains": ["full problem groups and sample IDs", "ordered evaluation IDs and records"]}
    summary["splits"] = {split: {key: value for key, value in info.items() if key != "problem_groups"}
                         for split, info in report["splits"].items()}
    summary["ordered_evaluation_id_count"] = len(report["ordered_evaluation_ids"])
    args.summary_output.parent.mkdir(parents=True, exist_ok=True)
    args.summary_output.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({"full_manifest": str(args.full_output), "summary": str(args.summary_output),
                      "overlap_status": "passed", "official_test_ordered_match": report["official_test_comparison"]["ordered_question_answer_match"]}))


if __name__ == "__main__":
    main()
