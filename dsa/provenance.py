"""Audit supplied LOTUS splits before preparing any annotated derivatives.

This module does not download data or manufacture benchmark partitions. Its
hashes identify bytes on disk; normalized question hashes detect formatting
duplicates and cannot detect arbitrary paraphrases.
"""

from __future__ import annotations

import hashlib
import io
import json
import re
from collections import defaultdict
from collections.abc import Mapping
from itertools import combinations
from pathlib import Path
from typing import Any

from dsa.data.adapters import canonical_record, normalized_question_hash, stable_json_hash

DATA_SOURCE_COMMIT = "e06a32ee5e4cd117171daeb4755d2a97ece62761"
SPLITS = ("train", "valid", "test")
LFS_PREFIX = b"version https://git-lfs.github.com/spec/v1"


class SplitOverlapError(ValueError):
    """Supplied problem groups overlap; comparison must stop without resplitting."""

    def __init__(self, overlaps: list[dict[str, Any]]):
        self.overlaps = overlaps
        detail = "; ".join(
            f"{item['splits'][0]}/{item['splits'][1]}: "
            f"{len(item['source_problem_ids'])} groups, {len(item['question_hashes'])} questions"
            for item in overlaps
        )
        super().__init__(f"cross-partition overlap detected ({detail}); no derivatives written")


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_lotus_array(path: str | Path) -> list[dict[str, Any]]:
    """Load a JSON array without altering rows, rejecting pointers and bad shapes."""
    source = Path(path)
    with source.open("rb") as handle:
        if handle.read(len(LFS_PREFIX)).startswith(LFS_PREFIX):
            raise ValueError(f"{source}: Git LFS pointer is not training data")
    try:
        rows = json.loads(source.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{source}: invalid JSON array: {exc}") from exc
    if not isinstance(rows, list):
        raise ValueError(f"{source}: expected a JSON array")
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError(f"{source}: row {index + 1} must be an object")
        if not isinstance(row.get("question"), str) or not isinstance(row.get("answer"), str):
            raise ValueError(f"{source}: row {index + 1} requires string question and answer")
        if not isinstance(row.get("steps"), list) or not all(isinstance(s, str) for s in row["steps"]):
            raise ValueError(f"{source}: row {index + 1} requires a list of string steps")
        try:
            canonical_record(row, steps_join="space")
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{source}: row {index + 1}: {exc}") from exc
    return rows


def verify_pinned_download_script(path: str | Path, commit: str = DATA_SOURCE_COMMIT) -> dict[str, Any]:
    """Verify the exact three source URLs in the actual LOTUS download script.

    This is source-code verification only. HTTP status and payload SHA256 are
    separate provenance facts and are never inferred from URL syntax.
    """
    text = Path(path).read_text(encoding="utf-8")
    urls = re.findall(r"https://(?:media|raw)\.githubusercontent\.com/[^\s\"']+", text)
    expected = {
        "train": f"https://media.githubusercontent.com/media/da03/Internalize_CoT_Step_by_Step/{commit}/data/gsm8k/train.txt",
        "valid": f"https://raw.githubusercontent.com/da03/Internalize_CoT_Step_by_Step/{commit}/data/gsm8k/valid.txt",
        "test": f"https://raw.githubusercontent.com/da03/Internalize_CoT_Step_by_Step/{commit}/data/gsm8k/test.txt",
    }
    if len(urls) != 3 or set(urls) != set(expected.values()):
        raise ValueError("LOTUS download URLs differ from the declared pinned benchmark; resolve before preparing runs")
    return {"commit": commit, "urls": expected, "source_code_verified": True,
            "http_payload_verification": "not_performed_by_manifest_builder"}


def _answer_key(answer: str) -> str:
    # Match scripts/eval.py's supplied-answer formatting exactly.
    return answer.replace(",", "").strip()


def compare_official_test(rows: list[dict[str, Any]], official_path: str | Path) -> dict[str, Any]:
    """Compare supplied test membership and answers with a separately named raw file."""
    official: list[dict[str, str]] = []
    path = Path(official_path)
    # JSON strings may contain U+2028/U+2029; only actual LF delimits JSONL.
    for line_number, line in enumerate(io.StringIO(path.read_text(encoding="utf-8")), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
            source = canonical_record(row)
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            raise ValueError(f"{path}: line {line_number}: {exc}") from exc
        official.append({"question_hash": source["question_hash"], "answer": _answer_key(source["final_answer"])})
    supplied = [{"question_hash": normalized_question_hash(row["question"]), "answer": _answer_key(row["answer"])} for row in rows]
    def indexed(items: list[dict[str, str]]) -> dict[str, set[str]]:
        result: dict[str, set[str]] = defaultdict(set)
        for item in items:
            result[item["question_hash"]].add(item["answer"])
        return result
    lhs, rhs = indexed(supplied), indexed(official)
    common = lhs.keys() & rhs.keys()
    return {
        "status": "compared", "path": str(path.resolve()), "sha256": file_sha256(path),
        "official_rows": len(official), "supplied_rows": len(supplied),
        "ordered_question_answer_match": supplied == official,
        "question_membership_match": lhs.keys() == rhs.keys(),
        "supplied_only_question_hashes": sorted(lhs.keys() - rhs.keys()),
        "official_only_question_hashes": sorted(rhs.keys() - lhs.keys()),
        "answer_mismatch_question_hashes": sorted(key for key in common if lhs[key] != rhs[key]),
        "normalization": "question NFKC/casefold/whitespace; answer comma removal and strip",
    }


def build_split_manifest(
    split_paths: Mapping[str, str | Path], *, lotus_sha: str,
    preprocessing_paths: Mapping[str, str | Path], data_source_commit: str = DATA_SOURCE_COMMIT,
    official_test_path: str | Path | None = None,
) -> dict[str, Any]:
    """Audit all original arrays, preserving row order and all rationale variants."""
    if set(split_paths) != set(SPLITS):
        raise ValueError("provide exactly train, valid, and test source JSON arrays")
    if not re.fullmatch(r"[0-9a-f]{40}", lotus_sha) or not re.fullmatch(r"[0-9a-f]{40}", data_source_commit):
        raise ValueError("LOTUS and dataset revisions must be full 40-character commit SHAs")
    if set(preprocessing_paths) != {"gsm_icot.bash", "gsm_icot.py"}:
        raise ValueError("hash both actual LOTUS preprocessing scripts")
    download = verify_pinned_download_script(preprocessing_paths["gsm_icot.bash"], data_source_commit)
    sources = {split: load_lotus_array(split_paths[split]) for split in SPLITS}
    info: dict[str, Any] = {}
    groups: dict[str, set[str]] = {}
    questions: dict[str, set[str]] = {}
    ordered_evaluation: list[dict[str, Any]] = []
    for split in SPLITS:
        group_rows: dict[str, list[str]] = defaultdict(list)
        qhashes: set[str] = set()
        explicit_id_rows = 0
        for row_index, row in enumerate(sources[split]):
            record = canonical_record(row, steps_join="space")
            group_rows[record["source_problem_id"]].append(record["sample_id"])
            qhashes.add(record["question_hash"])
            explicit_id_rows += any(row.get(key) is not None for key in ("source_problem_id", "problem_id", "id"))
            if split == "test":
                ordered_evaluation.append({
                    "row_index": row_index, "sample_id": record["sample_id"],
                    "source_problem_id": record["source_problem_id"], "question_hash": record["question_hash"],
                    "question_input_sha256": hashlib.sha256((row["question"].strip() + "\n").encode("utf-8")).hexdigest(),
                    "answer_normalized": _answer_key(row["answer"]),
                })
        groups[split], questions[split] = set(group_rows), qhashes
        path = Path(split_paths[split])
        info[split] = {
            "path": str(path.resolve()), "original_file_sha256": file_sha256(path),
            "rows": len(sources[split]), "unique_questions": len(qhashes), "unique_problem_groups": len(group_rows),
            "explicit_problem_id_rows": explicit_id_rows, "question_hash_fallback_rows": len(sources[split]) - explicit_id_rows,
            "problem_groups": dict(sorted(group_rows.items())),
        }
    overlaps = []
    for left, right in combinations(SPLITS, 2):
        shared_groups, shared_questions = groups[left] & groups[right], questions[left] & questions[right]
        if shared_groups or shared_questions:
            overlaps.append({"splits": [left, right], "source_problem_ids": sorted(shared_groups),
                             "question_hashes": sorted(shared_questions)})
    if overlaps:
        raise SplitOverlapError(overlaps)
    return {
        "schema_version": "dsa_split_manifest_v1", "regime": "lotus_mirror",
        "lotus_sha": lotus_sha, "data_source_commit": data_source_commit,
        "preprocessing_sha256": {name: file_sha256(path) for name, path in preprocessing_paths.items()},
        "download_source": download, "steps_join": "space", "splits": info,
        "grouping": {"priority": ["source_problem_id", "problem_id", "id", "normalized question hash"],
                     "question_normalization": "NFKC, casefold, whitespace collapse",
                     "limitation": "Exact/format duplicates only; arbitrary paraphrases are not detected."},
        "split_overlap_audit": {"status": "passed", "overlaps": []},
        "ordered_evaluation_ids": [record["sample_id"] for record in ordered_evaluation],
        "ordered_evaluation_records": ordered_evaluation,
        "ordered_evaluation_sha256": stable_json_hash(ordered_evaluation),
        "evaluation_format": {"question": "strip question, append newline", "answer_prefix": "### ",
                              "answer_normalization": "remove commas, strip whitespace",
                              "upstream_scoring": "float exact equality; fallback stripped string equality"},
        "official_test_comparison": compare_official_test(sources["test"], official_test_path)
            if official_test_path is not None else {"status": "not_run", "reason": "no official raw test JSONL supplied"},
    }
