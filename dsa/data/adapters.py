"""Lossless adapters for raw GSM8K and LOTUS's augmented JSON records."""

from __future__ import annotations

import copy
import hashlib
import json
import unicodedata
from collections.abc import Mapping
from typing import Any


def stable_json_hash(value: Any) -> str:
    """Hash a JSON value independently of object key insertion order."""
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def normalized_question_hash(question: str) -> str:
    """Detect case/Unicode-format/whitespace duplicates, not paraphrases."""
    normalized = " ".join(unicodedata.normalize("NFKC", question).casefold().split())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def canonical_record(source: Mapping[str, Any], steps_join: str = "space") -> dict[str, Any]:
    """Preserve text and derive problem-group and rationale identities.

    Raw GSM8K's ``answer`` must have exactly one ``####`` delimiter. LOTUS
    records instead have ``steps`` and ``answer``. Empty step fragments are
    retained: joining ``["a", "", "b"]`` with spaces yields ``"a  b"``.
    Existing canonical records are accepted for annotation policy rebuilding.
    """
    if not isinstance(source, Mapping):
        raise TypeError("source record must be a mapping")
    if steps_join not in {"space", "newline"}:
        raise ValueError("steps_join must be 'space' or 'newline'")
    if not isinstance(source.get("question"), str):
        raise ValueError("question must be a string")
    question = source["question"]
    if "rationale_raw" in source:
        rationale = source["rationale_raw"]
        final_answer = source.get("final_answer")
    elif "steps" in source:
        steps = source["steps"]
        if isinstance(steps, str):
            rationale = steps
        elif isinstance(steps, list) and all(isinstance(step, str) for step in steps):
            rationale = (" " if steps_join == "space" else "\n").join(steps)
        else:
            raise ValueError("LOTUS steps must be a string or a list of strings")
        final_answer = source.get("answer")
    else:
        answer = source.get("answer")
        if not isinstance(answer, str) or answer.count("####") != 1:
            raise ValueError("raw GSM8K answer must contain exactly one #### separator")
        rationale, final_answer = answer.split("####")
        final_answer = final_answer.strip()
    if not isinstance(rationale, str) or not isinstance(final_answer, str):
        raise ValueError("rationale and final answer must be strings")
    question_hash = normalized_question_hash(question)
    source_id = next(
        (source[key] for key in ("source_problem_id", "problem_id", "id")
         if key in source and source[key] is not None),
        None,
    )
    if source_id is not None and (isinstance(source_id, bool) or not isinstance(source_id, (str, int))):
        raise ValueError("source problem ID must be a string or integer")
    source_problem_id = str(source_id) if source_id is not None else f"question:{question_hash}"
    if not source_problem_id:
        raise ValueError("source problem ID must not be empty")
    original = copy.deepcopy(source.get("original_record", dict(source)))
    content = {
        "question": question,
        "rationale_raw": rationale,
        "final_answer": final_answer,
        "source_problem_id": source_problem_id,
    }
    return {
        **content,
        "question_hash": question_hash,
        "sample_id": stable_json_hash(content),
        "original_record": original,
        "original_sha256": stable_json_hash(original),
        "steps_join": steps_join,
    }
