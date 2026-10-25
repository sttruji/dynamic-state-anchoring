"""Answer extraction and scoring shared by every model in the harness.

``correct`` keeps the pinned LOTUS rules so new rows stay comparable with every
earlier DSA number. ``strict_correct`` additionally requires the answer region
to be one well-formed scalar, removing the incidental credit documented in
docs/FINDINGS.md (``iced.5`` or ``10*5`` scored as 5). Bump SCORER_VERSION on any
change; the harness refuses to mix results across versions.
"""
from __future__ import annotations

import math
import re

from dsa.evaluation.evaluate import extract_answer, score_answer

SCORER_VERSION = "bench-v1"
FLOAT_TOLERANCE = 1e-4  # Only used for gold answers that are non-integers (GSM-Hard).
_SCALAR = re.compile(r"-?\d+(?:\.\d+)?")
_MARKERS = ("###", "The answer is:", "The answer is", "answer is:", "answer is")
_CHOICE = re.compile(r"\b([A-E])\b")


def normalize_number(text: str) -> str:
    return text.replace(",", "").replace("$", "").strip().rstrip(".").strip()


def strict_answer(text: str) -> str | None:
    """Return the answer only when the region after the last marker is one scalar."""
    region = text
    for marker in _MARKERS:
        if marker in text:
            region = text.rsplit(marker, 1)[1]
            break
    lines = region.strip().splitlines()
    region = normalize_number(lines[0] if lines else "")
    return region if _SCALAR.fullmatch(region) else None


def _numeric_equal(prediction: str, gold: str) -> bool:
    try:
        left, right = float(prediction), float(gold)
    except (ValueError, OverflowError):
        return False
    if not (math.isfinite(left) and math.isfinite(right)):
        return False
    if right == int(right):
        return left == right
    return math.isclose(left, right, rel_tol=FLOAT_TOLERANCE, abs_tol=FLOAT_TOLERANCE)


def extract_choice(text: str) -> str | None:
    region = text
    for marker in _MARKERS:
        if marker in text:
            region = text.rsplit(marker, 1)[1]
            break
    found = _CHOICE.findall(region)
    return found[0] if found else None


def score(generated_text: str, gold: str, answer_type: str = "numeric") -> dict:
    """Score one raw model output against its gold answer."""
    if answer_type == "choice":
        prediction = extract_choice(generated_text)
        hit = prediction is not None and prediction == gold.strip().upper()
        return {"prediction": prediction or "", "correct": hit, "strict_correct": hit,
                "valid_answer": prediction is not None}
    if answer_type != "numeric":
        raise ValueError(f"unknown answer_type: {answer_type}")
    gold = normalize_number(gold)
    prediction = extract_answer(generated_text)
    strict = strict_answer(generated_text)
    try:
        valid = math.isfinite(float(prediction))
    except (ValueError, OverflowError):
        valid = False
    lotus = score_answer(prediction, gold)
    # LOTUS uses exact float equality; non-integer golds (GSM-Hard) also get a tolerance.
    correct = lotus or (valid and _numeric_equal(prediction, gold))
    return {"prediction": prediction, "correct": bool(correct),
            "strict_correct": strict is not None and _numeric_equal(strict, gold),
            "valid_answer": valid}
