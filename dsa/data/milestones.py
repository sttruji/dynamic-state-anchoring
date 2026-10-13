"""New, deterministic implementation of revision 4 milestone preprocessing.

Only local arithmetic consistency is verified. Structural variable assignments do
not establish correctness of a solution. No dataset text is executed.
"""

from __future__ import annotations

import ast
import copy
from collections import Counter
from fractions import Fraction
import re
from typing import Any, Iterable


SCHEMA_VERSION = "dsa_rules_v1"
MAX_EXPRESSION_CHARS = 512
MAX_AST_NODES = 128
MAX_AST_DEPTH = 32
MAX_LITERAL_DIGITS = 64
MAX_RESULT_BITS = 4096
_SYMBOLS = str.maketrans({"×": "*", "·": "*", "÷": "/", "−": "-", "–": "-"})
_NUMBER = re.compile(r"(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d*)?|\.\d+")
_VARIABLE = re.compile(r"[A-Za-z_][A-Za-z_0-9]*\Z")
_WRAPPER = re.compile(r"<<.*?>>", re.DOTALL)


class ArithmeticRejected(ValueError):
    """An unsupported or inconsistent local arithmetic candidate."""


def _check_size(value: Fraction) -> Fraction:
    if (value.numerator.bit_length() > MAX_RESULT_BITS
            or value.denominator.bit_length() > MAX_RESULT_BITS):
        raise ArithmeticRejected("arithmetic_limit")
    return value


def _evaluate(expression: str) -> Fraction:
    """Evaluate bounded numeric arithmetic through an explicit AST whitelist.

    Numeric lexemes become generated names before parsing, so their decimal text
    goes directly to Fraction without an intermediate binary floating point value.
    Generated names are the only accepted Name nodes.
    """
    expression = expression.strip().translate(_SYMBOLS)
    if not expression:
        raise ArithmeticRejected("empty_expression")
    if len(expression) > MAX_EXPRESSION_CHARS:
        raise ArithmeticRejected("expression_limit")
    pieces: list[str] = []
    numbers: dict[str, Fraction] = {}
    cursor = 0
    while cursor < len(expression):
        char = expression[cursor]
        if char.isspace():
            pieces.append(" ")
            cursor += 1
        elif char in "+-*/()":
            pieces.append(char)
            cursor += 1
        else:
            match = _NUMBER.match(expression, cursor)
            if not match:
                raise ArithmeticRejected("unsupported_expression")
            lexeme = match.group()
            if sum(c.isdigit() for c in lexeme) > MAX_LITERAL_DIGITS:
                raise ArithmeticRejected("literal_limit")
            # A comma left over after this token is an invalid thousands group.
            if match.end() < len(expression) and expression[match.end()] == ",":
                raise ArithmeticRejected("invalid_thousands_separator")
            name = f"n{len(numbers)}"
            numbers[name] = Fraction(lexeme.replace(",", ""))
            pieces.append(name)
            cursor = match.end()
    try:
        tree = ast.parse("".join(pieces), mode="eval")
    except (SyntaxError, RecursionError, MemoryError) as exc:
        raise ArithmeticRejected("unsupported_syntax") from exc
    if sum(1 for _ in ast.walk(tree)) > MAX_AST_NODES:
        raise ArithmeticRejected("ast_limit")

    def visit(node: ast.AST, depth: int = 0) -> Fraction:
        if depth > MAX_AST_DEPTH:
            raise ArithmeticRejected("ast_depth_limit")
        if isinstance(node, ast.Expression):
            return visit(node.body, depth + 1)
        if isinstance(node, ast.Name) and node.id in numbers:
            return numbers[node.id]
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            value = visit(node.operand, depth + 1)
            return value if isinstance(node.op, ast.UAdd) else -value
        if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Sub, ast.Mult, ast.Div)):
            left, right = visit(node.left, depth + 1), visit(node.right, depth + 1)
            if isinstance(node.op, ast.Add):
                return _check_size(left + right)
            if isinstance(node.op, ast.Sub):
                return _check_size(left - right)
            if isinstance(node.op, ast.Mult):
                return _check_size(left * right)
            if not right:
                raise ArithmeticRejected("division_by_zero")
            return _check_size(left / right)
        raise ArithmeticRejected("unsupported_expression")

    return visit(tree)


def _scalar(text: str) -> Fraction:
    # A result must be one signed number or fraction, not another calculation.
    compact = text.strip().translate(_SYMBOLS)
    number = r"(?:(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d*)?|\.\d+)"
    if not re.fullmatch(rf"[+-]?\s*{number}(?:\s*/\s*[+-]?\s*{number})?", compact):
        raise ArithmeticRejected("non_scalar_result")
    return _evaluate(compact)


def _value_string(value: Fraction) -> str:
    return str(value.numerator) if value.denominator == 1 else f"{value.numerator}/{value.denominator}"


def _candidate(raw: str, start: int, end: int, rule: str) -> dict[str, Any]:
    return {"start": start, "end": end, "text": raw[start:end], "rule": rule}


def _equation(text: str) -> Fraction:
    if text.count("=") != 1:
        raise ArithmeticRejected("ambiguous_equality")
    lhs, rhs = text.split("=")
    result = _scalar(rhs)
    if _evaluate(lhs) != result:
        raise ArithmeticRejected("incorrect_arithmetic")
    return result


def _scan(raw: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    wrappers = list(_WRAPPER.finditer(raw))
    for match in wrappers:
        candidate = _candidate(raw, match.start(), match.end(), "calculator_equation")
        try:
            value = _equation(match.group()[2:-2])
        except ArithmeticRejected as exc:
            rejected.append({**candidate, "reason": str(exc)})
        else:
            accepted.append({**candidate, "validation": "arithmetic_verified", "variable": None,
                             "value": _value_string(value)})

    # Record unpaired delimiters without attempting to invent their intended span.
    covered = [(m.start(), m.end()) for m in wrappers]
    for delimiter in re.finditer(r"<<|>>", raw):
        if not any(start <= delimiter.start() < end for start, end in covered):
            rejected.append({**_candidate(raw, delimiter.start(), delimiter.end(), "calculator_equation"),
                             "reason": "malformed_calculator_wrapper"})

    offset = 0
    for line in raw.splitlines(keepends=True):
        content = line.rstrip("\r\n")
        start = offset + len(content) - len(content.lstrip())
        body = content.strip()
        end = offset + len(content.rstrip())
        offset += len(line)
        if "=" not in body or "<<" in body or ">>" in body:
            continue
        if body and body[-1] in ".;!?":
            body = body[:-1].rstrip()
            end = start + len(body)
        lhs = body.split("=", 1)[0].strip()
        variable = lhs if _VARIABLE.fullmatch(lhs) else None
        rule = "isolated_variable" if variable else "bare_numeric_equation"
        candidate = _candidate(raw, start, end, rule)
        try:
            if body.count("=") != 1:
                raise ArithmeticRejected("ambiguous_equality")
            if variable:
                value = _scalar(body.split("=", 1)[1])
            else:
                value = _equation(body)
        except ArithmeticRejected as exc:
            rejected.append({**candidate, "reason": str(exc)})
        else:
            accepted.append({**candidate, "validation": "structural_only" if variable else "arithmetic_verified",
                             "variable": variable, "value": _value_string(value)})
    accepted.sort(key=lambda event: (event["end"], event["start"]))
    return accepted, rejected


def annotate_record(record: dict[str, Any], *, mode: str = "conclusive",
                    exclude_terminal: bool = True, marker_token: str = "<SA>",
                    steps_join: str = "space") -> dict[str, Any]:
    """Annotate a raw GSM8K or LOTUS record, rebuilding cached annotations.

    Offsets refer to unchanged Python Unicode character positions in rationale_raw.
    Rebuilding from original_record makes both same-policy calls idempotent and
    different-policy calls independent of previously inserted markers.
    """
    from dsa.data.adapters import canonical_record

    if mode not in {"conclusive", "verified_only"}:
        raise ValueError("mode must be conclusive or verified_only")
    if not isinstance(exclude_terminal, bool):
        raise ValueError("exclude_terminal must be a boolean")
    if not isinstance(marker_token, str) or not marker_token or marker_token.isspace():
        raise ValueError("marker_token must be a nonempty, non-whitespace string")
    if steps_join not in {"space", "newline"}:
        raise ValueError("steps_join must be space or newline")
    if not isinstance(record, dict):
        raise ValueError("record must be an object")
    source = record
    if "schema_version" in record:
        if record["schema_version"] != SCHEMA_VERSION:
            raise ValueError("unsupported annotation schema_version")
        if not isinstance(record.get("original_record"), dict):
            raise ValueError("annotated records require original_record for provenance and rebuilds")
        source = record["original_record"]
    canonical = canonical_record(source, steps_join=steps_join)
    for name in ("question", "rationale_raw", "final_answer"):
        if marker_token in canonical[name]:
            raise ValueError(f"literal marker collision in original {name}: {marker_token}")
    raw = canonical["rationale_raw"]
    events, rejected = _scan(raw)
    if mode == "verified_only":
        structural = [event for event in events if event["validation"] == "structural_only"]
        rejected.extend({**event, "reason": "structural_only_filtered"} for event in structural)
        events = [event for event in events if event["validation"] == "arithmetic_verified"]
    if exclude_terminal and events:
        try:
            final_value = _scalar(canonical["final_answer"])
        except ArithmeticRejected:
            final_value = None
        if final_value is not None and _scalar(events[-1]["value"]) == final_value:
            rejected.append({**events.pop(), "reason": "terminal_answer_match"})
    ends = [event["end"] for event in events]
    if len(ends) != len(set(ends)):
        raise ValueError("multiple accepted conclusions share a character end")
    chunks: list[str] = []
    previous = 0
    for event in events:
        chunks.extend((raw[previous:event["end"]], marker_token))
        previous = event["end"]
    chunks.append(raw[previous:])
    marked = "".join(chunks)
    if marked.replace(marker_token, "") != raw:
        raise ValueError("marker removal failed source-preservation invariant")
    policy = {"marker_token": marker_token, "mode": mode,
              "exclude_terminal": exclude_terminal, "steps_join": steps_join}
    return {
        "schema_version": SCHEMA_VERSION,
        **{key: copy.deepcopy(canonical[key]) for key in (
            "sample_id", "source_problem_id", "question_hash", "original_record",
            "question", "rationale_raw", "final_answer", "original_sha256")},
        "rationale_marked": marked,
        "policy": policy,
        "events": [{"ordinal": ordinal, **event} for ordinal, event in enumerate(events)],
        "rejected_candidates": sorted(rejected, key=lambda item: (item["start"], item["end"], item["reason"])),
        "event_count": len(events),
    }


def map_events_to_loops(event_ends: Iterable[int], rationale_length: int,
                        loop_budget: int) -> dict[str, Any]:
    """Map 1-based raw token ends onto strictly increasing 1-based loop indices."""
    ends = list(event_ends)
    if isinstance(loop_budget, bool) or not isinstance(loop_budget, int) or loop_budget < 1:
        raise ValueError("loop_budget must be a positive integer")
    if isinstance(rationale_length, bool) or not isinstance(rationale_length, int) or rationale_length < 0:
        raise ValueError("rationale_length must be a nonnegative integer")
    if any(isinstance(end, bool) or not isinstance(end, int) for end in ends):
        raise ValueError("event ends must be integer token positions")
    if any(end < 1 or end > rationale_length for end in ends):
        raise ValueError("event ends must be inside the raw rationale token sequence")
    if any(left > right for left, right in zip(ends, ends[1:])):
        raise ValueError("event ends must be sorted in nondecreasing order")
    result: dict[str, Any] = {"gate_targets": [0] * loop_budget, "event_loops": [],
                              "supervised": False, "reason": None}
    if not ends:
        result["reason"] = "no_events"
        return result
    if len(ends) > loop_budget:
        result["reason"] = "event_overflow"
        return result
    previous = 0
    for index, end in enumerate(ends, start=1):
        ideal = (loop_budget * end + rationale_length - 1) // rationale_length
        loop = min(loop_budget - (len(ends) - index), max(previous + 1, ideal))
        result["event_loops"].append(loop)
        result["gate_targets"][loop - 1] = 1
        previous = loop
    result["supervised"] = True
    return result


def annotation_report(records: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Summarize retained coverage and reasons; annotation never filters records."""
    records = list(records)
    rules: Counter[str] = Counter()
    evidence: Counter[str] = Counter()
    reasons: Counter[str] = Counter()
    histogram: Counter[int] = Counter()
    for record in records:
        histogram[record["event_count"]] += 1
        rules.update(event["rule"] for event in record["events"])
        evidence.update(event["validation"] for event in record["events"])
        reasons.update(event["reason"] for event in record["rejected_candidates"])
    total = sum(record["event_count"] for record in records)
    count = len(records)
    return {
        "schema_version": SCHEMA_VERSION,
        "implementation": "new implementation reconstructed from revision 4 PDF; original reference package absent",
        "policy": records[0]["policy"] if records else None,
        "record_count": count,
        "no_event_count": histogram[0],
        "no_event_rate": histogram[0] / count if count else 0.0,
        "event_count": total,
        "events_per_record": total / count if count else 0.0,
        "event_count_histogram": {str(key): histogram[key] for key in sorted(histogram)},
        "rule_counts": {key: rules[key] for key in ("calculator_equation", "bare_numeric_equation", "isolated_variable")},
        "evidence_counts": dict(sorted(evidence.items())),
        "terminal_exclusions": reasons["terminal_answer_match"],
        "rejection_reasons": dict(sorted(reasons.items())),
        "terminal_policy_limitation": "Excluding the last accepted conclusion when its value matches the final answer can withhold a legitimate intermediate with that same value.",
        "tokenizer_mapping_failures": None,
        "controller_label_overflow": None,
        "coverage_scope": "input records only; tokenizer alignment and loop-label overflow are reported by tokenization",
    }
