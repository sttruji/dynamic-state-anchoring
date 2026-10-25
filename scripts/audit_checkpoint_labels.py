#!/usr/bin/env python3
"""Read-only audit of the training labels actually used by the DSA pilots.

Reconstructs labels against pinned source rows, checks arithmetic independently,
and emits a deterministic TRAIN-only packet for personal semantic review. It
does not infer semantic verdicts or change the parser, data, or model.
"""
from __future__ import annotations

import argparse
import ast
from collections import Counter
from fractions import Fraction
import json
from pathlib import Path
import re
import random
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dsa.data.adapters import stable_json_hash
from dsa.data.milestones import annotate_record
from dsa.data.tokenize import tokenize_record
from dsa.provenance import file_sha256


def exact_arithmetic(text):
    """Independent AST/Fraction checker; never executes source expressions."""
    text = re.sub(r"(?<=\d),(?=\d)", "", text).strip()
    text = text.translate(str.maketrans({"×": "*", "÷": "/", "−": "-"}))
    tree = ast.parse(text, mode="eval")
    def visit(node):
        if isinstance(node, ast.Expression):
            return visit(node.body)
        if isinstance(node, ast.Constant) and type(node.value) in (int, float):
            return Fraction(ast.get_source_segment(text, node))
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            return visit(node.operand) * (-1 if isinstance(node.op, ast.USub) else 1)
        if isinstance(node, ast.BinOp):
            a, b = visit(node.left), visit(node.right)
            if isinstance(node.op, ast.Add): return a + b
            if isinstance(node.op, ast.Sub): return a - b
            if isinstance(node.op, ast.Mult): return a * b
            if isinstance(node.op, ast.Div): return a / b
        raise ValueError("unsupported arithmetic")
    return visit(tree)


def load_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def pairwise_auc(labels, probabilities):
    positive = [p for y, p in zip(labels, probabilities) if y]
    negative = [p for y, p in zip(labels, probabilities) if not y]
    return sum((a > b) + 0.5 * (a == b) for a in positive for b in negative) / (len(positive) * len(negative))


def loop_prior_baseline(rows):
    """Estimate timing on other TRAIN rows; reproduce the saved diagnostic IDs."""
    indices = set(random.Random(7).sample(range(len(rows)), 64))
    predicted_ids = [rows[i]["sample_id"] for i in sorted(indices)]
    path = ROOT / "artifacts/flt-data-scale-audit/bundle/review/learned/diagnostics/normal-training-set/predictions.jsonl"
    assert [r["sample_id"] for r in load_jsonl(path)] == predicted_ids
    fit = [r for i, r in enumerate(rows) if i not in indices and r["controller_supervised"]]
    scored = [r for i, r in enumerate(rows) if i in indices and r["controller_supervised"]]
    probabilities = [(1 + sum(r["gate_targets"][t] for r in fit)) / (len(fit) + 2) for t in range(8)]
    labels = [y for r in scored for y in r["gate_targets"]]
    predictions = probabilities * len(scored)
    true_positive = sum(y for y, p in zip(labels, predictions) if p >= 0.5)
    writes = sum(p >= 0.5 for p in predictions)
    return {"fit_records": len(fit), "evaluation_records": len(scored),
        "positive_decisions": sum(labels), "negative_decisions": len(labels) - sum(labels),
        "write_probabilities_by_loop": probabilities, "auc": pairwise_auc(labels, predictions),
        "threshold_0_5": {"precision": true_positive / writes, "recall": true_positive / sum(labels),
                          "write_rate": writes / len(labels)},
        "scope": "TRAIN only; fit on other rows, score on the saved 64-question diagnostic subset (51 supervised); no question content used",
        "fit_scoring_ids_disjoint": True, "diagnostic_ids_match_saved_predictions": True,
        "limitations": "Timing-label baseline, not answer accuracy, semantic grounding or inference evaluation."}


def summarize_reviews(output, report, ledger):
    reviews = load_jsonl(ledger)
    packet = load_jsonl(output / "review_packet.jsonl") + load_jsonl(output / "supplemental_review_packet.jsonl")
    by_row = {r["source_row_number"]: r for r in packet}
    assert len(by_row) == len(packet) == len(reviews) == 76
    assert {r["source_row_number"] for r in reviews} == set(by_row)
    raw = json.loads((ROOT / "data/gsm_train.json").read_text())
    for r in reviews:
        source = by_row[r["source_row_number"]]
        assert r["sample_id"] == source["sample_id"] and r["question_hash"] == source["question_hash"]
        assert r["source_row_sha256"] == stable_json_hash(raw[r["source_row_number"] - 1])
        assert r["reasoning"] and r["independent_solution"] and r["accepted_event_count"] == len(source["events"])
    local_ids = set(report["manual_packet"]["review_source_rows"][:32])
    supported_local = [r for r in reviews if r["source_row_number"] in local_ids and r["verdict"] == "supported"]
    replacements = sorted((r for r in reviews if r["source_row_number"] not in local_ids and r["verdict"] == "supported"),
        key=lambda r: stable_json_hash([19, "fixture_addition", r["sample_id"]]))[:3]
    fixture = sorted(supported_local + replacements, key=lambda r: r["source_row_number"])
    assert len(fixture) == 32 and len(supported_local) == 29
    return {"schema_version": "dsa_checkpoint_label_personal_review_v1", "date": "2026-10-19",
        "reviewer": "Primary assistant; individually written semantic judgments, not human or independent second-reviewer labels",
        "mechanical_audit": report, "ledger_path": str(ledger.relative_to(ROOT)), "ledger_sha256": file_sha256(ledger),
        "personal_review": {"records": len(reviews), "accepted_checkpoints": sum(r["accepted_event_count"] for r in reviews),
            "verdict_counts": dict(Counter(r["verdict"] for r in reviews)),
            "local_32_verdicts": dict(Counter(r["verdict"] for r in reviews if r["source_row_number"] in local_ids)),
            "checkpoint_tags": dict(Counter(t for r in reviews for t in r["checkpoint_tags"])),
            "all_equal_final_value_cases_reviewed": True, "all_after_rejected_arithmetic_cases_reviewed": True,
            "selection_bias": "Purposive risk coverage, not a population semantic error-rate estimate"},
        "proposed_fixture": {"status": "design only; source and prepared training data unchanged",
            "purpose": "small curated mechanism-development fixture, not a GSM8K benchmark",
            "expected_records": len(fixture),
            "expected_supervised_records": sum(r["accepted_event_count"] > 0 for r in fixture),
            "expected_checkpoint_events": sum(r["accepted_event_count"] for r in fixture) - 1,
            "training_rows": [{k: r[k] for k in ("source_row_number", "sample_id", "question_hash")} for r in fixture],
            "replacement_rows": [r["source_row_number"] for r in replacements],
            "omitted_local_rows": [r["source_row_number"] for r in reviews if r["source_row_number"] in local_ids and r["verdict"] != "supported"],
            "proposed_event_change": {"source_row_number": 112564, "remove_equation": "<<120=120>>",
                "reason": "given restatement; regenerate markers, teacher indices and loop labels in a separately named derivative"}},
        "source_or_model_modified": False, "new_model_training_or_answer_evaluation": False}


def build_audit(source, historical, local, tokenizer_dir, output):
    from transformers import AutoTokenizer
    if output.exists() and any(output.iterdir()):
        raise ValueError("Use an empty output directory; prior audit artifacts are preserved")
    expected = json.loads((ROOT / "provenance/lotus_download_manifest.json").read_text())
    source_hash = file_sha256(source)
    assert source_hash == expected["splits"]["train"]["json_sha256"]
    # This script intentionally opens only the TRAIN source and TRAIN derivatives.
    raw = json.loads(source.read_text())
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir, local_files_only=True)
    historical_rows, local_rows = load_jsonl(historical), load_jsonl(local)
    groups = {"historical_512": historical_rows, "local_32": local_rows}
    results, aggregate = {}, {}
    for name, rows in groups.items():
        counts, flags = Counter(), Counter()
        event_loops, positive_labels = Counter(), Counter()
        for saved in rows:
            row_number = saved["source_row_number"]
            annotation = annotate_record(raw[row_number - 1])
            assert annotation["sample_id"] == saved["sample_id"]
            assert annotation["question_hash"] == saved["question_hash"]
            assert saved["source_split_sha256"] == source_hash
            rebuilt = tokenize_record(annotation, tokenizer, loop_budget=saved["loop_budget"])
            # Compare every reconstructed model input and supervision field.
            assert all(saved[k] == v for k, v in rebuilt.items()), row_number
            explicit = saved["explicit_input_ids"]
            marker_positions = [i for i, t in enumerate(explicit) if t == saved["marker_token_id"]]
            assert marker_positions == [i + 1 for i in saved["checkpoint_positions"]]
            text = annotation["rationale_raw"]
            assert annotation["rationale_marked"].replace("<SA>", "") == text
            row_flags = []
            rejected = annotation["rejected_candidates"]
            bad = [r for r in rejected if r["reason"] == "incorrect_arithmetic"]
            terminal = [r for r in rejected if r["reason"] == "terminal_answer_match"]
            if bad: row_flags.append("rejected_arithmetic")
            if not annotation["events"]: row_flags.append("no_events")
            if terminal and any(r["end"] > terminal[0]["end"] for r in rejected):
                row_flags.append("terminal_exclusion_before_later_rejection")
            final_value = None
            try: final_value = exact_arithmetic(annotation["final_answer"])
            except (ValueError, SyntaxError, ZeroDivisionError): pass
            accepted = []
            for event, loop, checkpoint, end in zip(annotation["events"], saved["event_loops"],
                    saved["checkpoint_positions"], saved["raw_event_token_ends"]):
                body = event["text"][2:-2]
                lhs, rhs = body.split("=")
                assert exact_arithmetic(lhs) == exact_arithmetic(rhs)
                ideal = (saved["loop_budget"] * end + saved["raw_rationale_token_length"] - 1) // saved["raw_rationale_token_length"]
                counts["accepted_equations_independently_verified"] += 1
                counts["schedule_adjusted_events"] += loop != ideal
                counts["accepted_value_equals_final_answer"] += exact_arithmetic(rhs) == final_value
                if loop != ideal: row_flags.append("schedule_adjusted")
                if exact_arithmetic(rhs) == final_value: row_flags.append("accepted_final_value")
                if any(r["end"] < event["start"] for r in bad): row_flags.append("accepted_after_rejected_arithmetic")
                if loop == saved["loop_budget"]: row_flags.append("write_on_final_loop")
                target_token = tokenizer.decode([explicit[checkpoint]])
                counts["checkpoint_target_token:" + target_token] += 1
                accepted.append({"equation": event["text"], "loop": loop, "ideal_loop": ideal,
                    "target_token": target_token, "target_prefix": tokenizer.decode(explicit[:checkpoint + 1])})
                event_loops[loop] += 1
            row_flags = sorted(set(row_flags))
            if len(accepted) >= 5: row_flags.append("dense_events")
            counts["records"] += 1
            counts["controller_supervised_records"] += saved["controller_supervised"]
            counts["no_event_records"] += not annotation["events"]
            counts["masked_records"] += not saved["controller_supervised"]
            counts["rejected_arithmetic_candidates"] += len(bad)
            counts["terminal_exclusions"] += len(terminal)
            if saved["controller_supervised"]: positive_labels.update(saved["gate_targets"])
            flags.update(row_flags)
            results[row_number] = {"source_row_number": row_number, "sample_id": saved["sample_id"],
                "question_hash": saved["question_hash"], "question": annotation["question"],
                "rationale_marked": annotation["rationale_marked"], "answer": annotation["final_answer"],
                "gate_targets": saved["gate_targets"], "controller_supervised": saved["controller_supervised"],
                "events": accepted, "rejected": rejected, "flags": row_flags}
        aggregate[name] = {"counts": dict(counts), "flagged_records": dict(flags),
            "positive_negative_labels": dict(positive_labels), "accepted_events_by_loop": dict(event_loops),
            "unique_questions": len({r["question_hash"] for r in rows}),
            "all_reconstructed_fields_equal": True}
    # Inspect every local pilot row and add 32 historical rows, prioritizing
    # failure modes rather than estimating a population semantic error rate.
    chosen = [r["source_row_number"] for r in local_rows]
    selected = set(chosen)
    strata = ["terminal_exclusion_before_later_rejection", "accepted_final_value",
              "accepted_after_rejected_arithmetic", "rejected_arithmetic", "schedule_adjusted",
              "dense_events", "write_on_final_loop", "no_events"]
    def priority(n): return stable_json_hash([19, n])
    candidates = sorted({r["source_row_number"] for r in historical_rows} - selected, key=priority)
    for _ in range(4):
        for flag in strata:
            match = next((n for n in candidates if n not in selected and flag in results[n]["flags"]), None)
            if match is not None and len(chosen) < 64:
                chosen.append(match); selected.add(match)
    for n in candidates:
        if len(chosen) >= 64: break
        if n not in selected: chosen.append(n); selected.add(n)
    packet = [{"review_number": i + 1, **results[n]} for i, n in enumerate(chosen)]
    extra = [r for n, r in results.items() if n not in selected and
             ("accepted_final_value" in r["flags"] or "accepted_after_rejected_arithmetic" in r["flags"])]
    output.mkdir(parents=True)
    for name, content in [("all_training_rows.jsonl", list(results.values())), ("review_packet.jsonl", packet),
                          ("supplemental_review_packet.jsonl", extra)]:
        (output / name).write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in content))
    baseline = loop_prior_baseline(historical_rows)
    (output / "loop_prior_baseline.json").write_text(json.dumps(baseline, indent=2) + "\n")
    report = {"schema_version": "dsa_checkpoint_label_audit_v1", "date": "2026-10-19",
        "scope": "TRAIN-only reconstruction and independent arithmetic checks; semantic review is separate",
        "source_sha256": source_hash, "inputs_sha256": {str(p.relative_to(ROOT)): file_sha256(p)
            for p in (historical, local)}, "script_sha256": file_sha256(Path(__file__)),
        "groups": aggregate, "unique_records_audited": len(results),
        "local_rows_also_in_historical": len({r['sample_id'] for r in local_rows} & {r['sample_id'] for r in historical_rows}),
        "manual_packet": {"records": len(packet), "all_local_32_included": True,
            "selection": "all local 32; 32 historical additions, hash-ordered seed 19, four passes over eight risk strata then fill",
            "review_source_rows": chosen, "sha256": file_sha256(output / "review_packet.jsonl")},
        "supplemental_packet": {"records": len(extra), "scope": "all remaining accepted-final-value and accepted-after-rejected-arithmetic TRAIN cases",
            "sha256": file_sha256(output / "supplemental_review_packet.jsonl")},
        "loop_prior_baseline": baseline,
        "new_model_training_or_answer_evaluation": False,
        "limitations": ["Reconstruction parity verifies implementation, not semantic truth.",
            "Independent arithmetic verifies local equations, not their suitability for the word problem.",
            "The personal review packet is purposive; its verdict rates are not population estimates.",
            "Token-position scheduling is a heuristic, not verified latent progress."]}
    (output / "mechanical_report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--review-ledger", type=Path)
    parser.add_argument("--summary-output", type=Path)
    args = parser.parse_args()
    if bool(args.review_ledger) != bool(args.summary_output):
        parser.error("--review-ledger and --summary-output must be supplied together")
    report = build_audit(ROOT / "data/gsm_train.json",
        ROOT / "artifacts/flt-data-scale-audit/bundle/prepared/train.tokenized.jsonl",
        ROOT / "artifacts/local-mps-20261018/prepared/train.tokenized.jsonl",
        ROOT / "artifacts/local-mps-20261018/prepared/tokenizer", args.output.resolve())
    if args.review_ledger:
        summary = summarize_reviews(args.output.resolve(), report, args.review_ledger.resolve())
        args.summary_output.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(report, indent=2))
