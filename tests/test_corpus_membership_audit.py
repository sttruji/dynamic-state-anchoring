"""Validate audit failure signals without changing or filtering source rows."""

import json
import tempfile
import unittest
from pathlib import Path

from scripts.audit_lotus_membership import audit_corpus, compare_membership, inspect_rows, numeric_key
from dsa.data.adapters import normalized_question_hash


def lotus(question="Q", steps=None, answer="2", **extra):
    return {"question": question, "steps": ["one", "plus", "one"] if steps is None else steps,
            "answer": answer, **extra}


class MembershipAuditTests(unittest.TestCase):
    def test_occurrences_and_distinct_rationale_ids_are_not_conflated(self):
        rows = [lotus(), lotus(), lotus(steps=["different", "reasoning"])]
        summary, entries = inspect_rows(rows)
        self.assertEqual(summary["rows"], 3)
        self.assertEqual(summary["unique_normalized_questions"], 1)
        self.assertEqual(summary["unique_sample_ids"], 2)
        self.assertEqual(summary["repeated_sample_id_rows"], 1)
        self.assertEqual(summary["repeated_exact_original_record_rows"], 1)
        self.assertEqual(summary["distinct_rationales_per_question_histogram"], {"2": 1})
        self.assertEqual(summary["rows_per_question_histogram"], {"3": 1})
        self.assertEqual(entries[0]["sample_id"], entries[1]["sample_id"])
        self.assertNotEqual(entries[0]["sample_id"], entries[2]["sample_id"])

    def test_normalized_question_identity_survives_distinct_explicit_ids(self):
        summary, entries = inspect_rows([lotus(" Q  x ", id="a"), lotus("q X", id="b")])
        self.assertEqual(summary["unique_problem_groups"], 2)
        self.assertEqual(summary["unique_normalized_questions"], 1)
        self.assertEqual(entries[0]["question_hash"], entries[1]["question_hash"])

    def test_answer_formatting_and_numeric_conflict_are_separate(self):
        summary, _ = inspect_rows([lotus(answer="1,000"), lotus(answer="1000.0")])
        self.assertEqual(summary["questions_with_answer_normalization_conflicts"], 1)
        self.assertEqual(summary["questions_with_numeric_answer_conflicts"], 0)
        conflicting, _ = inspect_rows([lotus(answer="1,000"), lotus(answer="1001")])
        self.assertEqual(conflicting["questions_with_numeric_answer_conflicts"], 1)
        self.assertEqual(numeric_key("-0.0"), numeric_key("0"))
        self.assertIsNone(numeric_key("nan"))

    def test_malformed_and_control_tokens_remain_visible(self):
        rows = [lotus("<SA> Q", ["<SA>", "<|latent|>"], "<SA> 2"),
                lotus(steps=[None]), None, lotus(question=" ", steps=[], answer=" ")]
        summary, entries = inspect_rows(rows)
        self.assertEqual(summary["rows"], 4)
        self.assertEqual(summary["malformed_rows"], 2)
        self.assertEqual(len(entries), 2)
        self.assertEqual(summary["control_token_collisions"]["<SA>"]["rows"], 1)
        self.assertEqual(summary["control_token_collisions"]["<SA>"]["occurrences"], 3)
        self.assertEqual(summary["empty_field_rows"], {"question": 1, "rationale": 1, "answer": 1})

    def test_order_membership_and_answers_are_independent_checks(self):
        a = {"question_hash": "a", "answer_normalized": "1"}
        b = {"question_hash": "b", "answer_normalized": "2"}
        result = compare_membership([a, b], [b, a])
        self.assertTrue(result["question_membership_match"])
        self.assertFalse(result["ordered_question_answer_match"])
        conflict = compare_membership([a], [{**a, "answer_normalized": "2"}])
        self.assertEqual(conflict["answer_mismatch_question_hashes"], ["a"])

    def test_overlap_stops_comparison_but_preserves_descriptor_and_denominator(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sources = {"train": [lotus("Q", id="x"), lotus("Q", id="x")],
                       "valid": [lotus(" q ", id="y")], "test": [lotus("T")]}
            paths = {}
            originals = {}
            for name, rows in sources.items():
                paths[name] = root / f"{name}.json"
                paths[name].write_text(json.dumps(rows), encoding="utf-8")
                originals[name] = paths[name].read_bytes()
            train, test = root / "official_train.jsonl", root / "official_test.jsonl"
            train.write_text(json.dumps({"question": "Q", "answer": "1+1=2\n#### 2"}) + "\n", encoding="utf-8")
            test.write_text(json.dumps({"question": "T", "answer": "1+1=2\n#### 2"}) + "\n", encoding="utf-8")
            report = audit_corpus(paths, official_train=train, official_test=test)
            self.assertFalse(report["primary_comparison_eligibility"]["eligible"])
            overlap = report["split_overlap_audit"]["pairs"][0]
            self.assertEqual(overlap["question_hashes"], [normalized_question_hash("Q")])
            self.assertEqual(overlap["rows_from_overlapping_questions"], {"train": 2, "valid": 1})
            self.assertEqual(report["splits"]["train"]["rows"], 2)
            self.assertEqual(report["evaluation_denominator"]["supplied_test_rows"], 1)
            self.assertTrue(report["official_membership_comparisons"]["test_vs_official_test"]["ordered_question_answer_match"])
            for name, path in paths.items():
                self.assertEqual(path.read_bytes(), originals[name])

    def test_distinct_questions_sharing_explicit_problem_group_trigger_overlap(self):
        # Question-only checking would miss this explicit source ID collision.
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = {}
            for split, question in (("train", "A"), ("valid", "B"), ("test", "C")):
                paths[split] = root / f"{split}.json"
                paths[split].write_text(json.dumps([lotus(question, id="shared")]), encoding="utf-8")
            official = root / "official.jsonl"
            official.write_text(json.dumps({"question": "C", "answer": "reason\n#### 2"}) + "\n", encoding="utf-8")
            report = audit_corpus(paths, official_train=official, official_test=official)
            self.assertFalse(report["split_overlap_audit"]["passed"])
            self.assertFalse(report["primary_comparison_eligibility"]["eligible"])


if __name__ == "__main__":
    unittest.main()
