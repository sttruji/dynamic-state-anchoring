"""Behavioral checks for the newly reconstructed milestone implementation."""

import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from dsa.data.milestones import annotate_record, annotation_report, map_events_to_loops


ROOT = Path(__file__).resolve().parents[1]


def raw(rationale, answer="999", question="How many?", **extra):
    return {"question": question, "answer": rationale + "#### " + answer, **extra}


class AnnotationTests(unittest.TestCase):
    def test_calculator_wrappers_anywhere_and_repeated_result_preserved(self):
        rationale = "First, <<8+4=12>>12 marbles.\nThen <<12*2=24>>24.\n"
        result = annotate_record(raw(rationale, "24"))
        self.assertEqual(result["rationale_marked"], "First, <<8+4=12>><SA>12 marbles.\nThen <<12*2=24>>24.\n")
        event = result["events"][0]
        self.assertEqual(event["rule"], "calculator_equation")
        self.assertEqual(event["validation"], "arithmetic_verified")
        self.assertEqual(rationale[event["start"]:event["end"]], "<<8+4=12>>")

    def test_bare_equation_marker_before_punctuation(self):
        result = annotate_record(raw("  8 + 4 = 12.  \r\n"))
        self.assertEqual(result["rationale_marked"], "  8 + 4 = 12<SA>.  \r\n")
        self.assertEqual(result["events"][0]["rule"], "bare_numeric_equation")

    def test_isolated_variable_is_structural(self):
        result = annotate_record(raw("x = 12.\n"))
        self.assertEqual(result["rationale_marked"], "x = 12<SA>.\n")
        self.assertEqual(result["events"][0]["variable"], "x")
        self.assertEqual(result["events"][0]["validation"], "structural_only")

    def test_variable_names(self):
        for name in ("x", "total", "total_2", "_x"):
            with self.subTest(name=name):
                result = annotate_record(raw(f"{name} = -1/2."))
                self.assertEqual(result["events"][0]["value"], "-1/2")

    def test_verified_only_filters_structural(self):
        result = annotate_record(raw("<<2+2=4>>\nx=4."), mode="verified_only")
        self.assertEqual(result["event_count"], 1)
        self.assertEqual(result["rejected_candidates"][0]["reason"], "structural_only_filtered")

    def test_whole_line_requirement_rejects_prose_and_conditions(self):
        for rationale in ("If x = 12, continue.", "We have 8 + 4 = 12.", "x = 12 apples.", "Therefore x = 12."):
            with self.subTest(rationale=rationale):
                result = annotate_record(raw(rationale))
                self.assertEqual(result["event_count"], 0)
                self.assertEqual(len(result["rejected_candidates"]), 1)

    def test_boundaries_alone_are_not_labels(self):
        self.assertEqual(annotate_record(raw("Therefore, the answer follows.\nFirst. Second.\n"))["event_count"], 0)

    def test_decimal_fraction_exactness(self):
        result = annotate_record(raw("0.1 + 0.2 = 0.3.\n<<1/2 + 1/3 = 5/6>>"))
        self.assertEqual([event["value"] for event in result["events"]], ["3/10", "5/6"])

    def test_no_binary_float_rounding(self):
        result = annotate_record(raw("<<9007199254740992 + 1 = 9007199254740993>>"))
        self.assertEqual(result["events"][0]["value"], "9007199254740993")

    def test_unary_parentheses_and_symbols(self):
        result = annotate_record(raw("<<−(2 + 3) × 4 ÷ -2 = 10>>"))
        self.assertEqual(result["events"][0]["value"], "10")
        self.assertEqual(result["events"][0]["text"], "<<−(2 + 3) × 4 ÷ -2 = 10>>")

    def test_thousands_separators(self):
        result = annotate_record(raw("<<1,200 + 34,500 = 35,700>>"))
        self.assertEqual(result["events"][0]["value"], "35700")

    def test_invalid_thousands_separators(self):
        for expression in ("12,00+1=1201", "1,2345+1=12346", "1234,567+1=1234568", "1,,000=1000"):
            with self.subTest(expression=expression):
                result = annotate_record(raw(f"<<{expression}>>"))
                self.assertEqual(result["event_count"], 0)

    def test_wrong_arithmetic_rejected(self):
        result = annotate_record(raw("<<8+4=13>>"))
        self.assertEqual(result["event_count"], 0)
        self.assertEqual(result["rejected_candidates"][0]["reason"], "incorrect_arithmetic")

    def test_division_by_zero_rejected(self):
        result = annotate_record(raw("<<1/0=0>>\nx=1/0."))
        self.assertEqual(result["event_count"], 0)
        self.assertEqual([event["reason"] for event in result["rejected_candidates"]], ["division_by_zero"] * 2)

    def test_unsupported_syntax_is_never_executed(self):
        expressions = ("abs(-2)=2", "__import__('os').system('false')=0", "x+1=2", "2**3=8", "5%=0.05",
                       "2//1=2", "1<2=1", "1=1=1", "1_000=1000", "1e3=1000", "(lambda:1)()=1")
        for expression in expressions:
            with self.subTest(expression=expression):
                result = annotate_record(raw(f"<<{expression}>>"))
                self.assertEqual(result["event_count"], 0)
                self.assertEqual(len(result["rejected_candidates"]), 1)

    def test_result_must_be_scalar(self):
        for expression in ("2+2=2+2", "2+2=(4)", "2+2=four"):
            with self.subTest(expression=expression):
                self.assertEqual(annotate_record(raw(f"<<{expression}>>"))["event_count"], 0)

    def test_expression_and_literal_limits(self):
        for expression in ("1" * 65 + "=1", "+".join(["1"] * 300) + "=300", "(" * 250 + "1" + ")" * 250 + "=1"):
            with self.subTest(size=len(expression)):
                self.assertEqual(annotate_record(raw(f"<<{expression}>>"))["event_count"], 0)

    def test_malformed_wrappers_are_diagnostics(self):
        result = annotate_record(raw("<<1+1=2\n2+2=4>>"))
        self.assertEqual(result["event_count"], 0)
        self.assertTrue(result["rejected_candidates"])
        result = annotate_record(raw("text <<1+1=2\n"))
        self.assertEqual(result["rejected_candidates"][0]["reason"], "malformed_calculator_wrapper")

    def test_terminal_exclusion_only_last_accepted_event(self):
        result = annotate_record(raw("<<1+1=2>>\n<<2*1=2>>\n", "2"))
        self.assertEqual(result["event_count"], 1)
        self.assertEqual(result["events"][0]["text"], "<<1+1=2>>")
        self.assertEqual(result["rejected_candidates"][0]["reason"], "terminal_answer_match")

    def test_terminal_inclusion_option(self):
        result = annotate_record(raw("<<1+1=2>>", "2"), exclude_terminal=False)
        self.assertEqual(result["event_count"], 1)

    def test_final_comparison_exact_fraction(self):
        self.assertEqual(annotate_record(raw("<<1/4=0.25>>", "1/4"))["event_count"], 0)

    def test_non_numeric_final_does_not_invent_exclusion(self):
        self.assertEqual(annotate_record(raw("<<1+1=2>>", "two"))["event_count"], 1)

    def test_terminal_structural_event(self):
        self.assertEqual(annotate_record(raw("<<1+1=2>>\nx=2.", "2"))["event_count"], 1)

    def test_verified_policy_terminal_applies_after_filter(self):
        result = annotate_record(raw("<<1+1=2>>\nx=3.", "2"), mode="verified_only")
        self.assertEqual(result["event_count"], 0)

    def test_preserves_source_and_unicode_character_offsets(self):
        rationale = "🧮 café\t<<2+3=5>>5.  \r\nx = 7!\n"
        source = raw(rationale, question="Unchanged?  \n")
        original = copy.deepcopy(source)
        result = annotate_record(source)
        self.assertEqual(source, original)
        self.assertEqual(result["original_record"], original)
        self.assertEqual(result["rationale_raw"], rationale)
        self.assertEqual(result["rationale_marked"].replace("<SA>", ""), rationale)
        self.assertEqual(result["events"][0]["start"], rationale.index("<<"))
        self.assertEqual(result["question"], original["question"])

    def test_question_and_final_answer_never_scanned(self):
        result = annotate_record({"question": "<<2+2=4>>", "steps": "No equations.", "answer": "<<1+1=2>>"})
        self.assertEqual(result["event_count"], 0)

    def test_marker_collisions_rejected(self):
        for source in (raw("<SA>"), raw("x=1", question="<SA>"), raw("x=1", answer="<SA>")):
            with self.subTest(source=source), self.assertRaisesRegex(ValueError, "collision"):
                annotate_record(source)

    def test_custom_marker(self):
        result = annotate_record(raw("x=1."), marker_token="<ANCHOR>")
        self.assertEqual(result["rationale_marked"], "x=1<ANCHOR>.")

    def test_idempotent_annotation(self):
        result = annotate_record(raw("<<1+1=2>>\nx=2."))
        self.assertEqual(annotate_record(result), result)

    def test_policy_change_rebuilds_from_original(self):
        source = raw("<<1+1=2>>\nx=3.", "2")
        annotated = annotate_record(source, exclude_terminal=False)
        rebuilt = annotate_record(annotated, mode="verified_only", marker_token="<NEW>")
        self.assertEqual(rebuilt, annotate_record(source, mode="verified_only", marker_token="<NEW>"))
        self.assertNotIn("<SA>", rebuilt["rationale_marked"])

    def test_rebuild_does_not_trust_cached_events(self):
        annotated = annotate_record(raw("x=1."))
        annotated["rationale_marked"] = "tampered"
        annotated["events"] = []
        self.assertEqual(annotate_record(annotated)["event_count"], 1)

    def test_annotated_record_without_original_rejected(self):
        with self.assertRaisesRegex(ValueError, "original_record"):
            annotate_record({"schema_version": "dsa_rules_v1", "question": "?"})

    def test_invalid_policies_rejected(self):
        for policy in ({"mode": "all"}, {"exclude_terminal": 1}, {"marker_token": ""}, {"marker_token": " "}, {"steps_join": "auto"}):
            with self.subTest(policy=policy), self.assertRaises(ValueError):
                annotate_record(raw("x=1."), **policy)

    def test_bad_source_records_rejected(self):
        sources = ([], {}, {"question": "?", "answer": "x=1"}, {"question": "?", "answer": "#### 1 #### 2"},
                   {"question": "?", "steps": [1], "answer": "1"})
        for source in sources:
            with self.subTest(source=source), self.assertRaises((ValueError, TypeError)):
                annotate_record(source)

    def test_lotus_join_contract(self):
        source = {"question": "?", "steps": ["x", "=", "12.", "y", "=", "24."], "answer": "99"}
        result = annotate_record(source)
        self.assertEqual(result["rationale_raw"], "x = 12. y = 24.")
        self.assertEqual(result["event_count"], 0)
        lines = {"question": "?", "steps": ["x=12.", "y=24."], "answer": "99"}
        self.assertEqual(annotate_record(lines, steps_join="newline")["event_count"], 2)

    def test_stable_ids_and_shared_problem_groups(self):
        first = annotate_record(raw("x=1.", question=" HOW   MANY? "))
        second = annotate_record(raw("x=2.", question="how many?"))
        self.assertEqual(first["source_problem_id"], second["source_problem_id"])
        self.assertNotEqual(first["sample_id"], second["sample_id"])
        self.assertEqual(len(first["original_sha256"]), 64)

    def test_event_ordinals_are_contiguous(self):
        result = annotate_record(raw("<<1=1>><<2=2>>\nx=3.\n<<4=4>>"))
        self.assertEqual([event["ordinal"] for event in result["events"]], list(range(4)))

    def test_report_retains_zero_event_records(self):
        records = [annotate_record(raw("x=1.")), annotate_record(raw("Nothing.")), annotate_record(raw("<<1=1>>", "1"))]
        report = annotation_report(records)
        self.assertEqual(report["record_count"], 3)
        self.assertEqual(report["no_event_count"], 2)
        self.assertEqual(report["terminal_exclusions"], 1)
        self.assertEqual(report["rule_counts"]["isolated_variable"], 1)
        self.assertEqual(report["no_event_rate"], 2/3)


class LoopScheduleTests(unittest.TestCase):
    def test_text_progress_schedule(self):
        result = map_events_to_loops([2, 5, 10], 10, 8)
        self.assertEqual(result["event_loops"], [2, 4, 8])
        self.assertEqual(result["gate_targets"], [0, 1, 0, 1, 0, 0, 0, 1])
        self.assertTrue(result["supervised"])
        self.assertIsNone(result["reason"])

    def test_reserve_late_slots(self):
        self.assertEqual(map_events_to_loops([8, 9, 10], 10, 3)["event_loops"], [1, 2, 3])

    def test_multiple_boundaries_in_one_token(self):
        self.assertEqual(map_events_to_loops([2, 2, 2], 4, 4)["event_loops"], [2, 3, 4])

    def test_no_events_masks_auxiliary_losses(self):
        result = map_events_to_loops([], 0, 4)
        self.assertFalse(result["supervised"])
        self.assertEqual(result["reason"], "no_events")
        self.assertEqual(result["gate_targets"], [0, 0, 0, 0])

    def test_overflow_masks_without_merging(self):
        result = map_events_to_loops([1, 2, 3], 3, 2)
        self.assertFalse(result["supervised"])
        self.assertEqual(result["reason"], "event_overflow")
        self.assertEqual(result["event_loops"], [])

    def test_invalid_schedule_inputs(self):
        inputs = (([0], 3, 4), ([4], 3, 4), ([2, 1], 3, 4), ([True], 3, 4), ([1.0], 3, 4),
                  ([1], 0, 4), ([], -1, 4), ([], 1, 0), ([], 1, True))
        for arguments in inputs:
            with self.subTest(arguments=arguments), self.assertRaises(ValueError):
                map_events_to_loops(*arguments)

    def test_all_small_valid_schedules_are_monotonic_and_bounded(self):
        from itertools import combinations_with_replacement
        for length in range(1, 5):
            for budget in range(1, 5):
                for count in range(1, budget + 1):
                    for ends in combinations_with_replacement(range(1, length + 1), count):
                        result = map_events_to_loops(ends, length, budget)
                        loops = result["event_loops"]
                        self.assertEqual(len(set(loops)), count)
                        self.assertEqual(loops, sorted(loops))
                        self.assertTrue(all(1 <= loop <= budget for loop in loops))
                        self.assertEqual(sum(result["gate_targets"]), count)


class CLITests(unittest.TestCase):
    def run_cli(self, *arguments):
        return subprocess.run([sys.executable, str(ROOT / "scripts/annotate_milestones.py"), *map(str, arguments)],
                              capture_output=True, text=True, cwd=ROOT)

    def test_cli_writes_derivative_and_coverage(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source.jsonl"
            output = Path(directory) / "annotated.jsonl"
            report = Path(directory) / "report.json"
            source.write_text(json.dumps(raw("x=1.")) + "\n", encoding="utf-8")
            original = source.read_bytes()
            process = self.run_cli("--input", source, "--output", output, "--report", report)
            self.assertEqual(process.returncode, 0, process.stderr)
            self.assertEqual(json.loads(output.read_text())["event_count"], 1)
            self.assertEqual(json.loads(report.read_text())["record_count"], 1)
            self.assertEqual(source.read_bytes(), original)

    def test_cli_invalid_record_is_atomic_and_has_line_number(self):
        for invalid in ("not json", "{}", "[]", "", '{"question":"?","answer":"#### 1","extra":NaN}'):
            with self.subTest(invalid=invalid), tempfile.TemporaryDirectory() as directory:
                source = Path(directory) / "source.jsonl"
                output = Path(directory) / "annotated.jsonl"
                report = Path(directory) / "report.json"
                source.write_text(json.dumps(raw("x=1.")) + "\n" + invalid + "\n", encoding="utf-8")
                output.write_text("old corpus", encoding="utf-8")
                report.write_text("old report", encoding="utf-8")
                process = self.run_cli("--input", source, "--output", output, "--report", report)
                self.assertNotEqual(process.returncode, 0)
                self.assertIn("line 2", process.stderr)
                self.assertEqual(output.read_text(), "old corpus")
                self.assertEqual(report.read_text(), "old report")

    def test_cli_invalid_record_does_not_create_output(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source.jsonl"
            output = Path(directory) / "annotated.jsonl"
            source.write_text("{}\n", encoding="utf-8")
            process = self.run_cli("--input", source, "--output", output)
            self.assertNotEqual(process.returncode, 0)
            self.assertFalse(output.exists())

    def test_cli_preserves_unicode_separators_inside_json_strings(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source.jsonl"
            output = Path(directory) / "annotated.jsonl"
            source.write_text(json.dumps(raw("text\u2028<<1+1=2>>\u2029", question="q\u0085q"), ensure_ascii=False) + "\n", encoding="utf-8")
            process = self.run_cli("--input", source, "--output", output)
            self.assertEqual(process.returncode, 0, process.stderr)
            with output.open(encoding="utf-8") as handle:
                records = [json.loads(line) for line in handle]
            self.assertEqual(len(records), 1)
            self.assertEqual(records[0]["rationale_raw"], "text\u2028<<1+1=2>>\u2029")

    def test_cli_refuses_to_overwrite_input(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source.jsonl"
            source.write_text(json.dumps(raw("x=1.")) + "\n", encoding="utf-8")
            process = self.run_cli("--input", source, "--output", source)
            self.assertNotEqual(process.returncode, 0)
            self.assertIn("distinct files", process.stderr)


if __name__ == "__main__":
    unittest.main()
