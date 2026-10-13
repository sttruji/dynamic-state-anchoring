import copy
import unittest

from dsa.data.adapters import canonical_record, normalized_question_hash, stable_json_hash


class CanonicalRecordTests(unittest.TestCase):
    def test_raw_solution_preserves_question_and_rationale(self):
        source = {"question": "  Café?\n", "answer": " \n<<1+2=3>>3.\n####  3\n", "extra": ["untouched"]}
        snapshot = copy.deepcopy(source)
        result = canonical_record(source)
        self.assertEqual(result["question"], source["question"])
        self.assertEqual(result["rationale_raw"], " \n<<1+2=3>>3.\n")
        self.assertEqual(result["final_answer"], "3")
        self.assertEqual(source, snapshot)
        self.assertEqual(result["original_record"], source)
        result["original_record"]["extra"].append("safe copy")
        self.assertEqual(source, snapshot)

    def test_lotus_space_fragments_reconstruct_without_collapsing(self):
        source = {"question": "Q", "steps": ["First", "", "<<1+2=3>>3.\n"], "answer": " 3 "}
        result = canonical_record(source)
        self.assertEqual(result["rationale_raw"], "First  <<1+2=3>>3.\n")
        self.assertEqual(result["final_answer"], " 3 ")
        self.assertEqual(canonical_record(source, "newline")["rationale_raw"], "First\n\n<<1+2=3>>3.\n")

    def test_duplicate_format_groups_and_alternate_rationale_ids(self):
        first = canonical_record({"question": " How  MANY? ", "steps": ["x = 2"], "answer": "2"})
        second = canonical_record({"question": "how many?", "steps": ["y = 2"], "answer": "2"})
        self.assertEqual(first["question_hash"], second["question_hash"])
        self.assertEqual(first["source_problem_id"], second["source_problem_id"])
        self.assertNotEqual(first["sample_id"], second["sample_id"])
        self.assertEqual(normalized_question_hash("Ａ\tB"), normalized_question_hash("a b"))

    def test_explicit_source_problem_id_is_shared(self):
        first = canonical_record({"id": 17, "question": "Q", "answer": "one\n#### 1"})
        second = canonical_record({"id": 17, "question": "Q", "answer": "different\n#### 1"})
        self.assertEqual(first["source_problem_id"], "17")
        self.assertEqual(first["source_problem_id"], second["source_problem_id"])
        self.assertNotEqual(first["sample_id"], second["sample_id"])

    def test_json_hash_key_order_independent_and_sensitive_to_content(self):
        self.assertEqual(stable_json_hash({"b": 1, "a": 2}), stable_json_hash({"a": 2, "b": 1}))
        self.assertNotEqual(stable_json_hash("a b"), stable_json_hash("a  b"))

    def test_raw_separator_and_invalid_source_rejected(self):
        for answer in ("no delimiter", "one #### 1 #### 2", 12):
            with self.subTest(answer=answer), self.assertRaises(ValueError):
                canonical_record({"question": "Q", "answer": answer})
        for steps in (["text", 1], None, {}):
            with self.subTest(steps=steps), self.assertRaises(ValueError):
                canonical_record({"question": "Q", "steps": steps, "answer": "2"})
        with self.assertRaises(ValueError):
            canonical_record({"question": "Q", "steps": [], "answer": "2"}, "tab")

    def test_canonical_records_can_be_reused(self):
        first = canonical_record({"question": "Q", "answer": "x = 2\n#### 2"})
        second = canonical_record(first)
        for field in ("question", "rationale_raw", "final_answer", "sample_id", "original_sha256", "original_record"):
            self.assertEqual(first[field], second[field])


if __name__ == "__main__":
    unittest.main()
