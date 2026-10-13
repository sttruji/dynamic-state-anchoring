"""Coverage and artifact checks; these do not introduce annotation rules."""

import json
from pathlib import Path
import tempfile
import unittest

from dsa.data.milestones import annotate_record
from dsa.provenance import file_sha256
from scripts.audit_lotus_annotations import Coverage, ReviewPool, iter_json_array, run_audit


class CorpusAnnotationAuditTests(unittest.TestCase):
    def test_incremental_array_decoder_preserves_unicode_and_rejects_bad_shapes(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "source.json"
            records = [{"question": "🧮 Q\n\u2028", "steps": ["a", ""], "answer": "2"}, {"question": "B"}]
            path.write_text(json.dumps(records, ensure_ascii=False), encoding="utf-8")
            self.assertEqual(list(iter_json_array(path, chunk_chars=3)), records)
            for invalid in ("{}", "[1]", "[{},]", "[{}] trailing", "[{", "[NaN]"):
                path.write_text(invalid)
                with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                    list(iter_json_array(path, chunk_chars=3))

    def test_question_coverage_counts_any_and_all_rationales(self):
        counter = Coverage()
        sources = [{"question": "Q", "steps": ["No event"], "answer": "99"},
                   {"question": "Q", "steps": ["<<1+1=2>>"], "answer": "99"},
                   {"question": "R", "steps": ["<<1+1=2>> " * 5], "answer": "99"}]
        for source in sources:
            record = annotate_record(source)
            counter.add(source, record, json.dumps(record))
        result = counter.report()
        self.assertEqual(result["record_count"], 3)
        self.assertEqual(result["unique_questions"], 2)
        self.assertEqual(result["questions_with_no_events_in_any_rationale"], 0)
        self.assertEqual(result["questions_with_at_least_one_no_event_rationale"], 1)
        self.assertEqual(result["controller_coverage"]["4"]["eligible_rows"], 1)
        self.assertEqual(result["controller_coverage"]["4"]["masked_MgtK_rows"], 1)
        self.assertEqual(result["controller_coverage"]["8"]["questions_with_all_rationales_eligible"], 1)

    def test_review_is_deterministic_and_groups_do_not_repeat(self):
        sources = [{"question": str(i), "steps": ["<<1+1=2>>" if i % 2 else "Nothing"], "answer": "99"}
                   for i in range(20)]
        results = []
        for ordering in (sources, sources[::-1]):
            pool = ReviewPool(7, max_per_stratum=12)
            for index, source in enumerate(ordering):
                pool.add(index, annotate_record(source))
            selected = pool.select(16)
            self.assertEqual(len({row["question_hash"] for row in selected}), len(selected))
            results.append([(row["selection_stratum"], row["sample_id"]) for row in selected])
        self.assertEqual(results[0], results[1])

    def test_gate_precedes_derivatives_and_originals_are_lossless(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            paths = {}
            manifest = {"split_overlap_audit": {"status": "failed"}, "splits": {}}
            for split in ("train", "valid", "test"):
                paths[split] = directory / f"{split}.json"
                paths[split].write_text(json.dumps([{"question": split, "steps": ["<<1+1=2>>"], "answer": "99"}]))
                manifest["splits"][split] = {"original_file_sha256": file_sha256(paths[split]), "rows": 1}
            manifest_path = directory / "manifest.json"
            manifest_path.write_text(json.dumps(manifest))
            output = directory / "output"
            with self.assertRaisesRegex(ValueError, "passed primary"):
                run_audit(paths, manifest_path, output, directory / "report.json")
            self.assertFalse(output.exists())
            manifest["split_overlap_audit"]["status"] = "passed"
            manifest_path.write_text(json.dumps(manifest))
            result = run_audit(paths, manifest_path, output, directory / "report.json")
            self.assertEqual(result["review_sample"]["selected_count"], 1)
            for split, source_path in paths.items():
                derivative = [json.loads(line) for line in (output / f"{split}.original.jsonl").read_text().splitlines()]
                self.assertEqual(derivative, json.loads(source_path.read_text()))
                self.assertEqual(result["policies"]["conclusive"][split]["event_count"], 1)


if __name__ == "__main__":
    unittest.main()
