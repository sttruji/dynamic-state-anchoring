import copy
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

from dsa.data.adapters import canonical_record
from dsa.provenance import (
    DATA_SOURCE_COMMIT, SplitOverlapError, build_split_manifest,
    file_sha256, load_lotus_array, verify_pinned_download_script,
)


REPOSITORY = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("prepare_dsa_data", REPOSITORY / "scripts/prepare_dsa_data.py")
prepare_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prepare_module)


class ProvenanceTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.preprocessing = {name: REPOSITORY / "preprocessing" / name
                              for name in ("gsm_icot.bash", "gsm_icot.py")}
        self.rows = {
            "train": [{"question": "Train one?", "steps": ["<<1+2=3>>", "", "<<3+4=7>>"], "answer": "7", "id": "train-1"},
                      {"question": "Train one?", "steps": ["x = 3."], "answer": "7", "id": "train-1"}],
            "valid": [{"question": "Valid one?", "steps": ["No equations here."], "answer": "8"}],
            "test": [{"question": " Test two? ", "steps": ["Final."], "answer": " 1,000 "},
                     {"question": "Test one?", "steps": ["<<2+2=4>>"], "answer": "4"}],
        }
        self.paths = {split: self.root / f"{split}.json" for split in self.rows}
        self.save()

    def save(self):
        for split, rows in self.rows.items():
            self.paths[split].write_text(json.dumps(rows, ensure_ascii=False), encoding="utf-8")

    def manifest(self, **kwargs):
        return build_split_manifest(self.paths, lotus_sha="a" * 40,
                                    preprocessing_paths=self.preprocessing, **kwargs)

    def prepare(self, output, **kwargs):
        return prepare_module.prepare_data(self.paths, output, lotus_sha="a" * 40,
                                           preprocessing_paths=self.preprocessing, **kwargs)

    def test_counts_groups_hashes_and_ordered_eval(self):
        manifest = self.manifest()
        self.assertEqual(manifest["splits"]["train"]["rows"], 2)
        self.assertEqual(manifest["splits"]["train"]["unique_questions"], 1)
        self.assertEqual(manifest["splits"]["train"]["unique_problem_groups"], 1)
        samples = manifest["splits"]["train"]["problem_groups"]["train-1"]
        self.assertEqual(len(set(samples)), 2)
        expected = [canonical_record(row)["sample_id"] for row in self.rows["test"]]
        self.assertEqual(manifest["ordered_evaluation_ids"], expected)
        self.assertEqual(manifest["ordered_evaluation_records"][0]["answer_normalized"], "1000")
        self.assertEqual(manifest["splits"]["test"]["original_file_sha256"], file_sha256(self.paths["test"]))
        self.assertEqual(manifest["data_source_commit"], DATA_SOURCE_COMMIT)
        self.assertEqual(manifest["official_test_comparison"]["status"], "not_run")

    def test_explicit_group_collision_stops(self):
        self.rows["valid"][0]["id"] = "train-1"
        self.save()
        with self.assertRaises(SplitOverlapError) as caught:
            self.manifest()
        self.assertEqual(caught.exception.overlaps[0]["source_problem_ids"], ["train-1"])

    def test_normalized_question_collision_even_with_different_ids(self):
        self.rows["valid"][0].update(question="  TRAIN   ONE? ", id="different-id")
        self.save()
        with self.assertRaises(SplitOverlapError) as caught:
            self.manifest()
        self.assertEqual(len(caught.exception.overlaps[0]["question_hashes"]), 1)

    def test_overlap_creates_no_output(self):
        self.rows["test"][0]["question"] = self.rows["train"][0]["question"]
        self.save()
        output = self.root / "prepared"
        with self.assertRaises(SplitOverlapError):
            self.prepare(output)
        self.assertFalse(output.exists())

    def test_lossless_conversion_policies_and_no_event_retention(self):
        original_hashes = {split: file_sha256(path) for split, path in self.paths.items()}
        original_rows = copy.deepcopy(self.rows)
        output = self.root / "prepared"
        manifest = self.prepare(output)
        for split, rows in self.rows.items():
            copied = [json.loads(line) for line in (output / f"{split}.original.jsonl").read_text().splitlines()]
            self.assertEqual(copied, rows)
            self.assertEqual(file_sha256(self.paths[split]), original_hashes[split])
        records = [json.loads(line) for line in (output / "train.conclusive.jsonl").read_text().splitlines()]
        self.assertEqual(records[0]["rationale_raw"], "<<1+2=3>>  <<3+4=7>>")
        self.assertEqual(records[0]["original_record"], original_rows["train"][0])
        self.assertIn("<SA>", records[1]["rationale_marked"])
        verified = [json.loads(line) for line in (output / "train.verified_only.jsonl").read_text().splitlines()]
        self.assertEqual(verified[1]["event_count"], 0)
        valid_report = json.loads((output / "valid.conclusive.report.json").read_text())
        self.assertEqual(valid_report["record_count"], 1)
        self.assertEqual(len(manifest["ordered_evaluation_ids"]), len(original_rows["test"]))
        self.assertTrue((output / "split_manifest.json").exists())
        for name, sha in manifest["derivative_sha256"].items():
            self.assertEqual(file_sha256(output / name), sha)

    def test_bad_annotation_creates_no_output(self):
        self.rows["train"][0]["question"] += " <SA>"
        self.save()
        output = self.root / "prepared"
        with self.assertRaises(ValueError):
            self.prepare(output)
        self.assertFalse(output.exists())

    def test_requires_empty_output(self):
        output = self.root / "prepared"
        output.mkdir()
        (output / "keep.txt").write_text("keep")
        with self.assertRaises(ValueError):
            self.prepare(output)
        self.assertEqual((output / "keep.txt").read_text(), "keep")

    def test_rejects_lfs_pointer(self):
        self.paths["train"].write_text("version https://git-lfs.github.com/spec/v1\noid sha256:abc\nsize 100\n")
        with self.assertRaisesRegex(ValueError, "LFS pointer"):
            load_lotus_array(self.paths["train"])

    def test_rejects_array_and_row_shape_errors(self):
        for value in ({"question": "not array"}, [42], [{"question": "Q", "steps": "not list", "answer": "1"}]):
            with self.subTest(value=value):
                self.paths["train"].write_text(json.dumps(value))
                with self.assertRaises(ValueError):
                    load_lotus_array(self.paths["train"])

    def test_dataset_commit_discrepancy_stops(self):
        with self.assertRaisesRegex(ValueError, "download URLs differ"):
            self.manifest(data_source_commit="b" * 40)

    def test_pin_verified_without_claiming_payload_verification(self):
        result = verify_pinned_download_script(self.preprocessing["gsm_icot.bash"])
        self.assertTrue(result["source_code_verified"])
        self.assertEqual(len(result["urls"]), 3)
        self.assertEqual(result["http_payload_verification"], "not_performed_by_manifest_builder")

    def test_official_test_comparison_detects_order_and_answer_discrepancy(self):
        official = self.root / "official.jsonl"
        official.write_text("\n".join(json.dumps({"question": row["question"], "answer": "rationale\n#### " + row["answer"]})
                                       for row in reversed(self.rows["test"])) + "\n")
        result = self.manifest(official_test_path=official)["official_test_comparison"]
        self.assertTrue(result["question_membership_match"])
        self.assertFalse(result["ordered_question_answer_match"])
        self.assertEqual(result["answer_mismatch_question_hashes"], [])
        official.write_text(json.dumps({"question": self.rows["test"][0]["question"], "answer": "rationale#### 999"}) + "\n")
        result = self.manifest(official_test_path=official)["official_test_comparison"]
        self.assertEqual(len(result["answer_mismatch_question_hashes"]), 1)
        self.assertEqual(len(result["supplied_only_question_hashes"]), 1)

    def test_official_jsonl_unicode_separator_is_inside_record(self):
        self.rows["test"][0]["question"] = "Question\u2028with paragraph\u2029separator?"
        self.save()
        official = self.root / "official.jsonl"
        official.write_text("\n".join(json.dumps({"question": row["question"],
                          "answer": "rationale\u2028text#### " + row["answer"]}, ensure_ascii=False)
                          for row in self.rows["test"]) + "\n", encoding="utf-8")
        result = self.manifest(official_test_path=official)["official_test_comparison"]
        self.assertTrue(result["ordered_question_answer_match"])


if __name__ == "__main__":
    unittest.main()
