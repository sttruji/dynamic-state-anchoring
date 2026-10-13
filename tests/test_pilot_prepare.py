import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from dsa.data.pilot_prepare import (DEFAULT_REVISION, convert_upstream, directory_sha256,
                                    download_verified, load_pinned_manifest, prepare_pilot, select_indices)


class Response(io.BytesIO):
    status = 200


class PilotPreparationTests(unittest.TestCase):
    def fixture(self, root, count=10):
        root.mkdir()
        for split in ("train", "valid", "test"):
            rows = [{"question": f"{split} question {index}?", "steps": ["Reason without equations."], "answer": str(index)}
                    for index in range(count if split == "train" else 2)]
            if split == "train":
                rows[1]["steps"] = ["<<1+1=2>> <<2+1=3>> <<3+1=4>>"]
                rows[1]["answer"] = "5"
            (root / f"gsm_{split}.json").write_text(json.dumps(rows), encoding="utf-8")
        return root

    def test_verified_download_reuses_and_rejects_changed_payload(self):
        payload = b"source bytes\n"
        expected = {"bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()}
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "raw/train.txt"
            with patch("urllib.request.urlopen", return_value=Response(payload)) as request:
                self.assertTrue(download_verified("https://example.invalid/data", path, expected))
                self.assertFalse(download_verified("https://example.invalid/data", path, expected))
                self.assertEqual(request.call_count, 1)
            path.write_bytes(b"different")
            with patch("urllib.request.urlopen") as request:
                with self.assertRaisesRegex(ValueError, "preserved"):
                    download_verified("https://example.invalid/data", path, expected)
                request.assert_not_called()
            self.assertEqual(path.read_bytes(), b"different")

    def test_invalid_download_is_not_installed_and_staging_is_removed(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "train.txt"
            with patch("urllib.request.urlopen", return_value=Response(b"wrong")):
                with self.assertRaisesRegex(ValueError, "checksum/size"):
                    download_verified("https://example.invalid/data", path, {"bytes": 5, "sha256": "0" * 64})
            self.assertFalse(path.exists())
            self.assertEqual(list(path.parent.iterdir()), [])

    def test_conversion_matches_original_json_bytes_and_preserves_spaces(self):
        payload = "café question ||  a  b  ## 9\n".encode()
        rows = [{"question": "café question ", "steps": ["a", "", "b"], "answer": "9"}]
        wanted = json.dumps(rows).encode()
        expected = {"json_bytes": len(wanted), "json_sha256": hashlib.sha256(wanted).hexdigest()}
        with tempfile.TemporaryDirectory() as temporary:
            raw, output = Path(temporary) / "raw.txt", Path(temporary) / "data.json"
            raw.write_bytes(payload)
            self.assertTrue(convert_upstream(raw, output, expected))
            self.assertEqual(output.read_bytes(), wanted)
            self.assertFalse(convert_upstream(raw, output, expected))

    def test_deterministic_train_selection_and_full_eval_order(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = self.fixture(root / "source")
            first = prepare_pilot(source_dir=source, storage_dir=None, output_dir=root / "first", tiny=True, train_limit=4, seed=7)
            second = prepare_pilot(source_dir=source, storage_dir=None, output_dir=root / "second", tiny=True, train_limit=4, seed=7)
            third = prepare_pilot(source_dir=source, storage_dir=None, output_dir=root / "third", tiny=True, train_limit=4, seed=8)
            self.assertEqual(first["cache_identity_sha256"], second["cache_identity_sha256"])
            self.assertNotEqual(first["cache_identity_sha256"], third["cache_identity_sha256"])
            selection = json.loads((root / "first/selections.json").read_text())
            self.assertEqual([row["source_row_number"] for row in selection["train"]["selection"]], [i + 1 for i in select_indices(10, 4, 7)])
            self.assertEqual([row["source_row_number"] for row in selection["test"]["selection"]], [1, 2])
            self.assertEqual(first["source_provenance"]["regime"], "local_fixture")
            self.assertFalse(first["metadata"]["benchmark_result"])
            for split in ("valid", "test"):
                records = [json.loads(line) for line in (root / f"first/{split}.jsonl").read_text().splitlines()]
                self.assertTrue(all("events" not in record and "gate_targets" not in record for record in records))
                self.assertEqual([record["question"] for record in records], [f"{split} question 0?", f"{split} question 1?"])

    def test_no_event_and_overflow_are_retained_as_distinct_masks(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = self.fixture(root / "source", count=2)
            manifest = prepare_pilot(source_dir=source, storage_dir=None, output_dir=root / "prepared", tiny=True, train_limit=0, loops=1)
            records = [json.loads(line) for line in (root / "prepared/train.tokenized.jsonl").read_text().splitlines()]
            self.assertEqual(len(records), 2)
            self.assertEqual([record["controller_mask_reason"] for record in records], ["no_events", "event_overflow"])
            self.assertTrue(all(record["answer_input_ids"] for record in records))
            self.assertEqual(manifest["splits"]["train"]["records"], 2)
            report = json.loads((root / "prepared/preparation_report.json").read_text())["train"]
            self.assertEqual(report["controller_mask_reasons"], {"no_events": 1, "event_overflow": 1})

    def test_reuse_never_reinitializes_tokenizer_and_refuses_stale_parameters(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = self.fixture(root / "source")
            kwargs = dict(source_dir=source, storage_dir=None, output_dir=root / "prepared", tiny=True)
            first = prepare_pilot(**kwargs)
            with patch("dsa.data.pilot_prepare.TinyTokenizer", side_effect=AssertionError("cache must not rebuild")):
                reused = prepare_pilot(**kwargs)
            self.assertEqual(first, reused)
            with self.assertRaisesRegex(ValueError, "stale.*new --output-dir"):
                prepare_pilot(**kwargs, loops=8)
            self.assertEqual(first["tokenizer_sha256"], directory_sha256(root / "prepared/tokenizer"))

    def test_reuse_rejects_corrupt_data_or_tokenizer(self):
        for file in ("train.tokenized.jsonl", "tokenizer/tiny_tokenizer.json"):
            with self.subTest(file=file), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                source = self.fixture(root / "source")
                kwargs = dict(source_dir=source, storage_dir=None, output_dir=root / "prepared", tiny=True)
                prepare_pilot(**kwargs)
                (root / "prepared" / file).write_text("changed")
                with self.assertRaisesRegex(ValueError, "stale or invalid"):
                    prepare_pilot(**kwargs)

    def test_source_mutation_and_overlap_refused_without_replacing_artifacts(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = self.fixture(root / "source")
            kwargs = dict(source_dir=source, storage_dir=None, output_dir=root / "prepared", tiny=True)
            first = prepare_pilot(**kwargs)
            original = (root / "prepared/manifest.json").read_bytes()
            train = source / "gsm_train.json"
            rows = json.loads(train.read_text())
            rows[0]["answer"] = "changed"
            train.write_text(json.dumps(rows))
            with self.assertRaisesRegex(ValueError, "stale or invalid"):
                prepare_pilot(**kwargs)
            self.assertEqual((root / "prepared/manifest.json").read_bytes(), original)
            rows[0]["question"] = "valid question 0?"
            train.write_text(json.dumps(rows))
            with self.assertRaisesRegex(ValueError, "cross-partition"):
                prepare_pilot(**{**kwargs, "output_dir": root / "other"})
            self.assertFalse((root / "other").exists())

    def test_pinned_sources_and_preprocessing_are_verified_without_checkouts(self):
        manifest = load_pinned_manifest()
        self.assertEqual(manifest["splits"]["train"]["bytes"], 87805358)
        self.assertEqual(manifest["splits"]["test"]["json_sha256"], "46ce5b0c44fde0bfc8cf1e7492aa640b96111d9dcb80a54340d9911141a9caa9")

    def test_mutable_revision_rejected(self):
        with self.assertRaisesRegex(ValueError, "immutable"):
            prepare_pilot(storage_dir=None, output_dir=Path("unused"), tiny=True, revision="main")


if __name__ == "__main__":
    unittest.main()
