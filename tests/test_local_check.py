"""Exercise the local four-policy check and its matched initialization contract."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest

from dsa.data.pilot_prepare import prepare_pilot
from dsa.training.pilot import sha256_file
from scripts.check_local_dsa import check_local


class LocalCheckTests(unittest.TestCase):
    def test_matched_policy_updates_preserve_source_and_record_only_feasibility(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            source.mkdir()
            for split in ("train", "valid", "test"):
                (source / f"gsm_{split}.json").write_text(json.dumps([{
                    "question": f"{split}: 2 groups of 3 apples, then eat 1. How many remain?",
                    "steps": ["<<2*3=6>>", "<<6-1=5>>"], "answer": "5"}]))
            prepared = root / "prepared"
            prepare_pilot(storage_dir=None, source_dir=source, output_dir=prepared,
                          train_limit=0, loops=2, tiny=True)
            source_hashes = {p.name: sha256_file(p) for p in source.iterdir()}
            output = root / "check"
            with contextlib.redirect_stdout(io.StringIO()):
                result = check_local(prepared, output, device="cpu", steps=2)
            self.assertTrue(result["complete"])
            self.assertFalse(result["validation_or_test_evaluated"])
            self.assertEqual(set(result["cases"]), {"static", "every_loop", "periodic", "learned"})
            self.assertEqual(len({x["initial_state_sha256"] for x in result["cases"].values()}), 1)
            for policy, case in result["cases"].items():
                self.assertEqual(case["optimizer_updates"], 2)
                self.assertEqual(case["metrics_sha256"], sha256_file(output / f"{policy}.jsonl"))
                rows = [json.loads(row) for row in (output / f"{policy}.jsonl").read_text().splitlines()]
                self.assertEqual([row["sample_id"] for row in rows], result["sample_ids"])
                self.assertGreater(case["median_gradient_norm_before_clipping"], 0)
            self.assertEqual(source_hashes, {p.name: sha256_file(p) for p in source.iterdir()})
            with self.assertRaisesRegex(ValueError, "empty output"):
                check_local(prepared, output, device="cpu", steps=2)
