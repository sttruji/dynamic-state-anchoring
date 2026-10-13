"""Offline checks for the unmodified-checkpoint smoke runner, using random tiny GPT-2."""

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from scripts.smoke_pretrained_dsa import PIN, smoke_checkpoint


torch.set_num_threads(1)


class PretrainedSmokeTests(unittest.TestCase):
    def run_tiny(self, path, max_new_tokens=2):
        # Tiny smoke must never call the Hub or imply pretrained validation.
        with patch("transformers.AutoModelForCausalLM.from_pretrained", side_effect=AssertionError("offline smoke downloaded model")), \
             patch("transformers.AutoTokenizer.from_pretrained", side_effect=AssertionError("offline smoke downloaded tokenizer")):
            return smoke_checkpoint("yingfanbot/gsm-cot-gpt2", PIN, path, device="cpu", tiny=True,
                                    max_new_tokens=max_new_tokens)

    def test_tiny_smoke_checks_unmodified_inference_and_records_no_benchmark_claim(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "nested/smoke.json"
            report = self.run_tiny(path)
            self.assertTrue(report["passed"])
            self.assertTrue(report["tiny"])
            self.assertTrue(report["finite_logits"])
            self.assertFalse(report["vocabulary_changed"])
            self.assertFalse(report["anchor_wrapper_used"])
            self.assertFalse(report["model_weights_downloaded"])
            self.assertEqual(report["original_vocab_size"], 259)
            self.assertEqual(report["device"], "cpu")
            self.assertEqual(report["input_question"], "What is 2 plus 3?\n")
            self.assertIn("not a benchmark", report["accuracy_claim"])
            self.assertEqual(json.loads(path.read_text()), report)
            self.assertEqual(list(path.parent.iterdir()), [path])

    def test_matching_verified_smoke_is_reused_without_rebuilding_or_touching_file(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "smoke.json"
            first = self.run_tiny(path)
            original = path.read_bytes()
            with patch("dsa.models.backbone.BackboneAdapter.tiny", side_effect=AssertionError("matching cache rebuilt model")):
                reused = self.run_tiny(path)
            self.assertTrue(reused["reused_verified_smoke"])
            self.assertEqual({key: value for key, value in reused.items() if key != "reused_verified_smoke"}, first)
            self.assertEqual(path.read_bytes(), original)

    def test_different_generation_cap_rebuilds_cache_and_updates_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "smoke.json"
            first = self.run_tiny(path, max_new_tokens=1)
            second = self.run_tiny(path, max_new_tokens=2)
            self.assertNotEqual(first["identity"], second["identity"])
            self.assertEqual(second["identity"]["max_new_tokens"], 2)
            self.assertFalse(second.get("reused_verified_smoke", False))
            self.assertEqual(json.loads(path.read_text()), second)

    def test_failed_report_is_not_accepted_as_a_verified_cache(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "smoke.json"
            first = self.run_tiny(path)
            first["passed"] = False
            path.write_text(json.dumps(first))
            report = self.run_tiny(path)
            self.assertTrue(report["passed"])
            self.assertFalse(report.get("reused_verified_smoke", False))

    def test_real_checkpoint_requires_full_immutable_revision_before_download(self):
        for revision in ("main", "v1", "a" * 39, "a" * 41, "z" * 40):
            with self.subTest(revision=revision), tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "smoke.json"
                with patch("transformers.AutoModelForCausalLM.from_pretrained") as model_download, \
                     patch("transformers.AutoTokenizer.from_pretrained") as tokenizer_download:
                    with self.assertRaisesRegex(ValueError, "immutable"):
                        smoke_checkpoint("yingfanbot/gsm-cot-gpt2", revision, path, device="cpu", tiny=False)
                    model_download.assert_not_called()
                    tokenizer_download.assert_not_called()
                self.assertFalse(path.exists())

    def test_nonpositive_generation_caps_are_rejected_without_output(self):
        for cap in (0, -1):
            with self.subTest(cap=cap), tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "smoke.json"
                with self.assertRaisesRegex(ValueError, "positive"):
                    self.run_tiny(path, max_new_tokens=cap)
                self.assertFalse(path.exists())


if __name__ == "__main__":
    unittest.main()
