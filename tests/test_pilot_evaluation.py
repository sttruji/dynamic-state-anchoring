"""Persistent tiny train/evaluate integration and evaluation corruption guards."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import torch

from dsa.data.pilot_prepare import prepare_pilot
from dsa.training.pilot import PilotConfig, train_pilot
from dsa.evaluation.evaluate import predict_question
from scripts.evaluate_dsa import evaluate_run, resolve_checkpoint, _load_model


torch.set_num_threads(1)


class PilotEvaluationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).resolve()
        source = self.root / "source"
        source.mkdir()
        for split, count in (("train", 2), ("valid", 2), ("test", 3)):
            rows = [{"question": f"{split} example {index}: add 2 and 3.",
                     "steps": ["<<2+3=5>>"], "answer": "5"} for index in range(count)]
            (source / f"gsm_{split}.json").write_text(json.dumps(rows))
        self.prepared = self.root / "prepared"
        prepare_pilot(storage_dir=None, source_dir=source, output_dir=self.prepared,
                      train_limit=0, loops=2, tiny=True)
        self.run = self.root / "run"
        with contextlib.redirect_stdout(io.StringIO()):
            self.training = train_pilot(PilotConfig(self.prepared, self.run, tiny=True,
                                                   loops=2, steps=1, save_every=1, device="cpu"))
        self.output = self.root / "evaluation"

    def evaluate(self, **kwargs):
        return evaluate_run(self.prepared, self.run, self.output, max_new_tokens=2,
                            device="cpu", **kwargs)

    def predictions(self):
        return [json.loads(line) for line in
                (self.output / "predictions.jsonl").read_text().splitlines()]

    def test_tiny_training_evaluation_and_complete_cache(self):
        report = self.evaluate()
        self.assertTrue(report["complete"])
        self.assertEqual(report["total"], 2)
        self.assertEqual(report["executed_recurrent_forwards"], 4)
        self.assertTrue(report["all_source_examples_retained"])
        self.assertTrue(report["question_only_inference"])
        self.assertFalse(report["annotation_filtering"])
        self.assertEqual([row["index"] for row in self.predictions()], [0, 1])
        self.assertEqual(len(report["identity"]["checkpoint_files_sha256"]), 8)
        model = _load_model(Path(self.training["checkpoint"]) / "model")
        self.assertNotIn("sdpa", type(model.backbone.model.transformer.h[0].attn).__name__.lower())
        with mock.patch("scripts.evaluate_dsa._load_model", side_effect=AssertionError("cache reload")):
            cached = self.evaluate()
        self.assertTrue(cached["reused_completed_evaluation"])
        self.assertEqual(cached["predictions_sha256"], report["predictions_sha256"])

    def test_interrupted_predictions_with_torn_utf8_resume(self):
        call_count = 0
        def interrupt_after_one(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count > 1:
                raise RuntimeError("disconnected")
            return predict_question(*args, **kwargs)
        with mock.patch("dsa.evaluation.evaluate.predict_question", side_effect=interrupt_after_one):
            with self.assertRaisesRegex(RuntimeError, "disconnected"):
                self.evaluate()
        journal = self.output / "predictions.jsonl"
        first = journal.read_bytes()
        with journal.open("ab") as handle:
            handle.write(b'{"index":1,"generated_text":"\xe2\x82')
        report = self.evaluate()
        self.assertEqual(report["total"], 2)
        self.assertTrue(journal.read_bytes().startswith(first))
        self.assertEqual([row["index"] for row in self.predictions()], [0, 1])

    def test_valid_json_without_final_newline_is_retained_before_append(self):
        self.evaluate()
        (self.output / "report.json").unlink()
        journal = self.output / "predictions.jsonl"
        first = journal.read_bytes().splitlines()[0]
        journal.write_bytes(first)
        self.evaluate()
        self.assertTrue(journal.read_bytes().startswith(first + b"\n"))
        self.assertEqual([row["index"] for row in self.predictions()], [0, 1])

    def test_prediction_log_cannot_exceed_denominator(self):
        self.evaluate()
        (self.output / "report.json").unlink()
        extra = {**self.predictions()[-1], "index": 2}
        with (self.output / "predictions.jsonl").open("a") as handle:
            handle.write(json.dumps(extra) + "\n")
        with self.assertRaisesRegex(ValueError, "exceeds.*denominator"):
            self.evaluate()

    def test_completed_report_checks_prediction_bytes(self):
        self.evaluate()
        with (self.output / "predictions.jsonl").open("ab") as handle:
            handle.write(b" ")
        with self.assertRaisesRegex(ValueError, "prediction checksum"):
            self.evaluate()

    def test_changed_checkpoint_bytes_refuse_cached_evaluation(self):
        self.evaluate()
        checkpoint = resolve_checkpoint(self.run)
        with (checkpoint / "model/anchor.pt").open("ab") as handle:
            handle.write(b"extra bytes")
        with self.assertRaisesRegex(ValueError, "checkpoint bytes"):
            self.evaluate()

    def test_evaluation_resume_refuses_changed_hardware_identity(self):
        report = self.evaluate()
        self.assertIn("torch", report["identity"]["runtime"])
        self.assertIn("transformers", report["identity"]["runtime"])
        with mock.patch("scripts.evaluate_dsa.platform.processor", return_value="different CPU"):
            with self.assertRaisesRegex(ValueError, "different settings"):
                self.evaluate()

    def test_prepared_identity_and_completion_metadata_are_validated(self):
        manifest_path = self.prepared / "manifest.json"
        original = manifest_path.read_bytes()
        manifest = json.loads(original)
        manifest["metadata"]["seed"] = 999
        manifest_path.write_text(json.dumps(manifest))
        with self.assertRaisesRegex(ValueError, "dataset identity"):
            self.evaluate()
        manifest_path.write_bytes(original)
        complete = resolve_checkpoint(self.run) / "COMPLETE.json"
        completion = json.loads(complete.read_text())
        completion["step"] = 88
        complete.write_text(json.dumps(completion))
        with self.assertRaisesRegex(ValueError, "step disagree"):
            self.evaluate()

    def test_final_test_requires_explicit_full_denominator(self):
        with self.assertRaisesRegex(ValueError, "complete test denominator"):
            self.evaluate(split="test")
        with self.assertRaisesRegex(ValueError, "complete test denominator"):
            self.evaluate(split="test", final_test=True, limit=1)
        report = self.evaluate(split="test", final_test=True)
        self.assertEqual(report["total"], 3)
        self.assertTrue(report["all_source_examples_retained"])
        self.assertEqual([row["index"] for row in self.predictions()], [0, 1, 2])

    def test_saved_scoring_cannot_silently_change_on_resume(self):
        self.evaluate()
        (self.output / "report.json").unlink()
        rows = self.predictions()
        rows[0]["correct"] = not rows[0]["correct"]
        (self.output / "predictions.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
        with self.assertRaisesRegex(ValueError, "scoring fields"):
            self.evaluate()


if __name__ == "__main__":
    unittest.main()
