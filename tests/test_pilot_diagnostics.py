"""Meaningful checks for checkpoint-only generation and gradient instrumentation."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import torch
from dsa.data.pilot_prepare import prepare_pilot
from dsa.data.tokenize import TinyTokenizer, tokenize_record
from dsa.data.milestones import annotate_record
from dsa.models.anchored_loop import AnchorConfig, AnchoredLoop
from dsa.models.backbone import BackboneAdapter
from dsa.training.objectives import LossConfig, joint_objective
from dsa.training.pilot import PilotConfig, train_pilot, _seed, _state_digest
from scripts.diagnose_dsa_pilot import (diagnose, gradient_probe, inference_condition,
                                      parameter_gradients, predict_record)

torch.set_num_threads(1)
ROOT = Path(__file__).resolve().parents[1]


class DiagnosticNotebookTests(unittest.TestCase):
    def test_standalone_notebook_is_parseable_and_keeps_original_run_selected(self):
        notebook = json.loads((ROOT / "notebooks/DSA_Pilot_Diagnostics.ipynb").read_text())
        self.assertEqual(notebook["cells"][0]["id"], "clone")
        sources = {cell["id"]: "".join(cell["source"]) for cell in notebook["cells"]}
        for cell in notebook["cells"]:
            if cell["cell_type"] == "code":
                compile(sources[cell["id"]], cell["id"], "exec")
                self.assertIsNone(cell["execution_count"])
                self.assertEqual(cell["outputs"], [])
        self.assertIn('RUN_NAME = "gpt2-pilot-v1-ba4e69315086"', sources["diagnostic-config"])
        self.assertIn('RUN_ROOT = PROJECT_DIR / "runs" / RUN_NAME', sources["diagnostic-config"])
        self.assertIn('EXPECTED_STEP = 200', sources["diagnostic-config"])
        self.assertNotIn("train_dsa.py", "\n".join(sources.values()))
        self.assertNotIn("prepare_dsa_pilot.py", "\n".join(sources.values()))

    def test_configuration_finds_matching_drive_cache_without_new_run_paths(self):
        notebook = json.loads((ROOT / "notebooks/DSA_Pilot_Diagnostics.ipynb").read_text())
        source = next("".join(c["source"]) for c in notebook["cells"] if c["id"] == "diagnostic-config")
        import hashlib
        with tempfile.TemporaryDirectory() as temp:
            project = Path(temp)
            run = project / "runs/gpt2-pilot-v1-ba4e69315086"
            prepared = project / "prepared/original-cache"
            prepared.mkdir(parents=True)
            payload = b'{"fixture":true}\n'
            (prepared / "manifest.json").write_bytes(payload)
            digest = hashlib.sha256(payload).hexdigest()
            for arm in ("static", "learned"):
                directory = run / arm
                directory.mkdir(parents=True)
                (directory / "run_manifest.json").write_text(json.dumps({"completed_steps": 200,
                    "contract": {"prepared_manifest_sha256": digest}}))
                (directory / "latest.json").write_text('{"step":200}')
            import shutil
            namespace = {"Path": Path, "json": json, "hashlib": hashlib, "shutil": shutil,
                         "PROJECT_DIR": project, "REPO_DIR": ROOT, "REPO_COMMIT": "new-diagnostic-commit"}
            # Stage the VM copy into this temporary fixture instead of /content.
            source = source.replace('Path("/content/dsa-diagnostic-prepared")',
                                    f"Path({str(project / 'vm')!r})")
            with contextlib.redirect_stdout(io.StringIO()):
                exec(compile(source, "diagnostic-config", "exec"), namespace)
            self.assertEqual(namespace["RUN_ROOT"], run)
            self.assertEqual(namespace["PREPARED_DRIVE"], prepared)


class GradientDiagnosticTests(unittest.TestCase):
    def setUp(self):
        self.tokenizer = TinyTokenizer()
        self.record = tokenize_record(annotate_record({"question": "What is twice the sum of 2 and 3?",
            "steps": ["<<2+3=5>>", "<<5*2=10>>"], "answer": "10"}), self.tokenizer, 2)
        self.prefix = self.record["latent_answer_prefix_ids"]
        self.model = AnchoredLoop(BackboneAdapter.tiny(len(self.tokenizer)),
                                 AnchorConfig(loop_budget=2, marker_token_id=2))

    def test_hooks_preserve_original_joint_loss_and_parameter_gradients(self):
        before = _state_digest(self.model)
        _seed(7, torch.device("cpu"))
        self.model.train()
        original = joint_objective(self.model, self.record, self.prefix)
        expected_loss = original["total"].item()
        original["total"].backward()
        expected_gradient = parameter_gradients(self.model)["global_norm_before_clipping"]
        report = gradient_probe(self.model, self.record, self.prefix, 2, LossConfig(), 7, full_objective=True)
        actual = report["losses"]["total"]
        self.assertAlmostEqual(actual["weighted_loss"], expected_loss, places=6)
        self.assertAlmostEqual(actual["parameter_gradients"]["global_norm_before_clipping"], expected_gradient, places=6)
        self.assertEqual(before, _state_digest(self.model))
        self.assertTrue(all(p.grad is None for p in self.model.parameters()))
        self.assertTrue(all(s["connected_to_loss"] for s in actual["states"]))
        self.assertTrue(all(s["workspace_input_gradient_l2"] > 0 for s in actual["states"]))
        self.assertTrue(all(not s["connected_to_loss"] for s in report["losses"]["cot"]["states"]))
        self.assertGreater(report["losses"]["gate"]["weighted_loss"], 0)
        self.assertGreater(report["losses"]["anchor"]["weighted_loss"], 0)
        self.assertEqual(len(self.model.workspace_norm._forward_hooks), 0)

    def test_short_depth_probe_uses_answer_loss_without_rescheduling_training_labels(self):
        labels = list(self.record["gate_targets"])
        report = gradient_probe(self.model, self.record, self.prefix, 1, LossConfig(), 7)
        self.assertEqual(report["loop_budget"], 1)
        self.assertEqual(list(report["losses"]), ["answer"])
        self.assertEqual(self.record["gate_targets"], labels)
        self.assertEqual(self.model.config.loop_budget, 2)

    def test_write_interventions_restore_original_configuration_on_error(self):
        original = self.model.config
        with self.assertRaisesRegex(RuntimeError, "interrupted"):
            with inference_condition(self.model, "no_writes"):
                self.assertEqual(self.model.config.write_mode, "static")
                raise RuntimeError("interrupted")
        self.assertIs(self.model.config, original)

    def test_generation_passes_only_question_ids_and_retains_raw_stop_metadata(self):
        row = {"sample_id": self.record["sample_id"], "question_hash": "fixture", "question": "What is twice the sum of 2 and 3?", "final_answer": "10"}
        with mock.patch.object(self.model, "generate", wraps=self.model.generate) as generate:
            result = predict_record(self.model, self.tokenizer, self.record, row, 0, self.prefix, 2, 3)
        self.assertEqual(generate.call_args.args[0].tolist(), [self.record["question_input_ids"]])
        self.assertEqual(set(generate.call_args.kwargs), {"answer_prefix_ids", "loop_budget", "max_new_tokens"})
        self.assertEqual(result["generated_token_count"], len(result["generated_token_ids"]))
        self.assertIn(result["stop_reason"], {"eos", "token_cap"})


class PersistentDiagnosticTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        root = Path(self.temp.name)
        source = root / "source"
        source.mkdir()
        for split in ("train", "valid", "test"):
            (source / f"gsm_{split}.json").write_text(json.dumps([
                {"question": f"{split}: twice the sum of 2 and 3.",
                 "steps": ["<<2+3=5>>", "<<5*2=10>>"], "answer": "10"}]))
        self.prepared, self.run, self.out = root / "prepared", root / "run", root / "diagnostics"
        prepare_pilot(storage_dir=None, source_dir=source, output_dir=self.prepared, loops=2, tiny=True)
        with contextlib.redirect_stdout(io.StringIO()):
            train_pilot(PilotConfig(self.prepared, self.run, tiny=True, loops=2, steps=1, save_every=1, device="cpu"))

    def diagnose(self, **kwargs):
        with contextlib.redirect_stdout(io.StringIO()):
            return diagnose(self.prepared, self.run, self.out, device="cpu", gradient_limit=1,
                            probe_limit=1, depth_grid=(1, 2), max_new_tokens=2, **kwargs)

    def test_complete_run_cache_and_no_writes_to_training_artifacts(self):
        before = {str(p): p.read_bytes() for p in self.run.rglob("*") if p.is_file()}
        report = self.diagnose()
        after = {str(p): p.read_bytes() for p in self.run.rglob("*") if p.is_file()}
        self.assertEqual(before, after)
        self.assertTrue(report["complete"])
        self.assertTrue(report["model_state_unchanged"])
        self.assertEqual(report["optimizer_updates"], 0)
        self.assertFalse(report["test_or_validation_answers_read"])
        self.assertEqual(report["prediction_reports"]["normal_training_set"]["total"], 1)
        no_writes = report["prediction_reports"]["no_writes-k2-first1"]
        self.assertIn("paired_with_normal_same_questions", no_writes)
        self.assertEqual(no_writes["controller_training_labels"]["executed_write_rate"], 0)
        self.assertEqual(len(report["gradient_reports"]), 3)
        with mock.patch("scripts.diagnose_dsa_pilot._load_model", side_effect=AssertionError("cache should skip load")):
            self.assertTrue(self.diagnose()["reused_completed_diagnostic"])

    def test_partial_run_resumes_finished_predictions(self):
        from scripts.diagnose_dsa_pilot import gradient_probe as original
        with mock.patch("scripts.diagnose_dsa_pilot.gradient_probe", side_effect=RuntimeError("disconnected")):
            with self.assertRaisesRegex(RuntimeError, "disconnected"):
                self.diagnose()
        with mock.patch("scripts.diagnose_dsa_pilot.predict_record", side_effect=AssertionError("already generated")), \
                mock.patch("scripts.diagnose_dsa_pilot.gradient_probe", wraps=original) as probe:
            report = self.diagnose()
        self.assertTrue(report["complete"])
        self.assertEqual(probe.call_count, 3)

    def test_completed_artifact_tampering_is_detected(self):
        self.diagnose()
        path = next((self.out / "gradients").glob("*.json"))
        path.write_text(path.read_text() + " ")
        with self.assertRaisesRegex(ValueError, "checksum differs"):
            self.diagnose()

    def test_refuses_outputs_in_run_directory(self):
        with self.assertRaisesRegex(ValueError, "separate"):
            diagnose(self.prepared, self.run, self.run / "diagnostics", device="cpu")


if __name__ == "__main__":
    unittest.main()
