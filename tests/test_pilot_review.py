"""End-to-end review and export checks using tiny local checkpoints only."""
import contextlib
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock
from zipfile import ZipFile

import torch
from dsa.data.pilot_prepare import prepare_pilot
from dsa.training.pilot import PilotConfig, train_pilot
from dsa.training.objectives import LossConfig, joint_objective
from scripts.diagnose_dsa_pilot import diagnose, forward_profile, inference_condition
from scripts.evaluate_dsa import evaluate_run, _load_model, resolve_checkpoint
from scripts.pilot_explicit_controls import explicit_controls
from scripts.package_pilot_review import package_review

torch.set_num_threads(1)
ROOT = Path(__file__).resolve().parents[1]


class PilotReviewTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.project = Path(self.temp.name)
        source = self.project / "source"
        source.mkdir()
        for split in ("train", "valid", "test"):
            (source / f"gsm_{split}.json").write_text(json.dumps([
                {"question": f"{split}: twice 2+3?", "steps": ["<<2+3=5>>", "<<5*2=10>>"], "answer": "10"},
                {"question": f"{split}: add 2+3?", "steps": ["<<2+3=5>>"], "answer": "5"}]))
        self.prepared = self.project / "prepared/cache"
        prepare_pilot(storage_dir=None, source_dir=source, output_dir=self.prepared, loops=2, tiny=True, train_limit=0)
        self.run_name, self.review_name = "fixture-run", "review-v2"
        self.runs = self.project / "runs" / self.run_name
        self.review = self.project / "reports" / self.run_name / self.review_name
        self.review.mkdir(parents=True)
        for arm in ("static", "learned"):
            with contextlib.redirect_stdout(io.StringIO()):
                train_pilot(PilotConfig(self.prepared, self.runs / arm, tiny=True, write_mode=arm,
                                       loops=2, steps=2, save_every=2, device="cpu"))

    def complete(self):
        with contextlib.redirect_stdout(io.StringIO()):
            for arm in ("static", "learned"):
                run, out = self.runs / arm, self.review / arm
                diagnose(self.prepared, run, out / "diagnostics", device="cpu", gradient_limit=1,
                         probe_limit=1, depth_grid=(1,2), max_new_tokens=2, comprehensive=True)
                evaluate_run(self.prepared, run, out / "validation-step2-k2-n2-g2", device="cpu", limit=2, max_new_tokens=2)
                explicit_controls(self.prepared, out / "explicit-control", run_dir=run, device="cpu", limit=1, max_new_tokens=2)
            explicit_controls(self.prepared, self.review / "original-explicit-control", device="cpu", limit=1, max_new_tokens=2)
        (self.review / "protocol.json").write_text('{"fixture":true}')
        (self.review / "session_status.json").write_text('{"fixture":{"status":"passed"}}')

    def test_complete_bundle_contains_reproducible_evidence_without_weights_or_test_answers(self):
        self.complete()
        with contextlib.redirect_stdout(io.StringIO()):
            manifest = package_review(self.project, self.run_name, self.review_name, self.prepared)
        self.assertTrue(manifest["evidence_complete"])
        archive = self.project / "bundles" / f"{self.run_name}-{self.review_name}.zip"
        with ZipFile(archive) as zipped:
            names = zipped.namelist()
            self.assertIn("review/scale_readiness.json", names)
            self.assertIn("prepared/train.tokenized.jsonl", names)
            self.assertIn("prepared/valid.jsonl", names)
            self.assertFalse(any(name.endswith((".pt", ".bin", ".safetensors")) for name in names))
            self.assertFalse(any(name.endswith("test.jsonl") for name in names))
            for name, entry in manifest["included"].items():
                self.assertEqual(hashlib.sha256(zipped.read(name)).hexdigest(), entry["sha256"])
            readiness = json.loads(zipped.read("review/scale_readiness.json"))
            self.assertTrue(readiness["matched_controls"]["learned"]["matched"])
            self.assertEqual(readiness["paired_validation"]["n"], 2)

    def test_partial_export_lists_missing_evidence_and_keeps_error_logs(self):
        (self.review / "session_status.json").write_text('{"gpu":{"status":"failed","error":"CUDA unavailable"}}')
        for arm in ("static", "learned"):
            partial = self.review / arm / "validation-step2-k2-n2-g2"
            partial.mkdir(parents=True)
            (partial / "predictions.jsonl").write_bytes(b'{"torn":"prediction')
        with contextlib.redirect_stdout(io.StringIO()):
            manifest = package_review(self.project, self.run_name, self.review_name, self.prepared)
        self.assertFalse(manifest["evidence_complete"])
        evidence = json.loads((self.review / "scale_readiness.json").read_text())
        self.assertTrue(evidence["missing_required_artifacts"])
        self.assertTrue(evidence["artifact_errors"])

    def test_forward_profile_is_the_same_objective_and_does_not_modify_weights(self):
        model = _load_model(resolve_checkpoint(self.runs / "learned") / "model")
        record = json.loads((self.prepared / "train.tokenized.jsonl").read_text().splitlines()[0])
        prefix = record["latent_answer_prefix_ids"]
        model.eval()
        with torch.no_grad(): expected = joint_objective(model, record, prefix)["total"].item()
        profile = forward_profile(model, record, prefix, LossConfig())
        self.assertEqual(profile["losses"]["total"], expected)
        self.assertEqual(len(profile["states"]), 2)
        self.assertEqual(len(profile["scheduled_teacher_alignments"]), 1)
        self.assertTrue(all(p.grad is None for p in model.parameters()))

    def test_zeroing_final_memory_does_not_modify_latent_trajectory(self):
        model = _load_model(resolve_checkpoint(self.runs / "learned") / "model").eval()
        record = json.loads((self.prepared / "train.tokenized.jsonl").read_text().splitlines()[0])
        question = torch.tensor([record["question_input_ids"]])
        loop = model(question, torch.ones_like(question))
        anchor, workspace = loop.anchor.clone(), loop.workspace.clone()
        suffix = torch.tensor([record["latent_answer_prefix_ids"]])
        with mock.patch.object(model.backbone, "forward_embeds", wraps=model.backbone.forward_embeds) as forward:
            with inference_condition(model, "zero_memory"):
                model.answer_logits(loop, suffix, torch.ones_like(suffix))
            actual_input = forward.call_args.args[0]
        memory = actual_input[:, question.shape[1]:question.shape[1]+1+model.config.workspace_size]
        self.assertEqual(memory.abs().sum().item(), 0)
        self.assertTrue(torch.equal(loop.anchor, anchor))
        self.assertTrue(torch.equal(loop.workspace, workspace))

    def test_reconnected_notebook_review_skips_download_smoke_and_training(self):
        import os, shutil, sys
        notebook = json.loads((ROOT / "notebooks/DSA_Colab.ipynb").read_text())
        sources = {c["id"]: "".join(c["source"]) for c in notebook["cells"]}
        config = sources["config"].replace('EXISTING_RUN_NAME = "gpt2-pilot-v1-ba4e69315086"',
                                          f'EXISTING_RUN_NAME = "{self.run_name}"')
        config = config.replace('Path("/content/dsa-prepared")', f"Path({str(self.project / 'vm')!r})")
        namespace = {"Path": Path, "json": json, "hashlib": hashlib, "shutil": shutil, "os": os,
                     "PROJECT_DIR": self.project, "REPO_DIR": ROOT, "REPO_COMMIT": "new-review-code",
                     "PYTHON": sys.executable, "GPU_READY": True}
        with contextlib.redirect_stdout(io.StringIO()):
            exec(compile(config, "config", "exec"), namespace)
            namespace["run_project"] = mock.Mock(side_effect=AssertionError("review must not train/download/smoke"))
            for name in ("data", "pretrained-smoke", "training"):
                exec(compile(sources[name], name, "exec"), namespace)
        namespace["run_project"].assert_not_called()
        self.assertEqual(namespace["RUN_ROOT"], self.runs)
        self.assertEqual(namespace["PREPARED_DRIVE"], self.prepared)
        self.assertEqual(namespace["training_reports"]["learned"]["step"], 2)


class UnifiedNotebookTests(unittest.TestCase):
    def test_review_mode_selects_original_run_and_exports_after_diagnostics(self):
        notebook = json.loads((ROOT / "notebooks/DSA_Colab.ipynb").read_text())
        sources = {c["id"]: "".join(c["source"]) for c in notebook["cells"]}
        ids = [c["id"] for c in notebook["cells"]]
        self.assertIn('RUN_MODE = "review_existing"', sources["config"])
        self.assertIn('EXISTING_RUN_NAME = "gpt2-pilot-v1-ba4e69315086"', sources["config"])
        self.assertIn('if RUN_MODE == "train_pilot"', sources["data"])
        self.assertIn('if RUN_MODE == "train_pilot"', sources["training"])
        self.assertLess(ids.index("pilot-diagnostics"), ids.index("pilot-export"))
        self.assertLess(ids.index("pilot-controls"), ids.index("pilot-export"))
        self.assertIn("session_status.json", sources["config"])
        self.assertIn("files.download", sources["pilot-export"])
        for c in notebook["cells"]:
            if c["cell_type"] == "code": compile(sources[c["id"]], c["id"], "exec")


if __name__ == "__main__": unittest.main()
