"""End-to-end persistent pilot tests with a download-free tiny backbone."""
from dataclasses import replace
import json
import importlib.metadata
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import torch

from dsa.data.milestones import annotate_record
from dsa.data.tokenize import TinyTokenizer, tokenize_record
from dsa.models.anchored_loop import AnchoredLoop
from dsa.training.pilot import (PilotConfig, sha256_file, tokenizer_directory_sha256,
                                train_pilot)


torch.set_num_threads(1)


class PilotTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.prepared = self.root / "prepared"
        self.prepared.mkdir()
        tokenizer = TinyTokenizer()
        tokenizer.save_pretrained(self.prepared / "tokenizer")
        raw = [{"question": "Add 2 and 3.", "answer": "<<2+3=5>>\n#### 5"},
               {"question": "Add 3 and 4.", "answer": "<<3+4=7>>\n#### 7"}]
        records = [tokenize_record(annotate_record(row), tokenizer, 2) for row in raw]
        for index, record in enumerate(records):
            record["sample_id"] = str(index)
        train = self.prepared / "train.tokenized.jsonl"
        train.write_text("".join(json.dumps(row) + "\n" for row in records))
        for split in ("valid", "test"):
            (self.prepared / f"{split}.jsonl").write_text(json.dumps(raw[0]) + "\n")
        manifest = {"schema_version": 1, "backbone": "tiny-gpt2", "revision": "random",
                    "tiny": True, "loop_budget": 2,
                    "tokenizer_sha256": tokenizer_directory_sha256(self.prepared / "tokenizer"),
                    "splits": {split: {"path": path,
                                        "sha256": sha256_file(self.prepared / path)}
                               for split, path in (("train", "train.tokenized.jsonl"),
                                                   ("valid", "valid.jsonl"), ("test", "test.jsonl"))}}
        (self.prepared / "manifest.json").write_text(json.dumps(manifest))
        self.config = PilotConfig(self.prepared, self.root / "run", tiny=True,
                                  loops=2, steps=3, save_every=1, learning_rate=0.001,
                                  gradient_accumulation=2, device="cpu")

    def _metrics(self, output):
        return [json.loads(line) for line in (output / "metrics.jsonl").read_text().splitlines()]

    def test_resume_matches_uninterrupted_parameters_optimizer_and_losses(self):
        whole = train_pilot(self.config)
        interrupted_config = replace(self.config, output_dir=self.root / "resumed", steps=1)
        train_pilot(interrupted_config)
        result = train_pilot(replace(interrupted_config, steps=3))
        self.assertEqual(result["step"], 3)
        self.assertIsNotNone(result["resumed_from"])
        self.assertEqual(whole["initial_state_sha256"], result["initial_state_sha256"])
        expected = AnchoredLoop.from_pretrained(Path(whole["checkpoint"]) / "model")
        actual = AnchoredLoop.from_pretrained(Path(result["checkpoint"]) / "model")
        for key, tensor in expected.state_dict().items():
            self.assertTrue(torch.equal(tensor, actual.state_dict()[key]), key)
        whole_metrics = self._metrics(Path(self.config.output_dir))
        resumed_metrics = self._metrics(interrupted_config.output_dir)
        for left, right in zip(whole_metrics, resumed_metrics):
            for key in ("total", "answer", "cot", "gate", "anchor", "writes", "gate_probability",
                        "write_rate", "sample_ids", "gradient_norm", "gate_gradient_norm"):
                self.assertEqual(left[key], right[key], key)
        state_a = torch.load(Path(whole["checkpoint"]) / "training_state.pt", weights_only=False)
        state_b = torch.load(Path(result["checkpoint"]) / "training_state.pt", weights_only=False)
        self.assertEqual(state_a["scheduler"], state_b["scheduler"])
        for parameter_id, state in state_a["optimizer"]["state"].items():
            for key, tensor in state.items():
                self.assertTrue(torch.equal(tensor, state_b["optimizer"]["state"][parameter_id][key]))
        self.assertEqual([row["step"] for row in resumed_metrics], [1, 2, 3])

    def test_write_arms_share_initial_parameters(self):
        digests = []
        for mode in ("static", "every_loop", "periodic", "learned"):
            report = train_pilot(replace(self.config, output_dir=self.root / mode,
                                         write_mode=mode, steps=1))
            digests.append(report["initial_state_sha256"])
        self.assertEqual(len(set(digests)), 1)

    def test_resume_rejects_seed_loop_and_loss_changes(self):
        train_pilot(replace(self.config, steps=1))
        for change in ({"seed": 14}, {"cot_weight": 0.5}, {"workspace": 2},
                       {"gradient_accumulation": 1}):
            with self.assertRaisesRegex(ValueError, "settings differ"):
                train_pilot(replace(self.config, **change))
        with self.assertRaisesRegex(ValueError, "configuration differs"):
            train_pilot(replace(self.config, loops=3))

    def test_dataset_and_tokenizer_corruption_detected(self):
        train = self.prepared / "train.tokenized.jsonl"
        train.write_text(train.read_text() + "\n")
        with self.assertRaisesRegex(ValueError, "train checksum"):
            train_pilot(self.config)
        tokenizer_config = self.prepared / "tokenizer" / "tiny_tokenizer.json"
        tokenizer_config.write_text(tokenizer_config.read_text() + "\n")
        with self.assertRaisesRegex(ValueError, "tokenizer checksum"):
            train_pilot(self.config)

    def test_incomplete_checkpoint_ignored_and_metrics_reconciled(self):
        first = train_pilot(replace(self.config, steps=1))
        output = Path(self.config.output_dir)
        incomplete = output / "checkpoint-99999999"
        incomplete.mkdir()
        (incomplete / "training_state.pt").write_text("interrupted")
        metrics = output / "metrics.jsonl"
        with metrics.open("a") as handle:
            handle.write(json.dumps({"step": 2, "total": 123}) + "\n")
            handle.write('{"step":3')
        result = train_pilot(self.config)
        self.assertEqual(result["resumed_from"], first["checkpoint"])
        self.assertEqual([row["step"] for row in self._metrics(output)], [1, 2, 3])
        self.assertTrue(incomplete.exists())
        self.assertEqual(len([path for path in output.glob("checkpoint-*")
                              if (path / "COMPLETE.json").exists()]), 2)

    def test_resume_none_preserves_existing_run(self):
        train_pilot(replace(self.config, steps=1))
        metrics = (Path(self.config.output_dir) / "metrics.jsonl").read_bytes()
        with self.assertRaisesRegex(ValueError, "already contains"):
            train_pilot(replace(self.config, resume="none"))
        self.assertEqual(metrics, (Path(self.config.output_dir) / "metrics.jsonl").read_bytes())

    def test_disconnect_before_first_update_resumes_from_initial_checkpoint(self):
        with mock.patch("dsa.training.pilot.joint_objective", side_effect=RuntimeError("disconnected")):
            with self.assertRaisesRegex(RuntimeError, "disconnected"):
                train_pilot(self.config)
        initial = Path(self.config.output_dir) / "checkpoint-00000000"
        self.assertTrue((initial / "COMPLETE.json").exists())
        report = train_pilot(self.config)
        self.assertEqual(report["resumed_from"], str(initial.resolve()))
        self.assertEqual([row["step"] for row in self._metrics(Path(self.config.output_dir))], [1, 2, 3])

    def test_real_run_requires_matching_unmodified_smoke_before_weights_load(self):
        manifest_path = self.prepared / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        manifest.update(tiny=False, backbone=self.config.backbone, revision=self.config.revision)
        manifest_path.write_text(json.dumps(manifest))
        config = replace(self.config, tiny=False)
        with mock.patch("transformers.AutoTokenizer.from_pretrained", return_value=TinyTokenizer()):
            with self.assertRaisesRegex(ValueError, "require --smoke-report"):
                train_pilot(config)
            smoke = self.root / "smoke.json"
            smoke.write_text(json.dumps({"passed": True, "backbone": config.backbone,
                                         "revision": "wrong"}))
            with self.assertRaisesRegex(ValueError, "does not match"):
                train_pilot(replace(config, smoke_report=smoke))
            valid_proof = {"schema_version": "dsa_unmodified_checkpoint_smoke_v1",
                           "passed": True, "backbone": config.backbone, "revision": config.revision,
                           "tiny": False, "model_weights_downloaded": True, "finite_logits": True,
                           "vocabulary_changed": False, "anchor_wrapper_used": False,
                           "device": "cpu", "identity": {"torch": torch.__version__,
                           "transformers": importlib.metadata.version("transformers")}}
            for field, value in (("tiny", True), ("model_weights_downloaded", False),
                                  ("finite_logits", False), ("vocabulary_changed", True),
                                  ("anchor_wrapper_used", True), ("schema_version", "wrong"),
                                  ("device", "cuda")):
                smoke.write_text(json.dumps({**valid_proof, field: value}))
                with self.assertRaisesRegex(ValueError, "does not match", msg=field):
                    train_pilot(replace(config, smoke_report=smoke))


if __name__ == "__main__":
    unittest.main()
