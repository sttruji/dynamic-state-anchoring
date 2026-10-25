"""Device routing plus a hardware-gated MPS train/resume/evaluate integration."""
from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from dsa.devices import mps_memory_snapshot, resolve_device, synchronize


class DeviceTests(unittest.TestCase):
    def test_auto_prefers_cuda_then_mps_then_cpu(self):
        for cuda, mps, expected in ((True, True, "cuda"), (False, True, "mps"),
                                    (False, False, "cpu")):
            with self.subTest(expected=expected), \
                 patch("torch.cuda.is_available", return_value=cuda), \
                 patch("torch.backends.mps.is_available", return_value=mps):
                self.assertEqual(str(resolve_device("auto")), expected)

    def test_explicit_unavailable_accelerators_fail_without_cpu_fallback(self):
        with patch("torch.cuda.is_available", return_value=False), \
             patch("torch.backends.mps.is_available", return_value=False):
            for device in ("cuda", "mps"):
                with self.subTest(device=device), self.assertRaisesRegex(ValueError, "unavailable"):
                    resolve_device(device)
            self.assertEqual(str(resolve_device("cpu")), "cpu")
            with self.assertRaisesRegex(ValueError, "Device must"):
                resolve_device("unknown")

    def test_synchronization_uses_the_selected_backend(self):
        with patch("torch.cuda.synchronize") as cuda, patch("torch.mps.synchronize") as mps:
            synchronize("cpu")
            cuda.assert_not_called()
            mps.assert_not_called()
            synchronize("mps")
            mps.assert_called_once_with()
            synchronize("cuda")
            cuda.assert_called_once_with(torch.device("cuda"))

    def test_mps_memory_is_labeled_as_current_allocation(self):
        with patch("torch.mps.current_allocated_memory", return_value=10) as allocated, \
             patch("torch.mps.driver_allocated_memory", return_value=20), \
             patch("torch.mps.recommended_max_memory", return_value=100):
            self.assertEqual(mps_memory_snapshot("cpu"), {})
            allocated.assert_not_called()
            self.assertEqual(mps_memory_snapshot("mps"), {
                "mps_current_allocated_bytes": 10, "mps_driver_allocated_bytes": 20,
                "mps_recommended_max_bytes": 100})


@unittest.skipUnless(torch.backends.mps.is_available(), "requires native Apple MPS access")
class MPSIntegrationTests(unittest.TestCase):
    def test_flt_training_resume_and_question_only_validation(self):
        from dsa.data.pilot_prepare import prepare_pilot
        from dsa.models.anchored_loop import AnchoredLoop
        from dsa.training.pilot import PilotConfig, train_pilot
        from scripts.evaluate_dsa import evaluate_run

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            source.mkdir()
            for split in ("train", "valid", "test"):
                rows = [{"question": f"{split}: 2 groups of 3 apples, then eat 1. How many remain?",
                         "steps": ["<<2*3=6>>", "<<6-1=5>>"], "answer": "5"}]
                (source / f"gsm_{split}.json").write_text(json.dumps(rows))
            prepared = root / "prepared"
            prepare_pilot(storage_dir=None, source_dir=source, output_dir=prepared,
                          train_limit=0, loops=2, tiny=True)
            config = PilotConfig(prepared, root / "whole", tiny=True, device="mps",
                                 recurrence_mode="flt_memory", loops=2, steps=2, save_every=1)
            whole = train_pilot(config)
            interrupted = replace(config, output_dir=root / "resumed", steps=1)
            train_pilot(interrupted)
            resumed = train_pilot(replace(interrupted, steps=2))
            self.assertEqual(whole["initial_state_sha256"], resumed["initial_state_sha256"])
            expected = AnchoredLoop.from_pretrained(Path(whole["checkpoint"]) / "model")
            actual = AnchoredLoop.from_pretrained(Path(resumed["checkpoint"]) / "model")
            for key, tensor in expected.state_dict().items():
                torch.testing.assert_close(tensor, actual.state_dict()[key], rtol=1e-5, atol=1e-6)
            metrics = [json.loads(row) for row in (root / "resumed/metrics.jsonl").read_text().splitlines()]
            self.assertEqual([row["step"] for row in metrics], [1, 2])
            self.assertTrue(all(row["mps_current_allocated_bytes"] > 0 for row in metrics))
            report = evaluate_run(prepared, root / "resumed", root / "evaluation",
                                  device="mps", max_new_tokens=2)
            self.assertEqual(report["total"], 1)
            self.assertTrue(report["question_only_inference"])
            self.assertGreater(report["mps_driver_allocated_bytes"], 0)
