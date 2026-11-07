"""Smoke test of scripts/inner_loop/train.py and eval.py end to end on tiny random models (no downloads): a loop arm
trains and saves, its gate leaves zero, and the evaluator's identity arms reproduce `plain` exactly."""
import json
import math
import os
import runpy
import sys
import tempfile
import unittest
from pathlib import Path

import torch
from torch import nn
from transformers import GPT2Config, GPT2LMHeadModel

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts/inner_loop"))

import looped  # noqa: E402
from dsa.bench.models import codi as codi_module  # noqa: E402
from dsa.bench.models.gpt2_vanilla import GPT2_REVISION  # noqa: E402

WIDTH = 32


class FakeCODI:
    """CODIGPT2's interface over a tiny random GPT-2 with the real tokenizer and CODI's three extra ids."""

    def __init__(self, checkpoint=None, device="cpu", **_):
        from transformers import GPT2Tokenizer
        self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2", revision=GPT2_REVISION)
        self.vocab_size, self.bot_id, self.eot_id, self.max_new_tokens = 50260, 50258, 50259, 6
        torch.manual_seed(0)
        config = GPT2Config(n_layer=12, n_embd=WIDTH, n_head=4, vocab_size=self.vocab_size, n_positions=1024)
        config._attn_implementation = "eager"
        self.model = GPT2LMHeadModel(config).eval()
        self.projection = nn.Sequential(nn.Linear(WIDTH, WIDTH), nn.GELU(), nn.Linear(WIDTH, WIDTH), nn.LayerNorm(WIDTH))
        self.projection.eval()
        self.device = torch.device("cpu")
        path = Path(checkpoint) / "pytorch_model.bin" if checkpoint else None
        if path and path.exists():
            state = torch.load(path, map_location="cpu", weights_only=True)
            self.model.load_state_dict({k.removeprefix("codi.base_model.model."): v for k, v in state.items()
                                        if k.startswith("codi.")}, strict=False)
            self.projection.load_state_dict({k.removeprefix("prj."): v for k, v in state.items() if k.startswith("prj.")})


def fake_decoder(device):
    torch.manual_seed(5)
    config = GPT2Config(n_layer=2, n_embd=WIDTH, n_head=4, vocab_size=50257, n_positions=64)
    config._attn_implementation = "eager"
    decoder = GPT2LMHeadModel(config).eval()
    for p in decoder.parameters():
        p.requires_grad_(False)
    return decoder


def run(script, argv):
    old = sys.argv
    sys.argv = [script] + [str(a) for a in argv]
    try:
        runpy.run_path(str(ROOT / "scripts/inner_loop" / script), run_name="__main__")
    finally:
        sys.argv = old


class ScriptsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        os.environ["HF_HUB_OFFLINE"] = "1"
        try:
            FakeCODI()
        except OSError as error:
            raise unittest.SkipTest(f"GPT-2 tokenizer not cached: {error}")
        cls.saved = (codi_module.CODIGPT2, looped.load_step_decoder)
        codi_module.CODIGPT2, looped.load_step_decoder = FakeCODI, fake_decoder

    @classmethod
    def tearDownClass(cls):
        codi_module.CODIGPT2, looped.load_step_decoder = cls.saved

    def test_train_then_eval(self):
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            rows = [{"question": f"Tom has {n} apples and buys 3 more. How many now?", "steps": [f"<<{n}+3={n + 3}>>"],
                     "answer": str(n + 3)} for n in range(8)]
            (folder / "train.json").write_text(json.dumps(rows))
            (folder / "base").mkdir()
            run("train.py", ["--arm", "loop", "--checkpoint", folder / "base", "--output", folder / "loop",
                             "--train-file", folder / "train.json", "--records", 8, "--batch", 4, "--device", "cpu",
                             "--rank", 2, "--k-max", 3, "--p-k1", 0, "--save-every", 1])
            metrics = [json.loads(x) for x in (folder / "loop/metrics.jsonl").read_text().splitlines()]
            self.assertEqual(len(metrics), 2)
            self.assertIsNotNone(metrics[0]["step"])  # the step-decoder loss is in the objective
            self.assertTrue((folder / "loop/model/loop.pt").exists())
            self.assertFalse((folder / "loop/resume.pt").exists())
            gate = torch.load(folder / "loop/model/loop.pt", weights_only=True)["state"]["gate"]
            self.assertGreater(float(gate.abs().max()), 0.0)

            run("eval.py", ["--checkpoint", folder / "loop/model", "--dataset", "gsm8k_valid",
                            "--output", folder / "trained.jsonl", "--arms", "plain", "loop1", "loop3", "plain@8",
                            "--decoder", "--limit", 2, "--device", "cpu"])
            records = [json.loads(x) for x in (folder / "trained.jsonl").read_text().splitlines()]
            self.assertEqual(len(records), 2)
            for r in records:
                self.assertEqual(r["arms"]["loop1"]["latent_drift"], 0.0)  # pass 1 alone is the trained model
                self.assertGreater(r["arms"]["loop3"]["latent_drift"], 0.0)
                self.assertEqual(len(r["arms"]["plain"]["steps"]), 7)
                self.assertEqual(len(r["arms"]["plain@8"]["steps"]), 9)

            run("eval.py", ["--checkpoint", folder / "base", "--dataset", "gsm8k_valid", "--output", folder / "zero.jsonl",
                            "--arms", "plain", "loop4", "damp2x0.5", "rerun2", "--limit", 2, "--device", "cpu"])
            records = [json.loads(x) for x in (folder / "zero.jsonl").read_text().splitlines()]
            for r in records:
                self.assertEqual(r["arms"]["loop4"]["latent_drift"], 0.0)  # zero loop parameters: an identity
                self.assertEqual(r["arms"]["loop4"]["text"], r["arms"]["plain"]["text"])
                self.assertGreater(r["arms"]["damp2x0.5"]["latent_drift"], 0.0)
                self.assertGreater(r["arms"]["rerun2"]["latent_drift"], 0.0)

    def test_matched_and_more_steps_arms_train(self):
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            rows = [{"question": f"Ann has {n} pens. She gives 1 away. How many left?", "steps": [f"<<{n}-1={n - 1}>>"],
                     "answer": str(n - 1)} for n in range(2, 6)]
            (folder / "train.json").write_text(json.dumps(rows))
            (folder / "base").mkdir()
            for arm in ("loop_k1", "ft12"):
                run("train.py", ["--arm", arm, "--checkpoint", folder / "base", "--output", folder / arm,
                                 "--train-file", folder / "train.json", "--records", 4, "--batch", 4,
                                 "--device", "cpu", "--rank", 2])
                args = json.loads((folder / arm / "args.json").read_text())
                self.assertEqual(args["depth"], 12 if arm == "ft12" else 6)
                self.assertAlmostEqual(args["lr"], 2e-5 * math.sqrt(4 / 32))  # scaled from batch 32
            self.assertTrue((folder / "loop_k1/model/loop.pt").exists())
            self.assertFalse((folder / "ft12/model/loop.pt").exists())

    def test_kk_on_the_answer_loss_alone(self):
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            rows = [{"question": f"Sam has {n} cups and breaks 2. How many are left?", "steps": [f"<<{n}-2={n - 2}>>"],
                     "answer": str(n - 2)} for n in range(3, 11)]
            (folder / "train.json").write_text(json.dumps(rows))
            (folder / "base").mkdir()
            run("train.py", ["--arm", "ft6", "--kk", "--kk-beta", 1, "--step-weight", 0, "--checkpoint", folder / "base",
                             "--output", folder / "kk", "--train-file", folder / "train.json", "--records", 8,
                             "--batch", 4, "--device", "cpu"])
            metrics = [json.loads(x) for x in (folder / "kk/metrics.jsonl").read_text().splitlines()]
            self.assertIsNotNone(metrics[0]["step"])  # logged, though not trained on
            self.assertEqual(metrics[0]["loss"], metrics[0]["ce"])
            self.assertGreater(metrics[-1]["alpha"], 0.0)
            self.assertTrue((folder / "kk/model/kk.pt").exists())
            run("eval.py", ["--checkpoint", folder / "kk/model", "--dataset", "gsm8k_valid", "--output",
                            folder / "kk.jsonl", "--arms", "plain", "nokk", "--limit", 2, "--device", "cpu"])
            records = [json.loads(x) for x in (folder / "kk.jsonl").read_text().splitlines()]
            self.assertEqual(len(records), 2)
            self.assertTrue(all("latent_drift" in r["arms"]["nokk"] for r in records))


class ReportTest(unittest.TestCase):
    def report(self, folder, hits, gate, metrics=None):
        """Synthetic evaluations of 20 questions with the given correct counts per (eval, arm), and final training
        metrics per arm; returns the output."""
        import contextlib
        import io

        def write(name, arms):
            rows = [{"id": i, "gold": "1", "rationale_steps": 4 if i % 2 else 2,
                     "arms": {arm: {"text": f"{arm} {i < n}", "prediction": str(int(i < n)), "correct": i < n}
                              for arm, n in arms.items()}} for i in range(20)]
            for r in rows:  # identity arms share plain's text and drift 0
                for arm in arms:
                    if arm.startswith("loop") and name.startswith("base"):
                        r["arms"][arm]["text"], r["arms"][arm]["latent_drift"] = r["arms"]["plain"]["text"], 0.0
            (folder / name).mkdir(parents=True)
            (folder / name / "gsm8k_valid.jsonl").write_text("\n".join(json.dumps(r) for r in rows))

        for name, arms in hits.items():
            write(name, arms)
        (folder / "arms/loop").mkdir(parents=True)
        (folder / "arms/loop/metrics.jsonl").write_text(json.dumps({"update": 0, "gate": gate}) + "\n")
        for arm, final in (metrics or {}).items():
            (folder / "arms" / arm).mkdir(parents=True)
            (folder / "arms" / arm / "metrics.jsonl").write_text(json.dumps({"update": 0, **final}) + "\n")
        out = io.StringIO()
        old = sys.argv
        sys.argv = ["report.py", str(folder), "--training", str(folder / "arms")]
        try:
            with contextlib.redirect_stdout(out):
                runpy.run_path(str(ROOT / "scripts/inner_loop/report.py"), run_name="__main__")
        finally:
            sys.argv = old
        return out.getvalue()

    def test_go_and_stop(self):
        hits = {"base-simcot": {"plain": 8, "loop2": 8}, "ft6": {"plain": 5}, "ft12": {"plain": 6, "plain@12": 6},
                "loop": {"plain": 6, "loop1": 6, "loop3": 18}}
        with tempfile.TemporaryDirectory() as folder:
            text = self.report(Path(folder), hits, gate=0.01)
        self.assertIn("identity base-simcot loop2: PASS", text)
        self.assertIn("GO:", text)
        with tempfile.TemporaryDirectory() as folder:
            self.assertIn("STOP:", self.report(Path(folder), hits, gate=0.0))  # a closed gate stops it
        hits["loop"]["loop3"] = 9
        with tempfile.TemporaryDirectory() as folder:
            self.assertIn("STOP:", self.report(Path(folder), hits, gate=0.01))

    def test_stage1b(self):
        hits = {"base-simcot": {"plain": 8}, "ft6_step": {"plain": 5}, "ft6_ce": {"plain": 6},
                "loop_ce": {"plain": 6, "loop1": 6, "loop3": 6}, "kk_ce": {"plain": 17, "nokk": 8}}
        metrics = {"loop_ce": {"gate": 0.01}, "kk_ce": {"alpha": 0.2, "alpha_max": 0.5}}
        with tempfile.TemporaryDirectory() as folder:
            text = self.report(Path(folder), hits, gate=0.01, metrics=metrics)
        self.assertIn("no clear difference between the recipes", text)
        self.assertIn("loop on the answer loss alone: STOP", text)
        self.assertIn("KK-LSE at 6 steps: GO", text)
        hits["kk_ce"]["nokk"] = 15  # the gain does not depend on the bias at test time
        with tempfile.TemporaryDirectory() as folder:
            self.assertIn("KK-LSE at 6 steps: STOP", self.report(Path(folder), hits, gate=0.01, metrics=metrics))


if __name__ == "__main__":
    unittest.main()
