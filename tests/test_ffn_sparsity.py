"""Checks of SPARSE-0 (results/ffn_sparsity/PLAN.md). The hook with nothing dropped is GPT-2 exactly; the masked FFN
equals a gathered sparse FFN; ties go to the lower index; arms touch only their phases; the parameter counter matches
GPT-2 small. On a tiny random GPT-2, run.py calibrates, refuses a mismatched reference and resumes by arm. report.py's
rules give each verdict on synthetic arms."""
import contextlib
import io
import json
import os
import runpy
import sys
import tempfile
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts/ffn_sparsity"))

import sparsity  # noqa: E402
from dsa.bench import scoring  # noqa: E402
from dsa.bench.models import codi as codi_module  # noqa: E402
from tests.test_inner_loop_scripts import FakeCODI  # noqa: E402
from tests.test_step_probe import fake_score, problem  # noqa: E402


def run_script(script, argv):
    old, out = sys.argv, io.StringIO()
    sys.argv = [script] + [str(a) for a in argv]
    try:
        with contextlib.redirect_stdout(out):
            runpy.run_path(str(ROOT / "scripts/ffn_sparsity" / script), run_name="__main__")
    finally:
        sys.argv = old
    return out.getvalue()


def tiny_gpt2():
    from transformers import GPT2Config, GPT2LMHeadModel
    torch.manual_seed(0)
    config = GPT2Config(n_layer=2, n_embd=32, n_head=4, vocab_size=100, n_positions=64)
    config._attn_implementation = "eager"
    return GPT2LMHeadModel(config).eval()


class HookTest(unittest.TestCase):
    def setUp(self):
        self.model = tiny_gpt2()
        self.ids = torch.randint(0, 100, (2, 9))

    def test_identity(self):
        with torch.no_grad():
            dense = self.model(self.ids).logits
            masks = sparsity.FFNMasks(self.model)
            masks.phase = sparsity.phase_tags(8, 1)
            self.assertTrue(torch.equal(self.model(self.ids).logits, dense))

    def test_masked_equals_gathered(self):
        masks = sparsity.FFNMasks(self.model)
        mlp, inner = masks.blocks[0], masks.inner
        masks.mu[("oracle", 0.25)] = torch.randn(2, 3, inner) * 0.1
        masks.arm, masks.phase = ("oracle", 0.25, "QTA"), sparsity.phase_tags(5, 3, 1)
        x = torch.randn(2, 9, 32)
        with torch.no_grad():
            masked = mlp(x)
            a = mlp.act(mlp.c_fc(x))
            weight, bias = mlp.c_proj.weight, mlp.c_proj.bias  # Conv1D: [inner, width]
            for b in range(2):
                for s in range(9):
                    mu = masks.mu[("oracle", 0.25)][0, masks.phase[s]]
                    kept = sparsity.ranked(a[b, s].abs() * weight.norm(dim=1))[:inner // 4]
                    gathered = (a[b, s, kept] - mu[kept]) @ weight[kept] + mu @ weight + bias
                    torch.testing.assert_close(masked[b, s], gathered, atol=1e-5, rtol=0)

    def test_phases_and_static(self):
        masks = sparsity.FFNMasks(self.model)
        mlp, inner = masks.blocks[1], masks.inner
        masks.mu[("oracle", 0.25)] = torch.zeros(2, 3, inner)
        masks.mu[("static", 0.5)] = torch.zeros(2, 3, inner)
        masks.static[0.5] = torch.zeros(2, inner, dtype=torch.bool)
        masks.static[0.5][:, :inner // 2] = True
        masks.phase = sparsity.phase_tags(5, 3, 1)
        x = torch.randn(1, 9, 32)
        with torch.no_grad():
            dense = mlp(x)
            masks.arm = ("oracle", 0.25, "T")
            thinking = mlp(x)
            masks.arm = ("static", 0.5, "QTA")
            a = mlp.act(mlp.c_fc(x))
            static = mlp(x)
        torch.testing.assert_close(thinking[:, [0, 1, 2, 3, 4, 8]], dense[:, [0, 1, 2, 3, 4, 8]], atol=0, rtol=0)
        self.assertFalse(torch.allclose(thinking[:, 5:8], dense[:, 5:8]))
        kept = a.clone()
        kept[..., inner // 2:] = 0
        torch.testing.assert_close(static, mlp.c_proj(kept), atol=1e-6, rtol=0)

    def test_ties_go_to_the_lower_index(self):
        a = torch.tensor([[1.0, 3.0, 3.0, 3.0, 0.5]])
        keep = sparsity.oracle_keep(a, torch.ones(5), 2)
        self.assertEqual(keep.tolist(), [[False, True, True, False, False]])

    def test_parameter_counts(self):
        self.assertEqual(sparsity.active_params(1.0), 85_056_000)
        self.assertEqual(sparsity.active_params(0.5), 56_726_016)
        self.assertEqual(sparsity.active_params(0.25), 42_561_024)


class RunTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        os.environ["HF_HUB_OFFLINE"] = "1"
        try:
            FakeCODI()
        except OSError as error:
            raise unittest.SkipTest(f"GPT-2 tokenizer not cached: {error}")
        cls.saved = codi_module.CODIGPT2, scoring.score
        codi_module.CODIGPT2, scoring.score = FakeCODI, fake_score

    @classmethod
    def tearDownClass(cls):
        codi_module.CODIGPT2, scoring.score = cls.saved

    def files(self, folder):
        train = [problem(n, k) for n in (1, 2, 3) for k in range(1, 7)]
        (folder / "train.json").write_text(json.dumps(train))
        valid = [{"id": f"v/{i}", "rationale": " ".join(p["steps"]), **{k: p[k] for k in ("question", "answer")}}
                 for i, p in enumerate(problem(n, k) for n in (1, 2, 3) for k in (8, 9))]
        (folder / "valid.json").write_text(json.dumps(valid))
        (folder / "base").mkdir()
        return ["--checkpoint", folder / "base", "--train-file", folder / "train.json", "--valid-file",
                folder / "valid.json", "--calibration", 10, "--batch", 4, "--device", "cpu"]

    def test_calibrate_run_resume_and_report(self):
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            argv = self.files(folder) + ["--output", folder / "out"]
            run_script("run.py", argv)
            status = json.loads((folder / "out/status.json").read_text())
            self.assertEqual(status["calibration"]["records"], 10)
            self.assertEqual(set(status["arms"]), set(sparsity.ARMS))
            calibration = torch.load(folder / "out/calibration.pt", weights_only=True)
            self.assertEqual(tuple(calibration["importance"].shape), (12, 128))
            self.assertGreater(float(calibration["importance"].sum()), 0)
            self.assertEqual(tuple(calibration["mu"]["oracle-0.25"].shape), (12, 3, 128))
            records = [json.loads(x) for x in (folder / "out/evals/ID.jsonl").read_text().splitlines()]
            self.assertEqual(len(records), 6)
            self.assertEqual(records[0]["positions"][1], 7)  # <bot> and 6 latent steps
            overlap = json.loads((folder / "out/overlap.json").read_text())
            self.assertEqual(len(overlap["0.25"]["T"]), 12)
            self.assertTrue(all(0 <= x <= 1 for x in overlap["0.25"]["T"] + overlap["0.5"]["Q"]))
            self.assertGreater(sum(overlap["0.25"]["T"]), 0)
            stamps = {p.name: p.stat().st_mtime_ns for p in (folder / "out/evals").iterdir()}
            run_script("run.py", argv)  # resumes: nothing repeated
            self.assertEqual({p.name: p.stat().st_mtime_ns for p in (folder / "out/evals").iterdir()}, stamps)
            text = run_script("report.py", [folder / "out"])
            self.assertIn("SPARSE-0 verdict:", text)

    def test_identity_guard(self):
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            reference = folder / "reference.jsonl"
            reference.write_text("".join(json.dumps({"id": f"v/{i}", "greedy": {"prediction": "nonsense"}}) + "\n"
                                         for i in range(6)))
            with self.assertRaises(SystemExit):
                run_script("run.py", self.files(folder) + ["--output", folder / "out", "--reference", reference])
            self.assertFalse((folder / "out/evals/ID.jsonl").exists())
            self.assertFalse((folder / "out/evals/O50.jsonl").exists())


class RulesTest(unittest.TestCase):
    N, B = 400, 200

    def report(self, scores, phase=(0, 0)):
        """Synthetic arms: each arm right on its first `score` questions, except that O25-T and O25-Q are right on
        questions phase[0] (phase[1]) to B - 1."""
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            (folder / "evals").mkdir()
            scores = {"ID": self.B, **scores}
            for name in sparsity.ARMS:
                right = [i < scores.get(name, 0) for i in range(self.N)]
                if name in ("O25-T", "O25-Q"):
                    lost = phase[0] if name == "O25-T" else phase[1]
                    right = [lost <= i < self.B for i in range(self.N)]
                rows = [{"id": f"v/{i}", "rationale_steps": 2 + i % 4, "prediction": str(i if r else -1),
                         "correct": r, "positions": [50, 7, 6]} for i, r in enumerate(right)]
                (folder / f"evals/{name}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
            (folder / "status.json").write_text(json.dumps({"identity": {"questions": self.N, "matched": self.N}}))
            return run_script("report.py", [folder])

    def verdict(self, scores, phase=(0, 0)):
        return self.report(scores, phase).splitlines()[-1]

    def test_rules(self):
        go = {"O50": 199, "S50": 180, "O25": 185, "S25": 150}
        self.assertIn("STOP, no headroom (R1)", self.verdict({"O50": 180, "S50": 150, "O25": 120, "S25": 60}))
        self.assertIn("STOP, no routing value (R2)", self.verdict({"O50": 198, "S50": 195, "O25": 170, "S25": 165}))
        self.assertIn("GO (R3): D1-R at 25%", self.verdict({**go, "O25": 192}))
        self.assertIn("GO (R3): D1-R at 50%", self.verdict(go))
        self.assertIn("thinking is the sensitive phase", self.verdict(go, phase=(14, 2)))
        self.assertIn("reading is the sensitive phase", self.verdict(go, phase=(2, 14)))
        self.assertIn("no phase asymmetry at 25%", self.verdict(go, phase=(6, 2)))

    def test_pruning_note(self):
        self.assertIn("Pruning note", self.report({"O50": 198, "S50": 195, "O25": 170, "S25": 165}))
        self.assertNotIn("Pruning note", self.report({"O50": 198, "S50": 185, "O25": 170, "S25": 165}))


if __name__ == "__main__":
    unittest.main()
