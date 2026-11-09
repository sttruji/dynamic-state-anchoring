"""Checks of D1-R (results/ffn_sparsity/d1r/PLAN.md). The dense hook is GPT-2 exactly; the MoE mask equals a gathered
sparse FFN; the hash arm depends only on its keys; Prune is SPARSE-0's static arm bit for bit. On a tiny random GPT-2,
convert.py builds equal experts from SPARSE-0's calibration and refuses the wrong one, the untrained Prune arm repeats
SPARSE-0's S50 text for text, train_moe.py trains and checks itself against a reference, and eval_moe.py guards its
reference. report_d1r.py's rules give each verdict on synthetic arms."""
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
sys.path.insert(0, str(ROOT / "scripts/inner_loop"))

import looped  # noqa: E402
import moe  # noqa: E402
import sparsity  # noqa: E402
from dsa.bench import scoring  # noqa: E402
from dsa.bench.models import codi as codi_module  # noqa: E402
from tests.test_ffn_sparsity import RunTest as SparseRun, tiny_gpt2  # noqa: E402
from tests.test_inner_loop_scripts import FakeCODI, fake_decoder  # noqa: E402
from tests.test_step_probe import fake_score  # noqa: E402


def run_script(script, argv):
    old, out = sys.argv, io.StringIO()
    sys.argv = [script] + [str(a) for a in argv]
    try:
        with contextlib.redirect_stdout(out):
            runpy.run_path(str(ROOT / "scripts/ffn_sparsity" / script), run_name="__main__")
    finally:
        sys.argv = old
    return out.getvalue()


def synthetic_conversion(layers=2, inner=128, width=32, keys=120):
    """32 shared neurons, 12 experts of 8, Prune on the first 64; random routers and stand-in values."""
    torch.manual_seed(3)
    owner = torch.full((layers, inner), -1, dtype=torch.long)
    owner[:, 32:] = (torch.arange(inner - 32) // 8)[None]
    shared, prune = torch.zeros(layers, inner, dtype=torch.bool), torch.zeros(layers, inner, dtype=torch.bool)
    shared[:, :32], prune[:, :64] = True, True
    return {"shared": shared, "prune": prune, "owner": owner, "hash": moe.hash_table(layers, keys),
            "routers": [moe.make_router(width).state_dict() for _ in range(layers)],
            "mu": {name: torch.randn(layers, 3, inner) * 0.1 for name in ("moe", "hash", "prune")}}


class MaskTest(unittest.TestCase):
    def setUp(self):
        self.model = tiny_gpt2()
        self.ids = torch.randint(0, 100, (2, 9))
        self.conversion = synthetic_conversion()

    def masks(self):
        masks = moe.MoEMasks(self.model, self.conversion)
        masks.phase, masks.keys = sparsity.phase_tags(5, 3, 1), torch.randint(0, 120, (2, 9))
        return masks

    def test_dense_identity(self):
        with torch.no_grad():
            dense = self.model(self.ids).logits
            self.masks()
            self.assertTrue(torch.equal(self.model(self.ids).logits, dense))

    def test_moe_equals_gathered(self):
        masks = self.masks()
        masks.arm = "moe"
        mlp, x = masks.blocks[1], torch.randn(2, 9, 32)
        with torch.no_grad():
            masked = mlp(x)
            a = mlp.act(mlp.c_fc(x))
            chosen = moe.top_experts(masks.routers[1](x))
            weight, bias = mlp.c_proj.weight, mlp.c_proj.bias
            for b in range(2):
                for s in range(9):
                    mu = self.conversion["mu"]["moe"][1, masks.phase[s]]
                    on = [e for e in range(12) if chosen[b, s, e]]
                    kept = torch.cat([torch.arange(32)] + [torch.arange(32 + 8 * e, 40 + 8 * e) for e in on])
                    gathered = (a[b, s, kept] - mu[kept]) @ weight[kept] + mu @ weight + bias
                    torch.testing.assert_close(masked[b, s], gathered, atol=1e-5, rtol=0)
        self.assertTrue((chosen.sum(-1) == 4).all())

    def test_hash_follows_keys(self):
        masks = self.masks()
        masks.arm = "hash"
        x = torch.randn(2, 9, 32)
        first = masks.chosen(0, x)
        self.assertTrue(torch.equal(first, masks.chosen(0, torch.randn(2, 9, 32))))
        self.assertTrue(torch.equal(first, self.conversion["hash"][0][masks.keys]))
        self.assertTrue((self.conversion["hash"].sum(-1) == 4).all())

    def test_prune_is_sparse0_static(self):
        x = torch.randn(2, 9, 32)
        masks = self.masks()
        masks.arm = "prune"
        with torch.no_grad():
            ours = masks.blocks[0](x)
        reference = sparsity.FFNMasks(tiny_gpt2())
        reference.static[0.5] = self.conversion["prune"]
        reference.mu[("static", 0.5)] = self.conversion["mu"]["prune"]
        reference.arm, reference.phase = ("static", 0.5, "QTA"), masks.phase
        with torch.no_grad():
            self.assertTrue(torch.equal(reference.blocks[0](x), ours))

    def test_gradients_reach_kept_neurons_only(self):
        masks = self.masks()
        masks.arm = "prune"
        mlp = masks.blocks[0]
        mlp(torch.randn(2, 9, 32)).sum().backward()
        grad = mlp.c_fc.weight.grad  # Conv1D: [width, inner]
        self.assertTrue((grad[:, 64:] == 0).all())
        self.assertTrue((grad[:, :64] != 0).any())

    def test_parameter_counts(self):
        self.assertEqual(moe.router_params(), 1_199_760)
        self.assertEqual(moe.moe_active_params("moe"), 56_726_016 + 1_199_760)
        self.assertEqual(moe.moe_active_params("hash"), 56_726_016)
        self.assertEqual(moe.moe_active_params(None), 85_056_000)


class PipelineTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        os.environ["HF_HUB_OFFLINE"] = "1"
        try:
            FakeCODI()
        except OSError as error:
            raise unittest.SkipTest(f"GPT-2 tokenizer not cached: {error}")
        cls.saved = codi_module.CODIGPT2, scoring.score, looped.load_step_decoder
        codi_module.CODIGPT2, scoring.score, looped.load_step_decoder = FakeCODI, fake_score, fake_decoder

    @classmethod
    def tearDownClass(cls):
        codi_module.CODIGPT2, scoring.score, looped.load_step_decoder = cls.saved

    def test_pipeline(self):
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            common = SparseRun.files(None, folder)
            run_script("run.py", common + ["--output", folder / "sparse0"])
            calibration = folder / "sparse0/calibration.pt"
            base = ["--checkpoint", folder / "base", "--device", "cpu"]
            valid = ["--valid-file", folder / "valid.json"]
            with self.assertRaises(SystemExit):
                run_script("convert.py", base + ["--calibration", calibration, "--sha256", "0" * 64,
                                                 "--output", folder / "bad/conversion.pt"])
            convert = base + ["--calibration", calibration, "--output", folder / "conv/conversion.pt",
                              "--train-file", folder / "train.json", "--batch", 4, "--features", 64,
                              "--router-steps", 5, "--kmeans-iterations", 3]
            run_script("convert.py", convert)
            conversion = torch.load(folder / "conv/conversion.pt", weights_only=True)
            for layer in range(12):
                self.assertEqual(int(conversion["shared"][layer].sum()), 32)
                self.assertEqual(int(conversion["prune"][layer].sum()), 64)
                self.assertEqual(torch.bincount(conversion["owner"][layer][conversion["owner"][layer] >= 0]).tolist(),
                                 [8] * 12)
                self.assertTrue(torch.equal(conversion["owner"][layer] < 0, conversion["shared"][layer]))
            self.assertTrue(torch.equal(conversion["mu"]["prune"],
                                        torch.load(calibration, weights_only=True)["mu"]["static"]))
            summary = json.loads((folder / "conv/conversion.json").read_text())
            self.assertEqual(len(summary["layers"]), 12)

            evals, conv = folder / "evals", ["--conversion", folder / "conv/conversion.pt"]
            run_script("eval_moe.py", base + valid + conv + [
                "--arm", "prune", "--output", evals / "prune-0.jsonl", "--reference", folder / "sparse0/evals/S50.jsonl"])
            ours = [json.loads(x)["text"] for x in (evals / "prune-0.jsonl").read_text().splitlines()]
            theirs = [json.loads(x)["text"] for x in (folder / "sparse0/evals/S50.jsonl").read_text().splitlines()]
            self.assertEqual(ours, theirs)  # the untrained Prune arm is SPARSE-0's S50
            mismatched = folder / "nonsense.jsonl"
            mismatched.write_text("".join(json.dumps({"id": f"v/{i}", "prediction": "x"}) + "\n" for i in range(6)))
            with self.assertRaises(SystemExit):
                run_script("eval_moe.py", base + valid + ["--arm", "dense", "--output", evals / "base-dense.jsonl",
                                                          "--reference", mismatched])
            self.assertFalse((evals / "base-dense.jsonl").exists())

            train = ["--checkpoint", folder / "base", "--train-file", folder / "train.json", "--records", 8,
                     "--batch", 4, "--device", "cpu", "--seed", 7]
            run_script("train_moe.py", train + conv + ["--arm", "moe", "--output", folder / "arms/moe-s7"])
            self.assertTrue((folder / "arms/moe-s7/model/pytorch_model.bin").exists())
            run_script("eval_moe.py", ["--checkpoint", folder / "arms/moe-s7/model", "--device", "cpu"] + valid + conv
                       + ["--arm", "moe", "--output", evals / "moe-s7.jsonl"])
            agreement = json.loads((evals / "moe-s7.json").read_text())["oracle_agreement"]
            self.assertEqual(len(agreement), 36)  # 12 layers, 3 phases
            self.assertTrue(all(0 <= v <= 1 for v in agreement.values()))

            run_script("train_moe.py", train + ["--arm", "dense", "--output", folder / "arms/dense"])
            reference = folder / "arms/dense/metrics.jsonl"
            run_script("train_moe.py", train + ["--arm", "dense", "--output", folder / "arms/check", "--stop-after", 1,
                                                "--check", reference])
            self.assertTrue(json.loads((folder / "arms/check/check.json").read_text())["passed"])
            self.assertFalse((folder / "arms/check/model").exists())
            shifted = folder / "shifted.jsonl"
            shifted.write_text("".join(json.dumps({**json.loads(x), "loss": json.loads(x)["loss"] + 1}) + "\n"
                                       for x in reference.read_text().splitlines()))
            with self.assertRaises(SystemExit):
                run_script("train_moe.py", train + ["--arm", "dense", "--output", folder / "arms/check2",
                                                    "--stop-after", 1, "--check", shifted])


class RulesTest(unittest.TestCase):
    N = 500

    def verdict(self, scores, checks=True):
        """scores: (arm, seed) -> (first, last), the range of questions an arm gets right. ft6 is right on 0..199."""
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            (folder / "evals").mkdir()
            (folder / "training/dense-s7-check").mkdir(parents=True)

            def rows(first, last):
                return [{"id": f"v/{i}", "rationale_steps": 2 + i % 4, "prediction": str(i),
                         "correct": first <= i < last, "positions": [50, 7, 6]} for i in range(self.N)]

            ft6 = []
            for seed in (7, 8):
                path = folder / f"ft6-{seed}.jsonl"
                path.write_text("".join(json.dumps({"id": r["id"], "rationale_steps": r["rationale_steps"],
                                                    "arms": {"plain": {"correct": r["correct"]}}}) + "\n"
                                        for r in rows(0, 200)))
                ft6.append(path)
            for name, span in {"moe-0": (0, 50), "hash-0": (0, 10), **scores}.items():
                name = name if isinstance(name, str) else f"{name[0]}-s{name[1]}"
                (folder / f"evals/{name}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows(*span)))
            for name, n in (("base-dense", 500), ("prune-0", 100)):
                (folder / f"evals/{name}.json").write_text(json.dumps({"questions": n, "matched": n}))
            (folder / "training/dense-s7-check/check.json").write_text(json.dumps(
                {"passed": checks, "updates": [0, 10], "max_gap": 0.001, "tolerance": 0.02}))
            return run_script("report_d1r.py", [folder, "--ft6", *ft6]).splitlines()[-1]

    def arms(self, moe_, prune, hash_):
        return {(arm, seed): span for arm, span in (("moe", moe_), ("prune", prune), ("hash", hash_)) for seed in (7, 8)}

    def test_rules(self):
        self.assertIn("MoE-MATCH", self.verdict(self.arms((0, 200), (0, 180), (0, 190))))
        self.assertIn("PRUNE-SUFFICES", self.verdict(self.arms((0, 200), (0, 198), (0, 190))))
        self.assertIn("MATCH-UNRESOLVED", self.verdict(self.arms((0, 200), (0, 192), (0, 190))))
        self.assertIn("CAPACITY-NOT-ROUTING", self.verdict(self.arms((0, 200), (0, 180), (0, 197))))
        self.assertIn("NEGATIVE: MoE does not match", self.verdict(self.arms((0, 193), (0, 180), (0, 190))))
        self.assertIn("INCONCLUSIVE", self.verdict(self.arms((60, 257), (0, 180), (0, 190))))
        self.assertIn("early kill", self.verdict({("moe", 7): (0, 190)}))
        self.assertIn("INCOMPLETE", self.verdict({("moe", 7): (0, 200)}))
        self.assertIn("NO VERDICT", self.verdict(self.arms((0, 200), (0, 180), (0, 190)), checks=False))


if __name__ == "__main__":
    unittest.main()
