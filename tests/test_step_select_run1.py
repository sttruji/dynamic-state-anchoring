"""Checks of Run 1 of step selection (results/step_select/run1/PLAN.md): run1.py on a tiny random GPT-2 (reruns at
rate 0 are greedy, features are saved per candidate, the identity check, resuming and the time cap), and
run1_report.py's verdicts on synthetic runs with and without a planted signal."""
import contextlib
import io
import json
import os
import runpy
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts/step_select"))

import stepsel  # noqa: E402,F401  (puts scripts/inner_loop on the path)
import looped  # noqa: E402
from dsa.bench.models import codi as codi_module  # noqa: E402
from tests.test_inner_loop_scripts import FakeCODI  # noqa: E402


def run_script(script, argv):
    old, out = sys.argv, io.StringIO()
    sys.argv = [script] + [str(a) for a in argv]
    try:
        with contextlib.redirect_stdout(out):
            runpy.run_path(str(ROOT / "scripts/step_select" / script), run_name="__main__")
    finally:
        sys.argv = old
    return out.getvalue()


class RunTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        os.environ["HF_HUB_OFFLINE"] = "1"
        try:
            FakeCODI()
        except OSError as error:
            raise unittest.SkipTest(f"GPT-2 tokenizer not cached: {error}")
        cls.saved = codi_module.CODIGPT2
        codi_module.CODIGPT2 = FakeCODI

    @classmethod
    def tearDownClass(cls):
        codi_module.CODIGPT2 = cls.saved

    def test_run_identity_resume_and_cap(self):
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            (folder / "base").mkdir()
            common = ["--checkpoint", folder / "base", "--dataset", "gsm8k_test", "--device", "cpu", "--reruns", 3]
            run_script("run1.py", common + ["--output", folder / "zero", "--limit", 3, "--p", 0.0])
            records = [json.loads(x) for x in (folder / "zero/records.jsonl").read_text().splitlines()]
            self.assertEqual(len(records), 3)
            for r in records:  # at rate 0 every rerun is greedy
                self.assertEqual([len(b) for b in r["branches"]], [3, 3, 3])
                self.assertTrue(all(b["prediction"] == r["greedy"]["prediction"] for bs in r["branches"] for b in bs))
            with np.load(folder / "zero/features/c00000.npz") as part:
                self.assertEqual(part["hidden"].shape, (3, 3, 4, 32))
                self.assertEqual(part["latent"].shape, (3, 3, 4, 32))
                self.assertEqual(part["hidden"].dtype, np.float16)
                np.testing.assert_allclose(part["hidden"][:, :, 1], part["hidden"][:, :, 0], atol=1e-3)

            reference = folder / "reference.jsonl"
            reference.write_text("\n".join(json.dumps({"id": r["id"], "plain": {"6": {"text": r["greedy"]["text"]}}})
                                           for r in records))
            checked = common + ["--output", folder / "checked", "--reference", reference, "--p", 0.2]
            run_script("run1.py", checked + ["--limit", 2])
            run_script("run1.py", checked + ["--limit", 3])  # resumes at question 2
            status = json.loads((folder / "checked/status.json").read_text())
            self.assertEqual((status["done"], status["sessions"], status["identity"]), (3, 2, 3))
            self.assertEqual(sorted(p.name for p in (folder / "checked/features").iterdir()), ["c00000.npz", "c00002.npz"])

            run_script("run1.py", common + ["--output", folder / "capped", "--limit", 3, "--max-minutes", 0])
            status = json.loads((folder / "capped/status.json").read_text())
            self.assertEqual((status["done"], status["capped"]), (0, True))


class ReportTest(unittest.TestCase):
    def synthetic(self, folder: Path, signal: float, seed: int = 0, n: int = 1319):
        """A complete run: greedy right 40% of the time, each rerun right 40% of the time, and features that carry
        `signal` times a fixed direction for right candidates."""
        rng = np.random.default_rng(seed)
        greedy = rng.random(n) < 0.4
        reruns = rng.random((n, 3, 7)) < 0.4
        labels = np.concatenate([np.repeat(greedy[:, None, None], 3, 1), reruns], -1)
        direction = rng.normal(size=16)
        features = rng.normal(size=(n, 3, 8, 16)) + signal * labels[..., None] * direction
        records = []
        for i in range(n):
            records.append({"id": f"q{i:05d}", "gold": "1",
                            "greedy": {"text": "", "prediction": "1" if greedy[i] else "0", "correct": bool(greedy[i]),
                                       "matches_reference": True},
                            "branches": [[{"prediction": "1" if c else f"w{i}-{s}-{k}", "correct": bool(c)}
                                          for k, c in enumerate(row)] for s, row in enumerate(reruns[i])]})
        (folder / "features").mkdir(parents=True)
        (folder / "records.jsonl").write_text("\n".join(json.dumps(r) for r in records) + "\n")
        half = features.astype(np.float16)
        np.savez(folder / "features/c00000.npz", ids=np.array([r["id"] for r in records]),
                 hidden=half[..., :8], latent=half[..., 8:])

    def test_planted_signal_goes_and_noise_kills(self):
        with tempfile.TemporaryDirectory() as folder:
            self.synthetic(Path(folder) / "signal", signal=3.0)
            text = run_script("run1_report.py", [Path(folder) / "signal"])
            self.assertIn("matches the reference on 1319/1319", text)
            self.assertIn("A: CONFIRM", text)  # right reruns agree, wrong ones scatter
            self.assertIn("B: GO", text)
            self.synthetic(Path(folder) / "noise", signal=0.0, seed=1)
            text = run_script("run1_report.py", [Path(folder) / "noise"])
            self.assertIn("B: KILL", text)

    def test_incomplete_run_cannot_go(self):
        with tempfile.TemporaryDirectory() as folder:
            self.synthetic(Path(folder) / "short", signal=3.0, n=700)
            text = run_script("run1_report.py", [Path(folder) / "short"])
            self.assertIn("no CONFIRM or GO can be declared", text)
            self.assertIn("NO VERDICT (incomplete run)", text)
            self.assertIn("B: REPORT", text)


if __name__ == "__main__":
    unittest.main()
