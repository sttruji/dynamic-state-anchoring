"""Checks of Stage 0 of H, answer-conditioned plan search (results/plan_search/stage0/PLAN.md). On a tiny random GPT-2,
stage0.py passes its identity check, keeps every plan within the dropout radius, picks a wrong target near gold, and
resumes without repeating questions. report.py's gate gives GO and both kinds of KILL on synthetic searches."""
import contextlib
import io
import json
import os
import runpy
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dsa.bench.models import codi as codi_module  # noqa: E402
from tests.test_inner_loop_scripts import FakeCODI  # noqa: E402


def run_script(script, argv):
    old, out = sys.argv, io.StringIO()
    sys.argv = [script] + [str(a) for a in argv]
    try:
        with contextlib.redirect_stdout(out):
            runpy.run_path(str(ROOT / "scripts/plan_search" / script), run_name="__main__")
    finally:
        sys.argv = old
    return out.getvalue()


class SearchTest(unittest.TestCase):
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

    def test_search_stays_in_radius_and_resumes(self):
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            rows = [{"question": f"Mia has {n} apples and buys {n + 2} more." + " She is happy." * (n % 3) +
                                 " How many apples now?", "answer": str(2 * n + 2)}  # varied lengths exercise the padding
                    for n in range(3, 15)] + [{"question": "What is half of 3?", "answer": "1.5"}]  # a non-integer gold is skipped
            (folder / "train.json").write_text(json.dumps(rows))
            (folder / "base").mkdir()
            argv = ["--checkpoint", folder / "base", "--output", folder / "out", "--train-file", folder / "train.json",
                    "--wrong", 3, "--components", 8, "--steps", 3, "--samples", 2, "--window", 5, "--check", 3,
                    "--device", "cpu"]
            run_script("stage0.py", argv)
            phase1 = json.loads((folder / "out/phase1.json").read_text())
            self.assertEqual(phase1["identity_checked"], 3)
            self.assertEqual(phase1["wrong"], 3)  # a random tiny model gets these wrong
            searched = [json.loads(x) for x in (folder / "out/search.jsonl").read_text().splitlines()]
            self.assertEqual(len(searched), 3)
            for r in searched:
                self.assertGreater(r["radius"], 0.0)
                for name in ("gold", "wrong"):
                    self.assertLessEqual(r[name]["distance"], r["radius"] * (1 + 1e-4))
                self.assertEqual(r["gold"]["target"], r["gold_answer"])
                self.assertNotEqual(r["wrong"]["target"], r["gold_answer"])
                self.assertLessEqual(abs(r["wrong"]["target"] - r["gold_answer"]), 5)
            self.assertTrue(all(r[name]["distance"] > 0 for r in searched for name in ("gold", "wrong")))  # z1 moved
            run_script("stage0.py", argv)  # resumes: nothing searched twice
            again = (folder / "out/search.jsonl").read_text().splitlines()
            self.assertEqual(len(again), 3)
            text = run_script("report.py", [folder / "out"])
            self.assertIn("H Stage 0: KILL", text)  # three questions are under the floor of 150


class GateTest(unittest.TestCase):
    def gate(self, gold, wrong, n=1000):
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            (folder / "phase1.json").write_text(json.dumps({"processed": 4000, "wrong": n, "identity_checked": 20,
                                                            "seconds": 300.0}))
            (folder / "greedy.jsonl").write_text(json.dumps({"index": 0, "correct": True}) + "\n")
            lines = []
            for i in range(n):
                entry = lambda ok: {"target": 1, "accepted": ok, "prediction": "1", "final_logp": -1.0,
                                    "distance": 0.5, "at_boundary": False}
                lines.append(json.dumps({"index": i, "gold_answer": 1, "greedy_prediction": "2", "radius": 1.0,
                                         "base_logp_gold": -3.0, "base_logp_wrong": -3.1,
                                         "gold": entry(i < gold), "wrong": entry(i < wrong)}))
            (folder / "search.jsonl").write_text("\n".join(lines) + "\n")
            return run_script("report.py", [folder])

    def test_gate(self):
        self.assertIn("H Stage 0: GO", self.gate(200, 50))
        self.assertIn("KILL (gold under 1.5x wrong)", self.gate(200, 150))
        self.assertIn("KILL (fewer than 150 gold plans)", self.gate(100, 0))


if __name__ == "__main__":
    unittest.main()
