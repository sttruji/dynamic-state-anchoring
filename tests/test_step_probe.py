"""Checks of the probe diagnosis (results/step_probe/PLAN.md). On a tiny random GPT-2, run.py extracts states, refuses a
mismatched reference, resumes without repeating chunks and writes readings of the right shape. report.py's rules give
READOUT-LIMITED, COMPUTE-LIMITED with a late break, and INCONCLUSIVE on synthetic readings."""
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

from dsa.bench import scoring  # noqa: E402
from dsa.bench.models import codi as codi_module  # noqa: E402
from tests.test_inner_loop_scripts import FakeCODI  # noqa: E402


def run_script(script, argv):
    old, out = sys.argv, io.StringIO()
    sys.argv = [script] + [str(a) for a in argv]
    try:
        with contextlib.redirect_stdout(out):
            runpy.run_path(str(ROOT / "scripts/step_probe" / script), run_name="__main__")
    finally:
        sys.argv = old
    return out.getvalue()


def fake_score(text, gold, answer_type="numeric"):
    """Right on even golds, otherwise one too many, so a random model still gives both kinds of question."""
    value = int(float(gold))
    right = value % 2 == 0
    return {"prediction": str(value if right else value + 1), "correct": right}


def problem(n, k):
    values = [k + j for j in range(1, n + 1)]
    steps = [f"<<{v - 1}+1={v}>>" for v in values]
    return {"question": f"Tom has {k} pens and gets {n} more each day." + " He smiles." * (k % 3) + " How many?",
            "steps": steps, "answer": str(values[-1])}


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
                folder / "valid.json", "--per-length", 4, "--max-steps", 3, "--chunk", 5, "--folds", 2,
                "--probe-steps", 5, "--device", "cpu"]

    def test_extract_resume_and_report(self):
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            argv = self.files(folder) + ["--output", folder / "out"]
            run_script("run.py", argv)
            status = json.loads((folder / "out/status.json").read_text())
            self.assertEqual(status["valid"]["questions"], 6)
            self.assertEqual(status["train"]["done"], 12)  # 4 per step count
            chunks = sorted((folder / "out/train").glob("chunk_*.jsonl"))
            self.assertEqual(len(chunks), 3)
            order = [json.loads(x)["index"] for path in chunks for x in path.read_text().splitlines()]
            source = json.loads((folder / "train.json").read_text())
            self.assertEqual([len(source[i]["steps"]) for i in order[:3]], [1, 2, 3])  # round-robin over step counts
            with np.load(folder / "out/readings.npz") as data:
                self.assertEqual(data["train"].shape, (12, 4, 4, 8))  # questions, targets, layers, positions
                self.assertEqual(data["valid"].shape, (6, 4, 4, 8))
            features = np.load(folder / "out/valid/features.npy")
            self.assertEqual(features.shape[1:3], (8, 4))
            run_script("run.py", argv)  # resumes: nothing repeated
            self.assertEqual(json.loads((folder / "out/status.json").read_text())["train"]["done"], 12)
            text = run_script("report.py", [folder / "out", "--max-steps", 3])
            self.assertIn("Step probe diagnosis (validation):", text)

    def test_identity_guard(self):
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            reference = folder / "reference.jsonl"
            reference.write_text("".join(json.dumps({"id": f"v/{i}", "greedy": {"prediction": "nonsense"}}) + "\n"
                                         for i in range(6)))
            with self.assertRaises(SystemExit):
                run_script("run.py", self.files(folder) + ["--output", folder / "out", "--reference", reference])
            self.assertFalse((folder / "out/valid/records.jsonl").exists())


class RulesTest(unittest.TestCase):
    MISS = 9999

    def verdict(self, case):
        """Synthetic 4-step questions, half right; readings set by case."""
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            (folder / "train").mkdir()
            (folder / "valid").mkdir()
            arrays = {}
            for split, count in (("train", 40), ("valid", 20)):
                records, readings = [], np.full((count, 7, 4, 8), self.MISS, dtype=np.int32)
                targets = np.full((count, 7), -1, dtype=np.int64)
                for i in range(count):
                    right, values = i % 2 == 0, [i + 1, i + 2, i + 3, 100 + i]
                    records.append({"index": i, "id": f"{split}/{i}", "steps": 4, "values": values, "gold": values[-1],
                                    "prediction": str(values[-1] if right else values[-1] + 7), "correct": right,
                                    "in_question": [False] * 4})
                    targets[i, :4], targets[i, -1] = values, values[-1]
                    if case != "inconclusive" or not right:
                        for j, v in enumerate(values):
                            readings[i, j, :, 1:] = v
                    if right and case != "inconclusive":
                        readings[i, -1, :, 6:] = values[-1]
                    elif not right and case == "readout":
                        readings[i, -1, :, 6:] = values[-1]
                    elif not right and case == "compute":
                        readings[i, -1, :, 6:] = values[-1] + 7
                        readings[i, 3, :, :] = self.MISS  # the last step cannot be read
                arrays[split], arrays[f"{split}_targets"] = readings, targets
                path = folder / ("train/chunk_00000.jsonl" if split == "train" else "valid/records.jsonl")
                path.write_text("".join(json.dumps(r) + "\n" for r in records))
            np.savez(folder / "readings.npz", layers=np.array([3, 6, 9, 12]), **arrays)
            return run_script("report.py", [folder])

    def test_rules(self):
        self.assertIn("(validation): READOUT-LIMITED, break profile NONE", self.verdict("readout"))
        self.assertIn("(validation): COMPUTE-LIMITED, break profile LATE", self.verdict("compute"))
        self.assertIn("(validation): INCONCLUSIVE", self.verdict("inconclusive"))


if __name__ == "__main__":
    unittest.main()
