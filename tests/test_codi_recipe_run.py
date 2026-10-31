"""The Colab CODI-recipe driver's helpers and its notebook's script calls."""
import argparse
import ast
import contextlib
import io
import json
from pathlib import Path
import re
import runpy
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

from dsa.bench.data import EVAL_DATASETS, SUITES
from dsa.bench.harness import BASELINE_ORDER
from dsa.data.milestones import annotate_record
from dsa.data.tokenize import TinyTokenizer, tokenize_record
from dsa.training.pilot import sha256_file, tokenizer_directory_sha256


ROOT = Path(__file__).resolve().parents[1]
DRIVER = runpy.run_path(str(ROOT / "scripts/run_codi_recipe.py"))
NOTEBOOK = ROOT / "notebooks/DSA_CODI_Recipe_Colab.ipynb"


class DriverHelperTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def checkpoint(self, run, step):
        path = run / f"checkpoint-{step:08d}"
        path.mkdir(parents=True)
        (path / "COMPLETE.json").write_text(json.dumps({"step": step}))
        (run / "latest.json").write_text(json.dumps({"path": path.name, "step": step}))
        (run / "metrics.jsonl").write_text("".join(json.dumps({"step": s}) + "\n" for s in range(1, step + 2)))

    def test_backup_keeps_only_the_latest_checkpoint_and_restore_resumes_from_it(self):
        run, backup = self.root / "local/run", self.root / "drive/backup"
        self.assertIsNone(DRIVER["backup_checkpoint"](run, backup))
        self.checkpoint(run, 2)
        self.assertEqual(DRIVER["backup_checkpoint"](run, backup), "checkpoint-00000002")
        self.checkpoint(run, 4)
        DRIVER["backup_checkpoint"](run, backup)
        self.assertEqual(sorted(path.name for path in backup.iterdir()),
                         ["checkpoint-00000004", "latest.json", "metrics.jsonl"])
        self.assertEqual(sorted(path.name for path in backup.parent.iterdir()), ["backup"])
        fresh = self.root / "new-machine/run"
        self.assertTrue(DRIVER["restore_checkpoint"](backup, fresh))
        self.assertEqual(json.loads((fresh / "latest.json").read_text())["step"], 4)
        self.assertTrue((fresh / "checkpoint-00000004/COMPLETE.json").exists())
        self.assertFalse(DRIVER["restore_checkpoint"](backup, fresh))  # local run wins

    def test_longest_records_set_is_a_valid_prepared_split(self):
        prepared = self.root / "prepared"
        tokenizer = TinyTokenizer()
        tokenizer.save_pretrained(prepared / "tokenizer")
        raw = [{"question": "Q" * size, "answer": "<<1+1=2>>\n<<2+1=3>>\n#### 3"} for size in (5, 40, 12, 30)]
        lines = [json.dumps(tokenize_record(annotate_record(row), tokenizer, 4)) for row in raw]
        (prepared / "train.tokenized.jsonl").write_text("\n".join(lines) + "\n")
        for split in ("valid", "test"):
            (prepared / f"{split}.jsonl").write_text("{}\n")
        manifest = {"schema_version": 1, "tiny": True, "loop_budget": 4,
                    "tokenizer_sha256": tokenizer_directory_sha256(prepared / "tokenizer"),
                    "splits": {split: {"path": path, "sha256": sha256_file(prepared / path)}
                               for split, path in (("train", "train.tokenized.jsonl"),
                                                   ("valid", "valid.jsonl"), ("test", "test.jsonl"))}}
        (prepared / "manifest.json").write_text(json.dumps(manifest))
        output = DRIVER["longest_records_set"](prepared, 2, self.root / "longest")
        records = [json.loads(line) for line in (output / "train.tokenized.jsonl").read_text().splitlines()]
        self.assertEqual([len(record["question_input_ids"]) for record in records], [41, 31])
        saved = json.loads((output / "manifest.json").read_text())
        self.assertEqual(saved["splits"]["train"]["sha256"], sha256_file(output / "train.tokenized.jsonl"))
        self.assertEqual(saved["tokenizer_sha256"], tokenizer_directory_sha256(output / "tokenizer"))

    def test_paired_counts_and_exact_sign_test(self):
        left = {str(i): i < 5 for i in range(10)}
        right = {str(i): False for i in range(10)}
        counts = DRIVER["paired_counts"](left, right)
        self.assertEqual((counts["only_left"], counts["only_right"]), (5, 0))
        self.assertAlmostEqual(counts["p"], 2 / 32)
        self.assertEqual(DRIVER["paired_counts"](right, right)["p"], 1.0)

    def test_evaluate_runs_every_loop_budget(self):
        args = argparse.Namespace(name="run", tiny=True, suite="dev", device="cpu", work_dir=self.root / "run",
                                  results_dir=self.root / "results")
        with mock.patch("subprocess.run") as run:
            DRIVER["evaluate"](args)
        commands = [call.args[0] for call in run.call_args_list]
        self.assertEqual([command[command.index("--name") + 1] for command in commands],
                         ["run-k8", "run-k4", "run-k1"])
        self.assertNotIn("--arg", commands[0][commands[0].index("--name"):])
        self.assertIn("loops=4", commands[1])
        self.assertIn("loops=1", commands[2])

    def test_summary_compares_every_pair_of_loop_budgets(self):
        for loops, right in ((8, 3), (4, 2), (1, 1)):
            path = self.root / "results/predictions" / f"run-k{loops}/dev64.jsonl"
            path.parent.mkdir(parents=True)
            rows = [{"_identity": {}}] + [{"id": str(i), "correct": i < right} for i in range(4)]
            path.write_text("".join(json.dumps(row) + "\n" for row in rows))
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            DRIVER["summarize"](argparse.Namespace(name="run", results_dir=self.root / "results"))
        text = output.getvalue()
        self.assertIn("dev64: k8 75.0%, k4 50.0%, k1 25.0%", text)
        self.assertIn("k8 vs k4: only k8 right 1, only k4 right 0", text)
        self.assertIn("k8 vs k1: only k8 right 2, only k1 right 0", text)
        self.assertIn("k4 vs k1: only k4 right 1, only k1 right 0", text)


class NotebookTests(unittest.TestCase):
    def test_cells_parse_without_saved_outputs_and_call_real_flags(self):
        notebook = json.loads(NOTEBOOK.read_text())
        calls = {}
        for cell in notebook["cells"]:
            if cell["cell_type"] != "code":
                continue
            self.assertEqual(cell["outputs"], [])
            self.assertIsNone(cell["execution_count"])
            tree = ast.parse("".join(cell["source"]))
            for node in ast.walk(tree):
                if (isinstance(node, ast.Call) and getattr(node.func, "id", None) == "run" and node.args
                        and isinstance(node.args[0], ast.Constant)):
                    flags = {item.value for item in ast.walk(node) if isinstance(item, ast.Constant)
                             and isinstance(item.value, str) and item.value.startswith("--")}
                    calls.setdefault(node.args[0].value, set()).update(flags)
        self.assertIn("scripts/run_codi_recipe.py", calls)
        for script, flags in calls.items():
            usage = subprocess.run([sys.executable, str(ROOT / script), "--help"], cwd=ROOT,
                                   capture_output=True, text=True, check=True).stdout
            for flag in flags:
                self.assertIn(flag, usage, f"{script} has no {flag}")

    def test_baseline_cell_covers_every_baseline_and_evaluation_set(self):
        notebook = json.loads(NOTEBOOK.read_text())
        source = next("".join(cell["source"]) for cell in notebook["cells"] if "RUN_BASELINES:" in "".join(cell["source"]))
        models = ast.literal_eval(re.search(r"for model in (\(.*\)):", source).group(1))
        self.assertEqual(set(models), set(BASELINE_ORDER) - {"lotus"})  # LOTUS has no public GPT-2 weights
        suites = re.search(r'"--suite", "([^"]+)", "--test"', source).group(1).split(",")
        self.assertEqual({name for suite in suites for name in SUITES[suite]}, set(EVAL_DATASETS))


if __name__ == "__main__":
    unittest.main()
