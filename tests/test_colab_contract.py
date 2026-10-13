"""Notebook contracts that can be checked without Colab, Drive, or a GPU."""

import ast
from contextlib import redirect_stdout
import hashlib
import io
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

from packaging.requirements import Requirement
from packaging.version import Version


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks" / "DSA_Colab.ipynb"


class ColabNotebookContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
        cls.cells = cls.notebook["cells"]
        cls.code_cells = [cell for cell in cls.cells if cell["cell_type"] == "code"]
        cls.stages = {}
        for index, cell in enumerate(cls.cells):
            if cell["cell_type"] == "code":
                for tag in cell.get("metadata", {}).get("tags", []):
                    cls.stages.setdefault(tag, []).append((index, cell))

    @staticmethod
    def source(cell):
        return "".join(cell["source"])

    def stage_source(self, tag):
        self.assertEqual(len(self.stages.get(tag, [])), 1, f"one code cell tagged {tag}")
        return self.source(self.stages[tag][0][1])

    def test_first_cell_clones_published_repository(self):
        first = self.cells[0]
        self.assertEqual(first["cell_type"], "code")
        self.assertIn("clone", first["metadata"]["tags"])
        tree = ast.parse(self.source(first))
        strings = [node.value for node in ast.walk(tree)
                   if isinstance(node, ast.Constant) and isinstance(node.value, str)]
        self.assertIn("clone", strings)
        self.assertTrue(any(value.rstrip("/").removesuffix(".git") ==
                            "https://github.com/sttruji/dynamic-state-anchoring"
                            for value in strings))

    def test_code_is_parseable_and_notebook_has_no_saved_execution(self):
        self.assertEqual(self.notebook["nbformat"], 4)
        self.assertGreaterEqual(self.notebook["nbformat_minor"], 5)
        ids = [cell["id"] for cell in self.cells]
        self.assertEqual(len(ids), len(set(ids)))
        for identifier in ids:
            self.assertRegex(identifier, r"^[A-Za-z0-9_-]{1,64}$")
        for index, cell in enumerate(self.code_cells):
            with self.subTest(code_cell=index):
                compile(self.source(cell), f"{NOTEBOOK.name}:cell-{index}", "exec")
                self.assertIsNone(cell["execution_count"])
                self.assertEqual(cell["outputs"], [])

    def test_workflow_orders_setup_before_download_training_and_evaluation(self):
        tags = ("clone", "dependencies", "drive", "config", "preflight", "data",
                "pretrained-smoke", "training", "evaluation", "final-evaluation", "summary")
        positions = []
        for tag in tags:
            self.stage_source(tag)
            positions.append(self.stages[tag][0][0])
        self.assertEqual(positions, sorted(positions))

    def test_research_imports_are_isolated_from_the_notebook_kernel(self):
        research_packages = {"dsa", "torch", "transformers", "tokenizers",
                             "huggingface_hub", "numpy"}
        for index, cell in enumerate(self.code_cells):
            tree = ast.parse(self.source(cell))
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    names = {alias.name.split(".")[0] for alias in node.names}
                elif isinstance(node, ast.ImportFrom):
                    names = {(node.module or "").split(".")[0]}
                else:
                    continue
                self.assertFalse(names & research_packages,
                                 f"research import in notebook kernel cell {index}")
        dependencies = self.stage_source("dependencies")
        self.assertIn("--system-site-packages", dependencies)
        self.assertIn("requirements-colab.txt", dependencies)
        self.assertNotIn("requirements-dsa.lock.txt", dependencies)

    def test_dependency_cell_runs_the_targeted_setup_helper(self):
        runner = mock.Mock()
        namespace = {"Path": Path, "sys": sys, "subprocess": runner, "REPO_DIR": ROOT}
        with redirect_stdout(io.StringIO()):
            exec(compile(self.stage_source("dependencies"), "notebook-dependencies", "exec"), namespace)
        runner.run.assert_called_once_with(
            [sys.executable, str(ROOT / "scripts/setup_colab.py"),
             "--venv-dir", "/content/dsa-venv",
             "--requirements", str(ROOT / "requirements-colab.txt")], check=True)
        self.assertEqual(namespace["PYTHON"], "/content/dsa-venv/bin/python")
        self.assertTrue((ROOT / "scripts/setup_colab.py").is_file())

    def test_drive_and_hugging_face_persistence_precede_model_use(self):
        drive = self.stage_source("drive")
        tree = ast.parse(drive)
        self.assertTrue(any(isinstance(node, ast.Call)
                            and isinstance(node.func, ast.Attribute)
                            and node.func.attr == "mount" for node in ast.walk(tree)))
        before_preflight = "\n".join(self.source(cell) for index, cell in enumerate(self.cells)
                                      if cell["cell_type"] == "code"
                                      and index < self.stages["preflight"][0][0])
        self.assertIn("HF_HOME", before_preflight)
        self.assertIn("MyDrive", before_preflight)

    def test_test_set_evaluation_requires_explicit_opt_in(self):
        final_source = self.stage_source("final-evaluation")
        tree = ast.parse(final_source)
        assignments = [node for node in ast.walk(tree) if isinstance(node, ast.Assign)]
        self.assertTrue(any(any(isinstance(target, ast.Name) and target.id == "RUN_TEST"
                                for target in node.targets)
                            and isinstance(node.value, ast.Constant)
                            and node.value.value is False for node in assignments))
        self.assertTrue(any(isinstance(node, ast.If)
                            and isinstance(node.test, ast.Name)
                            and node.test.id == "RUN_TEST" for node in ast.walk(tree)))

    def test_embedded_runtime_check_runs_and_writes_valid_json_on_cpu(self):
        tree = ast.parse(self.stage_source("preflight"))
        runtime_code = next(node.value.value for node in tree.body
                            if isinstance(node, ast.Assign)
                            and any(isinstance(target, ast.Name) and target.id == "runtime_code"
                                    for target in node.targets))
        compile(runtime_code, "notebook-embedded-runtime-check", "exec")
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / "runtime.json"
            result = subprocess.run([sys.executable, "-c", runtime_code, "cpu", str(output)],
                                    cwd=ROOT, text=True, capture_output=True, timeout=60)
            self.assertEqual(result.returncode, 0, result.stderr)
            report = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(report["device"], "cpu")
            self.assertIs(report["tiny_gpt2_forward_passed"], True)
            self.assertIn("torch", report["packages"])
            self.assertTrue(output.read_bytes().endswith(b"\n"))

    def test_runtime_check_explains_a_missing_cuda_device(self):
        tree = ast.parse(self.stage_source("preflight"))
        runtime_code = next(node.value.value for node in tree.body
                            if isinstance(node, ast.Assign)
                            and any(isinstance(target, ast.Name) and target.id == "runtime_code"
                                    for target in node.targets))
        with tempfile.TemporaryDirectory() as temp:
            result = subprocess.run(
                [sys.executable, "-c", runtime_code, "cuda", str(Path(temp) / "runtime.json")],
                cwd=ROOT, text=True, capture_output=True, timeout=60,
                env={**os.environ, "CUDA_VISIBLE_DEVICES": ""})
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('"cuda_available": false', result.stdout)
        self.assertIn("CUDA is unavailable", result.stderr)
        self.assertIn("T4 GPU", result.stderr)

    def test_notebook_runner_surfaces_and_saves_the_underlying_error(self):
        with tempfile.TemporaryDirectory() as temp, redirect_stdout(io.StringIO()):
            namespace = {"Path": Path, "sys": sys, "os": os, "json": json,
                         "hashlib": hashlib,
                         "REPO_DIR": ROOT, "REPO_COMMIT": "fixture-commit",
                         "PROJECT_DIR": Path(temp), "PYTHON": sys.executable}
            config_source = self.stage_source("config").replace('RUN_MODE = "auto_pilot"', 'RUN_MODE = "train_pilot"')
            exec(compile(config_source, "notebook-config", "exec"), namespace)
            with self.assertRaisesRegex(RuntimeError, "underlying child error"):
                namespace["run_project"](
                    "-c", "import sys; print('underlying child error', file=sys.stderr); sys.exit(9)")
            logs = list((namespace["REVIEW_ROOT"] / "logs").glob("*.log"))
            self.assertEqual(len(logs), 1)
            self.assertIn("underlying child error", logs[0].read_text())

    def test_disabled_test_and_summary_cells_execute_without_external_access(self):
        def forbidden_run(*args):
            self.fail(f"Disabled test evaluation attempted to execute {args}")

        with tempfile.TemporaryDirectory() as temp, redirect_stdout(io.StringIO()):
            prepared = Path(temp) / "prepared"
            report_root = Path(temp) / "reports"
            run_root = Path(temp) / "runs"
            prepared.mkdir()
            report_root.mkdir()
            (run_root / "static").mkdir(parents=True)
            (prepared / "manifest.json").write_text('{"fixture":true}\n', encoding="utf-8")
            metrics = [{"step": 1, "seconds": 0.2, "peak_process_rss_bytes": 100,
                        "peak_cuda_allocated_bytes": None, "total": 2.0, "write_rate": 0.0},
                       {"step": 2, "seconds": 0.3, "peak_process_rss_bytes": 120,
                        "peak_cuda_allocated_bytes": None, "total": 1.0, "write_rate": 0.5}]
            (run_root / "static" / "metrics.jsonl").write_text(
                "".join(json.dumps(row) + "\n" for row in metrics), encoding="utf-8")
            namespace = {"json": json, "hashlib": hashlib, "run_project": forbidden_run,
                         "config": {"fixture": True}, "PREPARED_LOCAL": prepared,
                         "REPORT_ROOT": report_root, "RUN_ROOT": run_root,
                         "validation_reports": [{"arm": "static", "split": "valid",
                             "loop_budget": 4, "total": 32, "accuracy": 0.5,
                             "invalid_answer_rate": 0.0, "average_latency_seconds": 0.01}],
                         "training_reports": {"static": {"step": 2, "fixture": True}}}
            exec(compile(self.stage_source("final-evaluation"), "notebook-final-evaluation", "exec"),
                 namespace)
            self.assertEqual(namespace["final_reports"], [])
            exec(compile(self.stage_source("summary"), "notebook-summary", "exec"), namespace)
            output = report_root / "notebook-summary.json"
            summary = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(summary["final_test"], [])
            self.assertEqual(summary["validation"][0]["total"], 32)
            self.assertEqual(summary["training"]["static"]["step"], 2)
            profile = summary["training_profiles"]["static"]
            self.assertEqual(profile["updates_per_second"], 4.0)
            self.assertEqual(profile["peak_process_rss_bytes"], 120)
            self.assertEqual(profile["peak_cuda_allocated_bytes"], 0)
            self.assertEqual(profile["first_total_loss"], 2.0)
            self.assertEqual(profile["last_total_loss"], 1.0)
            self.assertEqual(profile["mean_write_rate"], 0.25)
            self.assertTrue(output.read_bytes().endswith(b"\n"))


class ColabDependencyContractTests(unittest.TestCase):
    def test_requirements_keep_cuda_stack_and_compatible_numpy(self):
        lines = (ROOT / "requirements-colab.txt").read_text(encoding="utf-8").splitlines()
        requirements = [Requirement(line) for line in lines if line.strip() and not line.startswith("#")]
        by_name = {re.sub(r"[-_.]+", "-", item.name).lower(): item for item in requirements}
        self.assertFalse({"torch", "torchvision", "torchaudio"} & by_name.keys())
        self.assertFalse(any(name.startswith("nvidia-") for name in by_name))
        self.assertEqual(str(by_name["transformers"].specifier), "==4.46.2")
        self.assertEqual(str(by_name["tokenizers"].specifier), "==0.20.3")
        self.assertEqual(str(by_name["huggingface-hub"].specifier), "==0.36.2")
        numpy = by_name["numpy"].specifier
        self.assertIn(Version("2.0.2"), numpy)  # Google's published Colab inventory.
        self.assertIn(Version("2.2.6"), numpy)  # Local CPU correctness environment.
        self.assertFalse(any(item.operator in {"==", "==="} for item in numpy))


if __name__ == "__main__":
    unittest.main()
