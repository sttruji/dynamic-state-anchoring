"""Colab bootstrap recovery without ensurepip, host installs, or network calls."""
import contextlib
import io
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

from scripts.setup_colab import (SetupError, ensure_environment, run_command,
                                 setup_colab)


class FakeRuntime:
    def __init__(self, directory):
        self.directory = directory
        self.commands = []
        self.fail_install = False
        self.wrong_target_prefix = False
        self.replace_torch = False
        self.stack = {"torch": {"version": "2.9.0+cu128", "location": "/system/site-packages"}}

    def __call__(self, command, *, stage, env=None):
        values = [str(value) for value in command]
        self.commands.append((values, stage, env))
        if values[-2:] == ["pip", "--version"]:
            return subprocess.CompletedProcess(values, 0, stdout="pip 25.1.1 from /system/pip (python 3.13)\n", stderr="")
        if "venv" in values:
            self.directory.mkdir(parents=True, exist_ok=True)
            (self.directory / "bin").mkdir(exist_ok=True)
            (self.directory / "bin/python").touch()
            (self.directory / "pyvenv.cfg").write_text(
                "version = 3.13.3\ninclude-system-site-packages = true\n")
            return subprocess.CompletedProcess(values, 0, stdout="", stderr="")
        if "install" in values:
            if self.fail_install:
                raise SetupError("Project dependency installation failed (exit 1).\nNo matching wheel")
            return subprocess.CompletedProcess(values, 0, stdout="Requirements satisfied\n", stderr="")
        is_target = values[0] == str(self.directory / "bin/python")
        stack = dict(self.stack)
        if is_target and self.replace_torch and any("install" in command[0] for command in self.commands):
            stack = {"torch": {"version": "2.9.0+cu128", "location": str(self.directory / "lib/site-packages")}}
        prefix = str(self.directory) if is_target and not self.wrong_target_prefix else "/system"
        report = {"version": [3, 13, 3], "prefix": prefix,
                  "base_prefix": "/system", "cuda_packages": stack}
        return subprocess.CompletedProcess(values, 0, stdout=json.dumps(report), stderr="")


class ColabSetupTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).resolve()
        self.directory = self.root / "dsa-venv"
        self.requirements = self.root / "requirements.txt"
        self.requirements.write_text("transformers==4.46.2\n")
        self.runtime = FakeRuntime(self.directory)

    def setup(self):
        with contextlib.redirect_stdout(io.StringIO()):
            return setup_colab(self.directory, self.requirements, python="/system/python",
                               runner=self.runtime)

    def test_creation_skips_ensurepip_and_targets_project_only(self):
        report = self.setup()
        self.assertFalse(report["ensurepip_used"])
        self.assertFalse(report["host_packages_modified"])
        self.assertEqual(report["python_version"][:2], [3, 13])
        commands = [entry[0] for entry in self.runtime.commands]
        venv = next(command for command in commands if "venv" in command)
        self.assertIn("--without-pip", venv)
        self.assertIn("--system-site-packages", venv)
        install = next(command for command in commands if "install" in command)
        self.assertEqual(install[:6], ["/system/python", "-m", "pip", "--python",
                                      str(self.directory / "bin/python"), "install"])
        self.assertFalse(any(flag in install for flag in
                             ("--ignore-installed", "--force-reinstall", "--upgrade", "--user", "--target")))
        self.assertEqual(len([command for command in commands if "install" in command]), 1)

    def test_partial_environment_is_repaired_in_place(self):
        self.directory.mkdir()
        (self.directory / "pyvenv.cfg").write_text("version = 3.13.3\ninclude-system-site-packages = true\n")
        sentinel = self.directory / "preserve-user-file.txt"
        sentinel.write_text("keep me")
        report = self.setup()
        self.assertIsNone(report["backup"])
        self.assertEqual(sentinel.read_text(), "keep me")
        self.assertTrue((self.directory / "bin/python").exists())

    def test_failed_install_can_retry_existing_environment(self):
        self.runtime.fail_install = True
        with self.assertRaisesRegex(SetupError, "No matching wheel"):
            self.setup()
        self.assertTrue((self.directory / "bin/python").exists())
        self.runtime.fail_install = False
        report = self.setup()
        self.assertIsNone(report["backup"])
        self.assertTrue(report["include_system_site_packages"])

    def test_stale_python_environment_is_backed_up(self):
        self.directory.mkdir()
        (self.directory / "pyvenv.cfg").write_text("version = 3.12.13\ninclude-system-site-packages = true\n")
        (self.directory / "preserved.txt").write_text("old runtime")
        report = self.setup()
        backup = Path(report["backup"])
        self.assertEqual((backup / "preserved.txt").read_text(), "old runtime")
        self.assertFalse((self.directory / "preserved.txt").exists())

    def test_unknown_contents_are_preserved_as_backup(self):
        self.directory.mkdir()
        (self.directory / "personal-data.txt").write_text("preserve")
        report = self.setup()
        self.assertEqual((Path(report["backup"]) / "personal-data.txt").read_text(), "preserve")

    def test_inherited_cuda_stack_may_not_be_shadowed(self):
        self.runtime.replace_torch = True
        with self.assertRaisesRegex(SetupError, "shadowed.*CUDA"):
            self.setup()

    def test_torch_requirement_is_rejected_before_installation(self):
        self.requirements.write_text("torch==2.9.0\n")
        with self.assertRaisesRegex(SetupError, "preserve.*torch"):
            self.setup()
        self.assertEqual(self.runtime.commands, [])

    def test_failed_command_preserves_stdout_and_stderr_diagnostics(self):
        with self.assertRaisesRegex(SetupError, "example diagnostic") as caught:
            run_command([sys.executable, "-c", "import sys; print('example diagnostic', file=sys.stderr); sys.exit(7)"],
                        stage="Example stage")
        self.assertIn("exit 7", str(caught.exception))

    def test_real_pipless_venv_creation_and_repair_offline(self):
        # Use the actual base interpreter, avoiding a nested project environment.
        python = getattr(sys, "_base_executable", sys.executable)
        report = ensure_environment(self.directory, python=python)
        result = subprocess.run([report["python"], "-c", "import sys; print(sys.prefix)"],
                                check=True, text=True, capture_output=True)
        self.assertEqual(Path(result.stdout.strip()).resolve(), self.directory)
        self.assertFalse(any((self.directory / "lib").glob("python*/site-packages/pip")))
        (self.directory / "user-file.txt").write_text("preserve")
        repaired = ensure_environment(self.directory, python=python)
        self.assertIsNone(repaired["backup"])
        self.assertEqual((self.directory / "user-file.txt").read_text(), "preserve")

    def test_real_host_pip_manages_pipless_target_with_empty_requirements(self):
        python = shutil.which("python3.13") or getattr(sys, "_base_executable", sys.executable)
        available = subprocess.run([python, "-m", "pip", "--version"],
                                   text=True, capture_output=True)
        if available.returncode:
            self.skipTest("The base interpreter has no host pip; mock coverage remains active")
        self.requirements.write_text("# No distributions requested: offline management smoke.\n")
        with contextlib.redirect_stdout(io.StringIO()):
            report = setup_colab(self.directory, self.requirements, python=python)
        self.assertFalse(report["host_packages_modified"])
        self.assertFalse(report["ensurepip_used"])
        self.assertTrue((self.directory / "bin/python").exists())
        self.assertFalse(any((self.directory / "lib").glob("python*/site-packages/pip")))


if __name__ == "__main__":
    unittest.main()
