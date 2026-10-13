"""Real child processes verify the notebook's standard-library logging helper."""

from contextlib import redirect_stdout
import io
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from scripts.colab_process import run_logged


class ColabProcessTests(unittest.TestCase):
    def test_child_progress_is_flushed_before_the_child_finishes(self):
        with tempfile.TemporaryDirectory() as temp:
            signal = Path(temp) / "progress acknowledged"

            class ProgressStream(io.StringIO):
                def flush(self):
                    super().flush()
                    if "ready before finish" in self.getvalue():
                        signal.write_text("seen", encoding="utf-8")

            code = ("from pathlib import Path\nimport sys,time\n"
                    "gate=Path(sys.argv[1])\nprint('ready before finish',flush=True)\n"
                    "for _ in range(500):\n"
                    "    if gate.exists(): break\n"
                    "    time.sleep(0.01)\n"
                    "else: sys.exit(9)\n"
                    "print('finished',flush=True)\n")
            output = ProgressStream()
            with redirect_stdout(output):
                result = run_logged([sys.executable, "-c", code, str(signal)], log_dir=Path(temp) / "logs")
            self.assertEqual(result.returncode, 0)
            self.assertIn("finished", output.getvalue())

    def test_success_streams_stdout_stderr_and_logs_paths_with_spaces(self):
        with tempfile.TemporaryDirectory(prefix="dsa process ") as temp:
            output = io.StringIO()
            work = Path(temp) / "working directory"
            logs = Path(temp) / "persistent logs"
            work.mkdir()
            env = {**os.environ, "DSA_LOGGING_FIXTURE": "forwarded value"}
            code = ("import os,sys; print('standard output',flush=True); "
                    "print('standard error',file=sys.stderr,flush=True); "
                    "print(os.environ['DSA_LOGGING_FIXTURE']); print(os.getcwd())")
            with redirect_stdout(output):
                result = run_logged([sys.executable, "-c", code], cwd=work, env=env, log_dir=logs)
            self.assertIsInstance(result, subprocess.CompletedProcess)
            self.assertEqual(result.returncode, 0)
            contents = next(logs.glob("*.log")).read_text(encoding="utf-8")
            for expected in ("standard output", "standard error", "forwarded value", str(work)):
                self.assertIn(expected, output.getvalue())
                self.assertIn(expected, contents)
            self.assertIn("Exit status: 0", contents)

    def test_nonzero_exit_reports_actual_stderr_and_omits_inline_payload(self):
        with tempfile.TemporaryDirectory() as temp, redirect_stdout(io.StringIO()):
            code = "import sys; payload_marker='do not repeat inline code'; print('CUDA unavailable',file=sys.stderr); sys.exit(7)"
            with self.assertRaises(RuntimeError) as caught:
                run_logged([sys.executable, "-c", code], log_dir=Path(temp), tail_lines=2)
            message = str(caught.exception)
            self.assertIn("exit 7", message)
            self.assertIn("CUDA unavailable", message)
            self.assertIn("Log:", message)
            self.assertNotIn("payload_marker", message)
            contents = next(Path(temp).glob("*.log")).read_text(encoding="utf-8")
            self.assertIn("CUDA unavailable", contents)
            self.assertNotIn("payload_marker", contents)

    def test_error_tail_is_bounded_but_full_log_is_retained(self):
        with tempfile.TemporaryDirectory() as temp, redirect_stdout(io.StringIO()):
            code = "import sys; [print('line-'+str(i)) for i in range(10)]; sys.exit(2)"
            with self.assertRaises(RuntimeError) as caught:
                run_logged([sys.executable, "-c", code], log_dir=temp, label="preflight", tail_lines=2)
            message = str(caught.exception)
            self.assertIn("preflight failed", message)
            self.assertIn("line-8", message)
            self.assertIn("line-9", message)
            self.assertNotIn("line-0", message)
            contents = next(Path(temp).glob("*.log")).read_text(encoding="utf-8")
            self.assertIn("line-0", contents)
            self.assertIn("line-9", contents)

    def test_missing_executable_has_short_error_and_persistent_diagnostic(self):
        with tempfile.TemporaryDirectory() as temp, redirect_stdout(io.StringIO()):
            with self.assertRaisesRegex(RuntimeError, "could not start"):
                run_logged([str(Path(temp) / "missing-python"), "-c", "payload"], log_dir=temp)
            contents = next(Path(temp).glob("*.log")).read_text(encoding="utf-8")
            self.assertIn("Could not start process", contents)
            self.assertNotIn("payload", contents)

    def test_invalid_commands_fail_before_creating_logs(self):
        with tempfile.TemporaryDirectory() as temp:
            logs = Path(temp) / "logs"
            for command in ([], "python -c pass"):
                with self.assertRaises(ValueError):
                    run_logged(command, log_dir=logs)
            with self.assertRaises(ValueError):
                run_logged([sys.executable], log_dir=logs, tail_lines=0)
            self.assertFalse(logs.exists())


if __name__ == "__main__":
    unittest.main()
