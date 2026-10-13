"""Stream project subprocess output and retain useful Colab error diagnostics."""

from collections import deque
from datetime import datetime, timezone
import os
from pathlib import Path
import re
import subprocess
import sys
import uuid


def _label(argv):
    executable = Path(argv[0]).name
    if len(argv) == 1:
        return executable
    if argv[1] == "-c":
        return f"{executable} -c"
    if argv[1] == "-m" and len(argv) > 2:
        return f"{executable} -m {argv[2]}"
    return f"{executable} {Path(argv[1]).name}"


def run_logged(command, *, cwd=None, env=None, log_dir, label=None, tail_lines=30):
    """Tee merged output to the notebook and a log; raise a concise failure.

    This helper imports no project dependencies. Inline ``python -c`` payloads
    and environment values are not copied into command labels or log headers.
    """
    if isinstance(command, (str, bytes)) or not command:
        raise ValueError("command must be a nonempty argument sequence")
    if tail_lines < 1:
        raise ValueError("tail_lines must be positive")
    argv = [os.fspath(value) for value in command]
    short_label = (str(label) if label is not None else _label(argv))[:120]
    directory = Path(log_dir)
    directory.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", short_label).strip("._")[:64] or "command"
    log_path = directory / f"{stamp}-{stem}-{uuid.uuid4().hex[:8]}.log"
    output_tail = deque(maxlen=tail_lines)
    child_env = dict(os.environ if env is None else env)
    child_env.setdefault("PYTHONUNBUFFERED", "1")
    print(f"Running {short_label}. Log: {log_path}", flush=True)
    with log_path.open("x", encoding="utf-8") as log:
        log.write(f"Command: {short_label}\n")
        log.flush()
        try:
            process = subprocess.Popen(argv, cwd=cwd, env=child_env, stdout=subprocess.PIPE,
                                       stderr=subprocess.STDOUT, text=True, encoding="utf-8",
                                       errors="replace", bufsize=1)
        except OSError as exc:
            log.write(f"Could not start process: {exc}\n")
            raise RuntimeError(f"{short_label} could not start. Log: {log_path}\n{exc}") from exc
        try:
            for line in process.stdout:
                sys.stdout.write(line)
                sys.stdout.flush()
                log.write(line)
                log.flush()
                output_tail.append(line.rstrip("\r\n")[-2000:])
            returncode = process.wait()
        except BaseException:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            raise
        finally:
            process.stdout.close()
        log.write(f"\nExit status: {returncode}\n")
    if returncode:
        tail = "\n".join(output_tail)
        if len(tail) > 6000:
            tail = "[earlier output omitted]\n" + tail[-6000:]
        raise RuntimeError(f"{short_label} failed (exit {returncode}). Log: {log_path}\n\n{tail}")
    return subprocess.CompletedProcess(argv, returncode)
