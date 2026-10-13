#!/usr/bin/env python3
"""Create a pip-less project venv while reusing Colab's installed CUDA stack.

Debian/Colab Python can provide pip without providing ensurepip. Host pip's
--python option manages the target environment directly, without ensurepip,
get-pip, apt, or package changes to the notebook kernel.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import uuid


CUDA_PACKAGES = ("torch", "torchvision", "torchaudio")
PROBE_CODE = r'''
import importlib.metadata, json, sys
from pathlib import Path
packages = {}
for name in ("torch", "torchvision", "torchaudio"):
    try:
        distribution = importlib.metadata.distribution(name)
    except importlib.metadata.PackageNotFoundError:
        continue
    packages[name] = {"version": distribution.version,
                      "location": str(Path(distribution.locate_file("")).resolve())}
print(json.dumps({"version": list(sys.version_info[:3]),
                  "prefix": str(Path(sys.prefix).resolve()),
                  "base_prefix": str(Path(sys.base_prefix).resolve()),
                  "executable": sys.executable, "cuda_packages": packages}))
'''


class SetupError(RuntimeError):
    """An actionable environment failure that preserves existing files."""


def run_command(command, *, stage, env=None):
    """Capture diagnostics so notebook subprocess errors explain the cause."""
    try:
        result = subprocess.run([str(value) for value in command], text=True,
                                capture_output=True, env=env, check=False)
    except OSError as exc:
        raise SetupError(f"{stage} could not start: {exc}") from exc
    if result.returncode:
        details = "\n".join(value.strip() for value in (result.stdout, result.stderr) if value.strip())
        raise SetupError(f"{stage} failed (exit {result.returncode}).\n{details}")
    return result


def _probe(python, *, runner=run_command, env=None):
    result = runner([python, "-c", PROBE_CODE], stage="Python environment inspection", env=env)
    try:
        value = json.loads(result.stdout)
        if not {"version", "prefix", "base_prefix", "cuda_packages"} <= value.keys():
            raise ValueError("incomplete inspection result")
        return value
    except (ValueError, TypeError) as exc:
        raise SetupError(f"Python environment inspection returned invalid JSON: {result.stdout}") from exc


def _clean_environment():
    # pip's prefix/target/user settings can otherwise redirect installation away
    # from --python. Never mutate the caller's environment or package database.
    result = os.environ.copy()
    for name in ("PYTHONHOME", "PYTHONPATH", "PIP_TARGET", "PIP_PREFIX", "PIP_USER"):
        result.pop(name, None)
    result["PIP_CONFIG_FILE"] = os.devnull
    result["PIP_DISABLE_PIP_VERSION_CHECK"] = "1"
    return result


def _venv_config(path):
    return {key.strip().lower(): value.strip() for key, value in
            (line.split("=", 1) for line in path.read_text(encoding="utf-8").splitlines()
             if "=" in line)}


def _backup(directory):
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    destination = directory.with_name(f"{directory.name}.backup-{stamp}-{uuid.uuid4().hex[:8]}")
    directory.rename(destination)
    return destination


def ensure_environment(venv_dir, *, python=None, runner=run_command, env=None):
    """Repair a compatible partial venv; preserve an incompatible one as backup."""
    python = str(python or sys.executable)
    env = _clean_environment() if env is None else env
    host = _probe(python, runner=runner, env=env)
    if not (3, 10) <= tuple(host["version"][:2]) <= (3, 13):
        raise SetupError("Use a Python 3.10–3.13 Colab runtime for this implementation.")
    directory = Path(venv_dir).expanduser().absolute()
    prefixes = {Path(host["prefix"]).resolve(), Path(host["base_prefix"]).resolve()}
    if any(directory.resolve() == prefix or directory.resolve() in prefix.parents for prefix in prefixes):
        raise SetupError("The project venv must differ from the notebook's Python environment.")
    if directory.is_symlink():
        raise SetupError("The project venv path is a symlink; choose a dedicated directory.")
    target_python = directory / "bin/python"
    backup = None
    if directory.exists():
        compatible = directory.is_dir()
        config_path = directory / "pyvenv.cfg"
        if compatible and config_path.is_file():
            config = _venv_config(config_path)
            version = config.get("version", "").split(".")[:2]
            compatible = (version == [str(value) for value in host["version"][:2]]
                          and config.get("include-system-site-packages", "").lower() == "true")
        elif compatible:
            # Preserve unknown contents. An empty directory is safe to create in.
            compatible = not any(directory.iterdir())
        if compatible and target_python.exists():
            try:
                target = _probe(target_python, runner=runner, env=env)
                compatible = (Path(target["prefix"]).resolve() == directory.resolve()
                              and target["base_prefix"] == host["base_prefix"]
                              and target["version"][:2] == host["version"][:2])
            except SetupError:
                compatible = False
        if not compatible:
            backup = _backup(directory)
    directory.parent.mkdir(parents=True, exist_ok=True)
    # Intentionally omit ensurepip, even when repairing a partially created venv.
    # No --clear: compatible existing files and locally installed packages survive.
    runner([python, "-m", "venv", "--without-pip", "--system-site-packages", directory],
           stage="Project virtual environment creation", env=env)
    target = _probe(target_python, runner=runner, env=env)
    if (Path(target["prefix"]).resolve() != directory.resolve()
            or target["base_prefix"] != host["base_prefix"]
            or target["version"][:2] != host["version"][:2]
            or _venv_config(directory / "pyvenv.cfg").get("include-system-site-packages", "").lower() != "true"):
        raise SetupError("Created Python does not use the requested project venv and shared system packages.")
    if target["cuda_packages"] != host["cuda_packages"]:
        raise SetupError("The target venv does not inherit the notebook's CUDA package stack. "
                         "Run setup from the Colab kernel Python rather than a nested virtual environment.")
    return {"python": str(target_python), "venv_dir": str(directory), "host": host,
            "target": target, "backup": str(backup) if backup else None}


def _validate_requirements(path):
    if not path.is_file():
        raise SetupError(f"Requirements file does not exist: {path}")
    # Colab supplies this stack. Reject direct replacement requests; constraints
    # below also preserve its versions when resolving transitive dependencies.
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.split("#", 1)[0].strip()
        match = re.match(r"([A-Za-z0-9_.-]+)", stripped)
        if match and match.group(1).lower().replace("_", "-") in CUDA_PACKAGES:
            raise SetupError("Colab requirements must preserve the existing torch/torchvision/torchaudio stack.")


def setup_colab(venv_dir, requirements, *, python=None, runner=run_command):
    """Set up isolated project dependencies without installing into host Python."""
    python = str(python or sys.executable)
    requirements = Path(requirements).expanduser().resolve()
    _validate_requirements(requirements)
    env = _clean_environment()
    pip = runner([python, "-m", "pip", "--version"], stage="Colab host pip inspection", env=env)
    match = re.search(r"\bpip (\d+)\.(\d+)", pip.stdout)
    if match is None or tuple(map(int, match.groups())) < (22, 3):
        raise SetupError("Colab's Python must provide pip 22.3 or later for --python support. "
                         "Select a current managed Colab runtime, then rerun this cell.")
    environment = ensure_environment(venv_dir, python=python, runner=runner, env=env)
    stack = environment["host"]["cuda_packages"]
    # Normal pip resolution reuses compatible inherited distributions. Never use
    # --ignore-installed, --force-reinstall, --upgrade, --target or --user here.
    with tempfile.TemporaryDirectory(prefix="dsa-colab-constraints-") as temporary:
        constraints = Path(temporary) / "cuda-stack.txt"
        constraints.write_text("".join(f"{name}=={info['version']}\n" for name, info in sorted(stack.items())),
                               encoding="utf-8")
        command = [python, "-m", "pip", "--python", environment["python"], "install",
                   "--disable-pip-version-check", "--constraint", constraints,
                   "--requirement", requirements]
        installed = runner(command, stage="Project dependency installation", env=env)
        if installed.stdout:
            print(installed.stdout, end="" if installed.stdout.endswith("\n") else "\n", flush=True)
        if installed.stderr:
            print(installed.stderr, file=sys.stderr, end="" if installed.stderr.endswith("\n") else "\n", flush=True)
    target = _probe(environment["python"], runner=runner, env=env)
    if target["cuda_packages"] != stack:
        raise SetupError("Project installation changed or shadowed the inherited CUDA package stack. "
                         "Preserve this venv for inspection and use a fresh dedicated venv directory.")
    return {"schema_version": "dsa_colab_setup_v1", "python": environment["python"],
            "venv_dir": environment["venv_dir"], "requirements": str(requirements),
            "python_version": target["version"], "cuda_packages": stack,
            "include_system_site_packages": True, "ensurepip_used": False,
            "host_packages_modified": False, "backup": environment["backup"]}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--venv-dir", required=True, type=Path)
    parser.add_argument("--requirements", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        report = setup_colab(args.venv_dir, args.requirements)
    except (SetupError, OSError, ValueError) as exc:
        parser.exit(1, f"DSA Colab setup stopped: {exc}\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
