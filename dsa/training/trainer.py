"""Single-device correctness trainer and exact CPU resume state."""
from __future__ import annotations

import json
from pathlib import Path
import random
import platform
import subprocess
import importlib.metadata
import hashlib
from dataclasses import asdict

import numpy as np
import torch

from .objectives import joint_objective, LossConfig


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def train_steps(model, records, answer_prefix_ids, optimizer, steps, *, start_step=0,
                loss_config=LossConfig()):
    if not records or steps < 1:
        raise ValueError("Nonempty records and positive step count required")
    model.train()
    metrics = []
    for step in range(start_step, start_step + steps):
        record = records[step % len(records)]
        optimizer.zero_grad(set_to_none=True)
        losses = joint_objective(model, record, answer_prefix_ids, loss_config)
        if not torch.isfinite(losses["total"]):
            raise FloatingPointError(f"Nonfinite training loss at step {step}")
        losses["total"].backward()
        gate_grad = model.gate_head.weight.grad
        gate_norm = 0.0 if gate_grad is None else gate_grad.norm().item()
        gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0).item()
        optimizer.step()
        metrics.append({"step": step + 1, **{key: losses[key].detach().item()
                        for key in ("total", "answer", "cot", "gate", "anchor")},
                        "gradient_norm": gradient_norm, "gate_gradient_norm": gate_norm,
                        "writes": losses["loop"].writes.detach().sum().item()})
    return metrics


def _record_digest(records):
    return None if records is None else hashlib.sha256(
        json.dumps(records, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()


def save_training_state(path, model, optimizer, step, scheduler=None, *,
                        loss_config=LossConfig(), records=None):
    payload = {"model": model.state_dict(), "optimizer": optimizer.state_dict(), "step": step,
               "torch_rng": torch.get_rng_state(), "python_rng": random.getstate(),
               "numpy_rng": np.random.get_state(),
               "scheduler": scheduler.state_dict() if scheduler is not None else None,
               "anchor_config": asdict(model.config), "loss_config": asdict(loss_config),
               "record_order_sha256": _record_digest(records)}
    torch.save(payload, path)


def load_training_state(path, model, optimizer, scheduler=None, *,
                        loss_config=LossConfig(), records=None):
    # Local trusted training-state files contain Python/NumPy RNG tuples.
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload["anchor_config"] != asdict(model.config):
        raise ValueError("Resume model settings differ from the saved experiment")
    if payload["loss_config"] != asdict(loss_config):
        raise ValueError("Resume loss settings differ from the saved experiment")
    if payload["record_order_sha256"] != _record_digest(records):
        raise ValueError("Resume data or record order differs from the saved experiment")
    if (payload["scheduler"] is None) != (scheduler is None):
        raise ValueError("Resume scheduler configuration differs from saved state")
    model.load_state_dict(payload["model"])
    optimizer.load_state_dict(payload["optimizer"])
    if scheduler is not None:
        scheduler.load_state_dict(payload["scheduler"])
    torch.set_rng_state(payload["torch_rng"])
    random.setstate(payload["python_rng"])
    np.random.set_state(payload["numpy_rng"])
    return payload["step"]


def runtime_manifest():
    root = Path(__file__).resolve().parents[2]
    sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    code_hashes = {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
                   for path in sorted(root.glob("dsa/**/*.py"))}
    for name in ("annotate_milestones.py", "prepare_dsa_data.py", "smoke_dsa.py", "tokenize_dsa_data.py"):
        path = root / "scripts" / name
        if path.exists():
            code_hashes[str(path.relative_to(root))] = hashlib.sha256(path.read_bytes()).hexdigest()
    return {"python": platform.python_version(), "device": "cpu", "upstream_lotus_sha": sha,
            "implementation_sha256": code_hashes,
            "environment_lock_sha256": hashlib.sha256((root / "requirements-dsa.lock.txt").read_bytes()).hexdigest(),
            "packages": {name: importlib.metadata.version(name)
                         for name in ("torch", "transformers", "tokenizers", "numpy")},
            "working_tree_dirty": bool(subprocess.check_output(
                ["git", "status", "--porcelain"], cwd=root, text=True).strip())}
