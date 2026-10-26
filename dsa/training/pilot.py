"""Single-device fp32 pilot with persistent, validated resumable checkpoints.

This correctness-first trainer processes one variable-length record at a time
by default; ``batch_size`` pads several records into each forward pass. The
explicit teacher and question-only latent route share the same backbone.
It never evaluates held-out examples or selects a checkpoint using their golds.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import random
import resource
import shutil
import subprocess
import time
import uuid

import numpy as np
import torch

from dsa.data.tokenize import TinyTokenizer, answer_alignment, register_marker_token
from dsa.devices import mps_memory_snapshot, resolve_device, synchronize
from dsa.models.anchored_loop import AnchorConfig, AnchoredLoop
from dsa.models.backbone import BackboneAdapter
from dsa.models.lora import apply_lora
from dsa.training.objectives import LossConfig, joint_objective


DEFAULT_BACKBONE = "yingfanbot/gsm-cot-gpt2"
DEFAULT_REVISION = "e0ef1887404adbf6c531d2cac7804694321f87bd"


@dataclass(frozen=True)
class PilotConfig:
    prepared_dir: str | Path
    output_dir: str | Path
    backbone: str = DEFAULT_BACKBONE
    revision: str = DEFAULT_REVISION
    device: str = "auto"
    write_mode: str = "learned"
    recurrence_mode: str = "standard"
    loops: int = 16
    workspace: int = 4
    steps: int = 200
    learning_rate: float = 3e-5
    # Anchor, workspace, controller and projection (plus the <SA> row under
    # LoRA). None uses learning_rate.
    new_module_learning_rate: float | None = None
    lr_schedule: str = "constant"
    warmup_steps: int = 0
    weight_decay: float = 0.01
    seed: int = 7
    save_every: int = 25
    resume: str = "auto"
    gradient_accumulation: int = 1
    batch_size: int = 1
    max_grad_norm: float = 1.0
    periodic_interval: int = 4
    gate_threshold: float = 0.5
    cot_weight: float = 1.0
    gate_weight: float = 0.1
    anchor_weight: float = 0.1
    gate_positive_weight: float = 1.0
    answer_distill_weight: float = 0.0
    # 0 trains the whole backbone. CODI's GPT-2 recipe: rank 128, alpha 32, dropout 0.1.
    lora_rank: int = 0
    lora_alpha: float = 32.0
    lora_dropout: float = 0.1
    keep_checkpoints: int = 2
    tiny: bool = False
    smoke_report: str | Path | None = None
    gradient_checkpointing: bool = True


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def tokenizer_directory_sha256(directory: str | Path) -> str:
    directory = Path(directory)
    values = [[str(path.relative_to(directory)), sha256_file(path)]
              for path in sorted(directory.rglob("*")) if path.is_file()]
    if not values:
        raise ValueError("Prepared tokenizer directory is empty")
    return hashlib.sha256(json.dumps(values, separators=(",", ":")).encode()).hexdigest()


def _json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _atomic_json(path: Path, value) -> None:
    temporary = path.with_name("." + path.name + ".tmp")
    _json(temporary, value)
    os.replace(temporary, path)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _resolve_under(directory: Path, relative: str) -> Path:
    path = (directory / relative).resolve()
    if not path.is_relative_to(directory.resolve()):
        raise ValueError("Prepared artifact path escapes the prepared directory")
    return path


def _device(name: str) -> torch.device:
    return resolve_device(name)


def _seed(seed: int, device: torch.device) -> None:
    # Required by deterministic CUDA BLAS kernels before the first matmul.
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if device.type == "mps":
        torch.mps.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.benchmark = False
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False


class _IndexedRecords:
    """Index line offsets without retaining the full tokenized corpus in RAM."""
    def __init__(self, path: Path):
        self.path = path
        self.offsets = []
        with path.open("rb") as handle:
            while True:
                offset = handle.tell()
                line = handle.readline()
                if not line:
                    break
                if line.strip():
                    self.offsets.append(offset)
        if not self.offsets:
            raise ValueError("Prepared training split is empty")

    def __len__(self):
        return len(self.offsets)

    def __getitem__(self, index: int):
        with self.path.open("rb") as handle:
            handle.seek(self.offsets[index])
            return json.loads(handle.readline())


def _prepared(config: PilotConfig):
    prepared = Path(config.prepared_dir).resolve()
    manifest_path = prepared / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    required = {"schema_version", "backbone", "revision", "tiny", "loop_budget",
                "tokenizer_sha256", "splits"}
    if not required <= manifest.keys() or manifest["schema_version"] != 1:
        raise ValueError("Unsupported or incomplete prepared-data manifest")
    if manifest["tiny"] != config.tiny or manifest["loop_budget"] != config.loops:
        raise ValueError("Prepared tiny/loop configuration differs from training")
    if not config.tiny and (manifest["backbone"] != config.backbone
                           or manifest["revision"] != config.revision):
        raise ValueError("Prepared model identity differs from training")
    tokenizer_dir = prepared / "tokenizer"
    if tokenizer_directory_sha256(tokenizer_dir) != manifest["tokenizer_sha256"]:
        raise ValueError("Prepared tokenizer checksum mismatch")
    train_path = None
    # Verify held-out file identity, but never inspect their answers or use them
    # to control training. The preparation manifest declares the source policy.
    for split in ("train", "valid", "test"):
        entry = manifest["splits"][split]
        path = _resolve_under(prepared, entry["path"])
        if sha256_file(path) != entry["sha256"]:
            raise ValueError(f"Prepared {split} checksum mismatch")
        if split == "train":
            train_path = path
    if config.tiny:
        tokenizer = TinyTokenizer.from_pretrained(tokenizer_dir)
    else:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir, local_files_only=True)
        if config.smoke_report is None:
            raise ValueError("Real pretrained runs require --smoke-report from the unmodified checkpoint smoke")
        smoke = json.loads(Path(config.smoke_report).read_text(encoding="utf-8"))
        if (smoke.get("schema_version") != "dsa_unmodified_checkpoint_smoke_v1"
                or smoke.get("passed") is not True or smoke.get("backbone") != config.backbone
                or smoke.get("revision") != config.revision or smoke.get("tiny") is not False
                or smoke.get("model_weights_downloaded") is not True
                or smoke.get("finite_logits") is not True
                or smoke.get("vocabulary_changed") is not False
                or smoke.get("anchor_wrapper_used") is not False
                or smoke.get("device") != str(_device(config.device))
                or smoke.get("identity", {}).get("torch") != torch.__version__
                or smoke.get("identity", {}).get("transformers") != importlib.metadata.version("transformers")):
            raise ValueError("Unmodified-checkpoint smoke report does not match model identity")
    return manifest, sha256_file(manifest_path), tokenizer, _IndexedRecords(train_path)


def _runtime(device: torch.device) -> dict:
    root = Path(__file__).resolve().parents[2]
    def git(*args):
        try:
            return subprocess.check_output(["git", *args], cwd=root, text=True,
                                           stderr=subprocess.DEVNULL).strip()
        except (OSError, subprocess.CalledProcessError):
            return None
    return {"python": platform.python_version(), "platform": platform.platform(),
            "device": str(device), "device_name": (torch.cuda.get_device_name(device)
            if device.type == "cuda" else platform.processor()),
            "cuda": torch.version.cuda, "cudnn": torch.backends.cudnn.version(),
            "packages": {name: importlib.metadata.version(name)
                         for name in ("torch", "transformers", "tokenizers", "numpy")},
            "git_commit": git("rev-parse", "HEAD"),
            "working_tree_dirty": bool(git("status", "--porcelain")),
            "implementation_sha256": {str(path.relative_to(root)): sha256_file(path)
                                      for path in sorted((root / "dsa").rglob("*.py"))},
            "deterministic_algorithms": True, "precision": "float32"}


def _contract(config, manifest, manifest_sha256, order, runtime):
    settings = asdict(config)
    # A cosine schedule depends on the total, so only it binds the step count.
    for key in ("prepared_dir", "output_dir", "save_every", "resume", "keep_checkpoints",
                "smoke_report", "device") + (("steps",) if config.lr_schedule == "constant" else ()):
        settings.pop(key)
    return {"settings": settings, "prepared_manifest_sha256": manifest_sha256,
            "prepared_identity": {key: manifest[key] for key in
                                  ("backbone", "revision", "tiny", "tokenizer_sha256", "splits")},
            "data_order": "seeded Python Random.shuffle once; cycle unchanged each epoch",
            "record_count": len(order),
            "record_order_sha256": hashlib.sha256(
                json.dumps(order, separators=(",", ":")).encode()).hexdigest(),
            "runtime": {key: runtime[key] for key in
                        ("device", "device_name", "packages", "cuda", "cudnn", "implementation_sha256")},
            "optimizer": "AdamW", "scheduler": _schedule_description(config)}


def _schedule_description(config):
    if config.lr_schedule == "constant" and not config.warmup_steps:
        return "constant learning rate"
    return f"{config.warmup_steps} linear warmup steps, then {config.lr_schedule}"


def _scheduler(optimizer, config):
    from transformers import get_constant_schedule_with_warmup, get_cosine_schedule_with_warmup
    if config.lr_schedule == "cosine":
        return get_cosine_schedule_with_warmup(optimizer, config.warmup_steps, config.steps)
    if config.warmup_steps:
        return get_constant_schedule_with_warmup(optimizer, config.warmup_steps)
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)


def _parameter_groups(model, config):
    """The single default group, or backbone / new-module groups."""
    if not config.lora_rank and config.new_module_learning_rate is None:
        return model.parameters()
    new_rate = config.learning_rate if config.new_module_learning_rate is None else config.new_module_learning_rate
    embedding = model.backbone.embeddings.weight
    frozen_rows = bool(config.lora_rank)  # LoRA trains only the embedding's <SA> row
    groups = [{"params": [parameter for parameter in model.backbone.parameters() if parameter.requires_grad
                          and not (frozen_rows and parameter is embedding)]},
              {"params": [parameter for name, parameter in model.named_parameters()
                          if not name.startswith("backbone.")], "lr": new_rate}]
    if frozen_rows:
        # No decay, so rows that never receive gradient stay exactly as loaded.
        groups.append({"params": [embedding], "lr": new_rate, "weight_decay": 0.0})
    return groups


def _anchor_config(config, tokenizer):
    marker = tokenizer.encode("<SA>", add_special_tokens=False)
    if len(marker) != 1 or marker[0] in {tokenizer.pad_token_id, tokenizer.eos_token_id}:
        raise ValueError("Prepared marker must be a distinct single registered token")
    return AnchorConfig(
        workspace_size=config.workspace, loop_budget=config.loops,
        gate_threshold=config.gate_threshold, write_mode=config.write_mode,
        periodic_interval=config.periodic_interval, pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id, marker_token_id=marker[0], recurrence_mode=config.recurrence_mode)


def _initial_model(config, tokenizer, device):
    if config.tiny:
        backbone = BackboneAdapter.tiny(len(tokenizer))
    else:
        from transformers import AutoModelForCausalLM
        causal = AutoModelForCausalLM.from_pretrained(
            config.backbone, revision=config.revision, torch_dtype=torch.float32,
            attn_implementation="eager")
        register_marker_token(tokenizer, causal)
        backbone = BackboneAdapter(causal)
    model = AnchoredLoop(backbone, _anchor_config(config, tokenizer))
    if config.lora_rank:
        apply_lora(model, config.lora_rank, config.lora_alpha, config.lora_dropout)
    return _configure_model(model, config, device)


def _configure_model(model, config, device):
    model.to(device=device, dtype=torch.float32)
    if config.gradient_checkpointing and not config.tiny:
        model.backbone.model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False})
    model.backbone.model.config.use_cache = False
    return model


def _load_model(directory: Path, config, device):
    # Transformers may select SDPA when reloading a checkpoint whose original
    # runtime used eager attention. Select the same implementation explicitly.
    from transformers import AutoModelForCausalLM
    backbone = BackboneAdapter(AutoModelForCausalLM.from_pretrained(
        directory / "backbone", attn_implementation="eager", torch_dtype=torch.float32))
    model = AnchoredLoop(backbone, AnchorConfig(**json.loads(
        (directory / "anchor_config.json").read_text(encoding="utf-8"))))
    memory = torch.load(directory / "anchor.pt", map_location="cpu", weights_only=True)
    missing, unexpected = model.load_state_dict(memory, strict=False)
    if unexpected or any(not key.startswith("backbone.") for key in missing):
        raise ValueError("Incomplete or incompatible anchor checkpoint")
    return _configure_model(model, config, device)


def _restore_trainable(model, state):
    expected = {name for name, parameter in model.named_parameters() if parameter.requires_grad}
    if set(state) != expected:
        raise ValueError("Saved trainable tensors do not match the LoRA model")
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if name in state:
                parameter.copy_(state[name])


def _state_digest(model) -> str:
    digest = hashlib.sha256()
    for key, tensor in sorted(model.state_dict().items()):
        digest.update(key.encode())
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _rng_state(device):
    return {"python": random.getstate(), "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            "mps": torch.mps.get_rng_state() if device.type == "mps" else None}


def _restore_rng(state, device):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if state["cuda"] is not None:
        if not torch.cuda.is_available():
            raise ValueError("Saved CUDA RNG state requires CUDA")
        torch.cuda.set_rng_state_all(state["cuda"])
    if state["mps"] is not None:
        if device.type != "mps":
            raise ValueError("Saved MPS RNG state requires MPS")
        torch.mps.set_rng_state(state["mps"])


def _completed_checkpoints(output: Path):
    return sorted(path for path in output.glob("checkpoint-[0-9]*")
                  if path.is_dir() and (path / "COMPLETE.json").exists()
                  and (path / "training_state.pt").exists())


def _find_resume(output: Path, resume: str) -> Path | None:
    if resume == "none":
        if (output / "run_manifest.json").exists() or _completed_checkpoints(output):
            raise ValueError("Output already contains a run; choose a new directory or --resume auto")
        return None
    if resume == "auto":
        completed = _completed_checkpoints(output)
        if completed:
            return completed[-1]
        if (output / "run_manifest.json").exists():
            raise ValueError("Interrupted run has no complete checkpoint; use a new output directory")
        return None
    checkpoint = Path(resume).expanduser().resolve()
    if checkpoint.parent != output.resolve() or checkpoint not in _completed_checkpoints(output):
        raise ValueError("Resume checkpoint must be a completed checkpoint inside the output directory")
    return checkpoint


def _reconcile_metrics(path: Path, step: int):
    """Discard only post-checkpoint lines from an interrupted optimizer update."""
    if not path.exists():
        if step:
            raise ValueError("Resumed run is missing its metrics journal")
        return
    rows = []
    lines = path.read_text(encoding="utf-8").splitlines()
    for index, line in enumerate(lines):
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            if index == len(lines) - 1:
                break  # a torn final line can arise on runtime disconnect
            raise ValueError("Metrics journal is corrupt before its final line")
        if row["step"] <= step:
            rows.append(row)
    if [row["step"] for row in rows] != list(range(1, step + 1)):
        raise ValueError("Metrics journal does not match the resumed checkpoint")
    temporary = path.with_suffix(".jsonl.tmp")
    temporary.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")
    os.replace(temporary, path)


def _save(output, model, tokenizer, optimizer, scheduler, step, contract,
          initial_digest, device, keep):
    final = output / f"checkpoint-{step:08d}"
    if (final / "COMPLETE.json").exists():
        return final
    staging = output / f".incomplete-checkpoint-{step:08d}-{uuid.uuid4().hex}"
    staging.mkdir()
    model.save_pretrained(staging / "model", tokenizer)
    settings = contract["settings"]
    payload = {"schema_version": 1, "step": step, "contract": contract,
               "initial_state_sha256": initial_digest,
               "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
               "rng": _rng_state(device),
               "records_consumed": step * settings["gradient_accumulation"] * settings["batch_size"]}
    if settings["lora_rank"]:
        # The saved model holds merged weights; adapters restart from these.
        payload["trainable_state"] = {name: parameter.detach().cpu() for name, parameter
                                      in model.named_parameters() if parameter.requires_grad}
    torch.save(payload, staging / "training_state.pt")
    _json(staging / "COMPLETE.json", {"step": step, "completed_at": _utc_now(),
                                     "schema_version": 1})
    os.replace(staging, final)
    temporary = output / ".latest.json.tmp"
    _json(temporary, {"path": final.name, "step": step})
    os.replace(temporary, output / "latest.json")
    # The default bounds Drive usage at two full model/Adam states. Only this
    # run's completed checkpoint directories are eligible; incomplete files
    # remain available for diagnosis and are never loaded.
    for old in _completed_checkpoints(output)[:-keep]:
        shutil.rmtree(old)
    return final


def _validate(config):
    for name in ("loops", "workspace", "steps", "save_every", "gradient_accumulation", "batch_size",
                 "keep_checkpoints"):
        if getattr(config, name) < 1:
            raise ValueError(f"{name} must be positive")
    if config.keep_checkpoints < 2:
        raise ValueError("keep_checkpoints must retain at least the latest and previous checkpoints")
    for name in ("learning_rate", "max_grad_norm", "weight_decay", "cot_weight",
                 "gate_weight", "anchor_weight", "gate_positive_weight", "gate_threshold",
                 "answer_distill_weight", "lora_alpha", "lora_dropout"):
        if not math.isfinite(getattr(config, name)):
            raise ValueError(f"{name} must be finite")
    if config.new_module_learning_rate is not None and not (
            math.isfinite(config.new_module_learning_rate) and config.new_module_learning_rate > 0):
        raise ValueError("new_module_learning_rate must be positive and finite")
    if config.lr_schedule not in {"constant", "cosine"}:
        raise ValueError("lr_schedule must be constant or cosine")
    if not 0 <= config.warmup_steps <= config.steps:
        raise ValueError("warmup_steps must be between 0 and steps")
    if config.lora_rank < 0 or (config.lora_rank and (config.lora_alpha <= 0 or not 0 <= config.lora_dropout < 1)):
        raise ValueError("LoRA needs a nonnegative rank, positive alpha and dropout in [0, 1)")
    if config.learning_rate <= 0 or config.max_grad_norm <= 0 or config.weight_decay < 0:
        raise ValueError("Learning rate/gradient limit must be positive and weight decay nonnegative")
    if config.seed < 0 or any(value < 0 for value in
            (config.cot_weight, config.gate_weight, config.anchor_weight, config.gate_positive_weight,
             config.answer_distill_weight)):
        raise ValueError("Seed and loss weights must be nonnegative")
    if not config.tiny and (len(config.revision) != 40 or any(
            char not in "0123456789abcdef" for char in config.revision)):
        raise ValueError("Pretrained model revision must be an immutable 40-character commit SHA")


def train_pilot(config: PilotConfig) -> dict:
    """Train to ``steps`` total updates; auto-resume may extend that total."""
    _validate(config)
    device = _device(config.device)
    _seed(config.seed, device)
    manifest, manifest_sha256, tokenizer, records = _prepared(config)
    output = Path(config.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    resume = _find_resume(output, config.resume)
    runtime = _runtime(device)
    order = list(range(len(records)))
    random.Random(config.seed).shuffle(order)
    contract = _contract(config, manifest, manifest_sha256, order, runtime)
    payload = None
    if resume is not None:
        # This local training artifact is trusted. It contains Python and NumPy
        # RNG tuples in addition to tensors, so weights_only cannot load it.
        payload = torch.load(resume / "training_state.pt", map_location="cpu", weights_only=False)
        if payload["schema_version"] != 1 or payload["contract"] != contract:
            raise ValueError("Resume experiment/data/tokenizer/runtime settings differ from saved run")
        if payload["records_consumed"] != payload["step"] * config.gradient_accumulation * config.batch_size:
            raise ValueError("Resume data cursor does not agree with optimizer step")
        if config.lora_rank:
            # Merged weights cannot restart the adapters exactly. Rebuild the seeded
            # initial model, prove it matches the saved run, then restore what trained.
            model = _initial_model(config, tokenizer, device)
            if _state_digest(model) != payload["initial_state_sha256"]:
                raise ValueError("Rebuilt LoRA initial state differs from the saved run")
            _restore_trainable(model, payload["trainable_state"])
        else:
            model = _load_model(resume / "model", config, device)
        if model.config != _anchor_config(config, tokenizer):
            raise ValueError("Checkpoint anchor configuration differs from the saved experiment")
        initial_digest = payload["initial_state_sha256"]
    else:
        model = _initial_model(config, tokenizer, device)
        initial_digest = _state_digest(model)
    optimizer = torch.optim.AdamW(_parameter_groups(model, config), lr=config.learning_rate,
                                 weight_decay=config.weight_decay, foreach=False)
    scheduler = _scheduler(optimizer, config)
    step = 0
    if payload is not None:
        optimizer.load_state_dict(payload["optimizer"])
        scheduler.load_state_dict(payload["scheduler"])
        step = payload["step"]
        _restore_rng(payload["rng"], device)
    if config.steps < step:
        raise ValueError("Requested total steps is smaller than completed training")
    metrics_path = output / "metrics.jsonl"
    _reconcile_metrics(metrics_path, step)
    run_path = output / "run_manifest.json"
    previous = json.loads(run_path.read_text()) if run_path.exists() else {}
    run_manifest = {"schema_version": 1, "created_at": previous.get("created_at", _utc_now()),
                    "updated_at": _utc_now(), "status": "running", "contract": contract,
                    "runtime": runtime, "prepared_manifest": manifest,
                    "initial_state_sha256": initial_digest,
                    "target_steps": config.steps, "completed_steps": step,
                    "resumed_from": str(resume) if resume else None,
                    "purpose": "small correctness/feasibility pilot; not a benchmark result",
                    "exact_resume_scope": "same device, software and implementation; CPU verified"}
    _atomic_json(run_path, run_manifest)
    prefix = tokenizer.encode("\n### ", add_special_tokens=False)
    loss_config = LossConfig(config.cot_weight, config.gate_weight, config.anchor_weight,
                             config.gate_positive_weight, config.answer_distill_weight)
    model.train()
    # Persist a resumable initial state before the first expensive optimizer
    # update, including for runtimes disconnected before save_every is reached.
    checkpoint = resume if resume is not None else _save(
        output, model, tokenizer, optimizer, scheduler, 0, contract,
        initial_digest, device, config.keep_checkpoints)
    with metrics_path.open("a", encoding="utf-8") as journal:
        for next_step in range(step + 1, config.steps + 1):
            synchronize(device)
            started = time.monotonic()
            optimizer.zero_grad(set_to_none=True)
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
            totals = {key: 0.0 for key in ("total", "answer", "cot", "gate", "anchor", "distill",
                                           "writes", "gate_probability")}
            sample_ids = []
            for microbatch in range(config.gradient_accumulation):
                start = ((next_step - 1) * config.gradient_accumulation + microbatch) * config.batch_size
                batch = [records[order[cursor % len(order)]] for cursor in range(start, start + config.batch_size)]
                if any(record.get("loop_budget") != config.loops for record in batch):
                    raise ValueError("Tokenized record loop budget differs from training")
                # Padding sets the sequence widths; positions still count real tokens.
                positions = getattr(model.backbone.model.config, "max_position_embeddings", None)
                if positions is not None and max(
                        max(len(record["explicit_input_ids"]) for record in batch),
                        max(len(record["question_input_ids"]) for record in batch) + 1 + config.workspace
                        + max(len(record["latent_suffix_input_ids"]) for record in batch)) > positions:
                    raise ValueError("Prepared record exceeds model context; training never truncates")
                if config.answer_distill_weight:
                    for record in batch:
                        record["answer_alignment"] = answer_alignment(record, tokenizer)
                losses = joint_objective(model, batch, prefix, loss_config)
                if not torch.isfinite(losses["total"]):
                    raise FloatingPointError(f"Nonfinite loss at optimizer step {next_step}")
                (losses["total"] / config.gradient_accumulation).backward()
                for key in ("total", "answer", "cot", "gate", "anchor", "distill"):
                    totals[key] += losses[key].detach().item() / config.gradient_accumulation
                totals["writes"] += losses["loop"].writes.detach().sum(1).mean().item() / config.gradient_accumulation
                totals["gate_probability"] += losses["loop"].gate_logits.detach().sigmoid().mean().item() / config.gradient_accumulation
                sample_ids.extend(record.get("sample_id") for record in batch)
            gate_grad = model.gate_head.weight.grad
            gate_norm = 0.0 if gate_grad is None else gate_grad.norm().item()
            gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm,
                                                         error_if_nonfinite=True).item()
            optimizer.step()
            scheduler.step()
            synchronize(device)
            rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            row = {"step": next_step, **totals, "write_rate": totals["writes"] / config.loops,
                   "sample_ids": sample_ids, "gradient_norm": gradient_norm,
                   "gate_gradient_norm": gate_norm, "learning_rate": scheduler.get_last_lr()[0],
                   "seconds": time.monotonic() - started,
                   "peak_process_rss_bytes": rss if platform.system() == "Darwin" else rss * 1024,
                   "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(device)
                   if device.type == "cuda" else None,
                   **mps_memory_snapshot(device)}
            journal.write(json.dumps(row, sort_keys=True) + "\n")
            journal.flush()
            if next_step % config.save_every == 0 or next_step == config.steps:
                checkpoint = _save(output, model, tokenizer, optimizer, scheduler, next_step,
                                   contract, initial_digest, device, config.keep_checkpoints)
                print(json.dumps({"step": next_step, "total": row["total"],
                                  "writes": row["writes"], "checkpoint": str(checkpoint)}), flush=True)
    run_manifest.update(status="complete", completed_steps=config.steps,
                        updated_at=_utc_now(), checkpoint=str(checkpoint))
    _atomic_json(run_path, run_manifest)
    report = {"step": config.steps, "checkpoint": str(checkpoint), "metrics_path": str(metrics_path),
              "manifest_path": str(run_path), "initial_state_sha256": initial_digest,
              "resumed_from": str(resume) if resume else None, "device": str(device)}
    _atomic_json(output / "train_report.json", report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--backbone", default=DEFAULT_BACKBONE)
    parser.add_argument("--revision", default=DEFAULT_REVISION)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    parser.add_argument("--write-mode", choices=("static", "every_loop", "periodic", "learned"), default="learned")
    parser.add_argument("--recurrence-mode", choices=("standard", "flt_memory"), default="standard")
    for name, default in (("loops", 16), ("workspace", 4), ("steps", 200), ("seed", 7),
                          ("save-every", 25), ("gradient-accumulation", 1), ("batch-size", 1),
                          ("periodic-interval", 4), ("keep-checkpoints", 2), ("warmup-steps", 0)):
        parser.add_argument("--" + name, type=int, default=default)
    for name, default in (("learning-rate", 3e-5), ("weight-decay", 0.01), ("max-grad-norm", 1.0),
                          ("gate-threshold", 0.5), ("cot-weight", 1.0), ("gate-weight", 0.1),
                          ("anchor-weight", 0.1), ("gate-positive-weight", 1.0),
                          ("lora-alpha", 32.0), ("lora-dropout", 0.1)):
        parser.add_argument("--" + name, type=float, default=default)
    parser.add_argument("--new-module-learning-rate", type=float,
                        help="Anchor, workspace, controller and projection (and the <SA> row under LoRA); "
                             "default: --learning-rate")
    parser.add_argument("--lr-schedule", choices=("constant", "cosine"), default="constant")
    parser.add_argument("--answer-distill-weight", type=float, default=0.0,
                        help="CODI's answer-position hidden-state distillation (CODI uses 1); 0 disables it")
    parser.add_argument("--lora-rank", type=int, default=0,
                        help="LoRA rank on c_attn/c_proj/c_fc (CODI uses 128); 0 trains the whole backbone")
    parser.add_argument("--resume", default="auto", help="auto, none, or a complete checkpoint directory")
    parser.add_argument("--smoke-report", type=Path)
    parser.add_argument("--tiny", action="store_true", help="Download-free correctness smoke only")
    parser.add_argument("--no-gradient-checkpointing", action="store_false", dest="gradient_checkpointing")
    args = parser.parse_args(argv)
    print(json.dumps(train_pilot(PilotConfig(**vars(args))), indent=2))


if __name__ == "__main__":
    main()
