#!/usr/bin/env python3
"""Reuse a verified finished pilot arm, otherwise restart it locally.

Small pilots restart partial/incompatible runs rather than spending time repairing
them. Checkpoints stay on VM disk during training. Finished states are archived
as one checksummed file on Drive and retained locally for independent download.
"""
import argparse
import hashlib
import json
from pathlib import Path
import random
import shutil
import sys
import tempfile
import uuid
from zipfile import ZipFile, ZIP_STORED

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
import dsa.training.pilot as pilot
from scripts.evaluate_dsa import _load_model, _validate_tokenizers, _validate_training_identity
from scripts.smoke_pretrained_dsa import atomic_json


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def validate_finished(run_dir, config, manifest, manifest_sha, expected, initial_sha=None):
    run_dir = Path(run_dir)
    checkpoint = run_dir / f"checkpoint-{config.steps:08d}"
    marker = load(checkpoint / "COMPLETE.json")
    if marker.get("schema_version") != 1 or marker.get("step") != config.steps:
        raise ValueError("No completed checkpoint at the requested target")
    run = load(run_dir / "run_manifest.json")
    if run["contract"] != expected:
        raise ValueError("Saved training settings/data/runtime differ from this pilot")
    if initial_sha and run["initial_state_sha256"] != initial_sha:
        raise ValueError("Saved initialization differs from the matched arm")
    anchor, _ = _validate_training_identity(run_dir, checkpoint, manifest, manifest_sha)
    _validate_tokenizers(Path(config.prepared_dir), checkpoint, manifest, anchor)
    # Load actual model bytes on CPU; a marker/config alone is insufficient.
    model = _load_model(checkpoint / "model")
    if any(not bool(torch.isfinite(p).all()) for p in model.parameters()):
        raise ValueError("Checkpoint weights are nonfinite")
    del model
    return checkpoint, run


def make_backup(run_dir, checkpoint, archive):
    run_dir, archive = Path(run_dir), Path(archive)
    files = [p for p in checkpoint.rglob("*") if p.is_file()]
    files += [run_dir / name for name in ("run_manifest.json", "latest.json", "metrics.jsonl", "train_report.json")
              if (run_dir / name).is_file()]
    inventory = {str(p.relative_to(run_dir)): {"sha256": pilot.sha256_file(p), "bytes": p.stat().st_size} for p in files}
    temporary = archive.with_suffix(".part")
    archive.parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(temporary, "w", ZIP_STORED) as bundle:
        bundle.writestr("CHECKPOINT_BACKUP.json", json.dumps({"schema_version": 1, "files": inventory}, indent=2))
        for path in files:
            bundle.write(path, str(path.relative_to(run_dir)))
    temporary.replace(archive)
    return inventory


def restore_backup(archive, destination):
    archive, destination = Path(archive), Path(destination)
    with ZipFile(archive) as bundle:
        inventory = json.loads(bundle.read("CHECKPOINT_BACKUP.json"))["files"]
        if set(bundle.namelist()) != set(inventory) | {"CHECKPOINT_BACKUP.json"}:
            raise ValueError("Backup inventory differs from archive contents")
        for name, entry in inventory.items():
            path = (destination / name).resolve()
            if not path.is_relative_to(destination.resolve()) or bundle.getinfo(name).file_size != entry["bytes"]:
                raise ValueError("Invalid backup path/size")
            path.parent.mkdir(parents=True, exist_ok=True)
            with bundle.open(name) as source, path.open("wb") as target:
                shutil.copyfileobj(source, target)
            if pilot.sha256_file(path) != entry["sha256"]:
                raise ValueError("Backup checksum mismatch")


def ensure_arm(prepared_dir, working_dir, legacy_dir, backup_dir, report_path, *, arm="learned", steps=200,
               device="cuda", smoke_report=None, template_dir=None, force_restart=False, expected_initial_sha=None,
               seed=None, workspace=None, learning_rate=None, gradient_accumulation=None,
               write_mode=None, recurrence_mode=None):
    prepared_dir, working_dir, legacy_dir, backup_dir = map(Path, (prepared_dir, working_dir, legacy_dir, backup_dir))
    if working_dir.resolve() == legacy_dir.resolve() or working_dir.resolve().is_relative_to(legacy_dir.resolve()):
        raise ValueError("Working checkpoints must be separate from historical Drive files")
    manifest = load(prepared_dir / "manifest.json")
    settings = {}
    template = Path(template_dir) if template_dir else legacy_dir
    if (template / "run_manifest.json").is_file():
        try:
            settings = load(template / "run_manifest.json")["contract"]["settings"].copy()
        except (OSError, ValueError, KeyError, TypeError):
            print("Old metadata is unreadable; using the declared pilot settings.", flush=True)
    for name, value in dict(seed=seed, workspace=workspace, learning_rate=learning_rate,
                            gradient_accumulation=gradient_accumulation, recurrence_mode=recurrence_mode).items():
        if value is not None: settings[name] = value
    settings.update(prepared_dir=prepared_dir, output_dir=working_dir, write_mode=write_mode or arm, steps=steps,
                    device=device, smoke_report=smoke_report, resume="none", save_every=25, keep_checkpoints=2,
                    tiny=manifest["tiny"], backbone=manifest["backbone"], revision=manifest["revision"], loops=manifest["loop_budget"])
    config = pilot.PilotConfig(**settings)
    dev = pilot._device(device)
    pilot._seed(config.seed, dev)
    verified_manifest, manifest_sha, tokenizer, records = pilot._prepared(config)
    order = list(range(len(records)))
    random.Random(config.seed).shuffle(order)
    expected = pilot._contract(config, verified_manifest, manifest_sha, order, pilot._runtime(dev))
    # A changed runtime/settings contract gets a separate archive location.
    # Previous completed backups remain available even when an arm must restart.
    contract_id = hashlib.sha256(json.dumps(expected, sort_keys=True).encode()).hexdigest()[:12]
    backup_dir = backup_dir / contract_id
    reasons, selected = [], None
    archive_name = f"{arm}-step{steps:08d}-full.zip"
    stored_archive = backup_dir / archive_name
    candidates = [(working_dir, False), (stored_archive, True), (legacy_dir, False)]
    restored = None
    working_dir.parent.mkdir(parents=True, exist_ok=True)
    if not force_restart:
        for candidate, is_archive in candidates:
            try:
                if is_archive:
                    if not candidate.is_file(): continue
                    restored = Path(tempfile.mkdtemp(prefix="restored-pilot-", dir=working_dir.parent))
                    restore_backup(candidate, restored)
                    candidate = restored
                checkpoint, run = validate_finished(candidate, config, verified_manifest, manifest_sha, expected, expected_initial_sha)
                selected = candidate
                break
            except Exception as error:
                reasons.append(f"{candidate.name}: {type(error).__name__}: {error}")
    action = "reused" if selected else "restarted"
    if selected != working_dir:
        if working_dir.exists():
            preserved = working_dir.with_name(working_dir.name + "-previous-" + uuid.uuid4().hex[:8])
            working_dir.rename(preserved)
        if selected:
            working_dir.mkdir()
            shutil.copytree(checkpoint, working_dir / checkpoint.name)
            for name in ("run_manifest.json", "metrics.jsonl", "train_report.json"):
                if (selected / name).is_file(): shutil.copy2(selected / name, working_dir / name)
        else:
            print(f"{arm}: restarting {steps} small-pilot updates; no checkpoint repair or partial resume.", flush=True)
            original = pilot._initial_model
            def checked_initial(cfg, tok, target):
                model = original(cfg, tok, target)
                if expected_initial_sha and pilot._state_digest(model) != expected_initial_sha:
                    raise ValueError("Fresh initialization differs from the matched arm; no optimizer updates performed")
                return model
            pilot._initial_model = checked_initial
            try:
                pilot.train_pilot(config)
            finally:
                pilot._initial_model = original
    checkpoint = working_dir / f"checkpoint-{steps:08d}"
    atomic_json(working_dir / "latest.json", {"path": checkpoint.name, "step": steps})
    checkpoint, run = validate_finished(working_dir, config, verified_manifest, manifest_sha, expected, expected_initial_sha)
    run.update(status="complete", completed_steps=steps, checkpoint=str(checkpoint))
    atomic_json(working_dir / "run_manifest.json", run)
    local_archive = working_dir.parent / archive_name
    inventory = make_backup(working_dir, checkpoint, local_archive)
    backup_status, backup_error = "verified_mounted_copy", None
    try:
        backup_dir.mkdir(parents=True, exist_ok=True)
        temporary = stored_archive.with_suffix(".part")
        shutil.copy2(local_archive, temporary)
        if pilot.sha256_file(temporary) != pilot.sha256_file(local_archive):
            raise ValueError("Drive archive readback checksum mismatch")
        temporary.replace(stored_archive)
    except Exception as error:
        backup_status, backup_error = "failed", f"{type(error).__name__}: {error}"
        print(f"Drive backup failed; local full checkpoint remains downloadable: {backup_error}", flush=True)
    report = {"complete": True, "action": action, "arm": arm, "step": steps,
              "working_dir": str(working_dir), "local_full_backup": str(local_archive),
              "drive_full_backup": str(stored_archive), "backup_status": backup_status, "backup_error": backup_error,
              "initial_state_sha256": run["initial_state_sha256"], "selection_notes": reasons,
              "recurrence_mode": config.recurrence_mode, "write_mode": config.write_mode,
              "backup_files": inventory,
              "durability_note": "Mounted readback is checked; download the independent full backup before disconnecting."}
    atomic_json(report_path, report)
    if restored: shutil.rmtree(restored, ignore_errors=True)
    print(json.dumps({k: report[k] for k in ("arm", "action", "step", "backup_status", "local_full_backup")}, indent=2))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("prepared-dir", "working-dir", "legacy-dir", "backup-dir", "report-path"):
        parser.add_argument("--"+name, required=True, type=Path)
    parser.add_argument("--arm", default="learned")
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--smoke-report", type=Path)
    parser.add_argument("--template-dir", type=Path)
    parser.add_argument("--expected-initial-sha")
    parser.add_argument("--force-restart", action="store_true")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--workspace", type=int)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--gradient-accumulation", type=int)
    parser.add_argument("--write-mode", choices=("static", "learned", "every_loop", "periodic"))
    parser.add_argument("--recurrence-mode", choices=("standard", "flt_memory"))
    ensure_arm(**vars(parser.parse_args()))
