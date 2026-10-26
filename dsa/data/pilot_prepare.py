"""Persistent, checksum-verified LOTUS storage and reproducible pilot preparation.

The upstream arrays remain byte-identical and complete. Only the separate pilot
training derivative is sampled. No source inspection checkout or model weights
are needed; the frozen audited manifest supplies the download checksums.
"""

from __future__ import annotations

from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import random
import re
import shutil
import tempfile
from typing import Any
import urllib.request

from dsa.data.adapters import canonical_record, stable_json_hash
from dsa.data.milestones import annotate_record, annotation_report
from dsa.data.tokenize import TinyTokenizer, register_marker_token, tokenize_record
from dsa.provenance import DATA_SOURCE_COMMIT, SPLITS, file_sha256, load_lotus_array, verify_pinned_download_script

REPOSITORY = Path(__file__).resolve().parents[2]
DEFAULT_BACKBONE = "yingfanbot/gsm-cot-gpt2"
DEFAULT_REVISION = "e0ef1887404adbf6c531d2cac7804694321f87bd"
CODE_FILES = ("dsa/data/pilot_prepare.py", "scripts/prepare_dsa_pilot.py", "dsa/data/adapters.py",
              "dsa/data/milestones.py", "dsa/data/tokenize.py", "dsa/provenance.py")


def _json(value: Any) -> str:
    return json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n"


def directory_sha256(directory: Path) -> str:
    """Hash every file, including tokenizer_identity.json, by relative name."""
    entries = [[str(path.relative_to(directory)), file_sha256(path)]
               for path in sorted(directory.rglob("*")) if path.is_file()]
    return hashlib.sha256(json.dumps(entries, separators=(",", ":")).encode()).hexdigest()


def load_pinned_manifest() -> dict[str, Any]:
    manifest = json.loads((REPOSITORY / "provenance/lotus_download_manifest.json").read_text(encoding="utf-8"))
    if manifest["data_source_commit"] != DATA_SOURCE_COMMIT or set(manifest["splits"]) != set(SPLITS):
        raise ValueError("download manifest does not identify the pinned three-split corpus")
    scripts = REPOSITORY / "preprocessing"
    urls = verify_pinned_download_script(scripts / "gsm_icot.bash")["urls"]
    for name, expected in manifest["preprocessing_sha256"].items():
        if file_sha256(scripts / name) != expected:
            raise ValueError(f"pinned preprocessing source changed: {name}")
    for split in SPLITS:
        if manifest["splits"][split]["url"] != urls[split]:
            raise ValueError(f"pinned download URL changed: {split}")
    return manifest


def _verify(path: Path, sha256: str, size: int | None = None) -> None:
    if not path.is_file() or (size is not None and path.stat().st_size != size) or file_sha256(path) != sha256:
        raise ValueError(f"checksum/size mismatch for {path}; existing file preserved, use a fresh storage directory")


def download_verified(url: str, destination: Path, expected: dict[str, Any]) -> bool:
    """Download atomically once; return True on download, False on verified reuse."""
    destination = Path(destination)
    if destination.exists():
        _verify(destination, expected["sha256"], expected["bytes"])
        return False
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, filename = tempfile.mkstemp(prefix=f".{destination.name}.", suffix=".part", dir=destination.parent)
    temporary = Path(filename)
    try:
        request = urllib.request.Request(url, headers={"User-Agent": "dynamic-state-anchoring/1"})
        with os.fdopen(descriptor, "wb") as output, urllib.request.urlopen(request, timeout=120) as response:
            if getattr(response, "status", 200) != 200:
                raise ValueError(f"unexpected HTTP status for {url}")
            shutil.copyfileobj(response, output, length=1024 * 1024)
            output.flush()
            os.fsync(output.fileno())
        _verify(temporary, expected["sha256"], expected["bytes"])
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return True


def convert_upstream(raw_path: Path, json_path: Path, expected: dict[str, Any]) -> bool:
    """Reproduce pinned gsm_icot.py, including literal-space splitting and dump bytes."""
    if json_path.exists():
        _verify(json_path, expected["json_sha256"], expected["json_bytes"])
        return False
    # readlines(), split('||'), strip().split(' '), and default json.dump are
    # intentional: changing whitespace or ensure_ascii changes the audited data.
    with raw_path.open(encoding="utf-8") as handle:
        rows = [{"question": line.split("||")[0],
                 "steps": line.split("||")[1].split("##")[0].strip().split(" "),
                 "answer": line.split("##")[-1].strip()}
                for line in handle.readlines()]
    json_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, filename = tempfile.mkstemp(prefix=f".{json_path.name}.", suffix=".part", dir=json_path.parent)
    temporary = Path(filename)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(rows, handle)
            handle.flush()
            os.fsync(handle.fileno())
        _verify(temporary, expected["json_sha256"], expected["json_bytes"])
        os.replace(temporary, json_path)
    finally:
        temporary.unlink(missing_ok=True)
    return True


def ensure_corpus(storage_dir: Path) -> dict[str, Path]:
    """Keep the complete raw and canonical source corpus under explicit storage."""
    storage_dir = Path(storage_dir).resolve()
    if storage_dir == REPOSITORY or REPOSITORY in storage_dir.parents:
        raise ValueError("--storage-dir must be outside the repository (for Colab, mounted Drive)")
    manifest = load_pinned_manifest()
    paths = {}
    for split in SPLITS:
        expected = manifest["splits"][split]
        raw = storage_dir / "raw" / f"gsm_{split}.txt"
        converted = storage_dir / "data" / f"gsm_{split}.json"
        download_verified(expected["url"], raw, expected)
        convert_upstream(raw, converted, expected)
        paths[split] = converted
    return paths


FIXTURES = {"overfit32": "provenance/mps_experiment_plan_20261019.json"}


def load_fixture(name: str) -> dict[str, Any]:
    """Reviewed TRAIN membership plus its single annotation override."""
    if name not in FIXTURES:
        raise ValueError(f"unknown fixture {name!r}; choose from {', '.join(FIXTURES)}")
    plan_path = REPOSITORY / FIXTURES[name]
    fixture = json.loads(plan_path.read_text(encoding="utf-8"))["fixture"]
    return {**fixture, "name": name, "plan_sha256": file_sha256(plan_path)}


def drop_event(record: dict[str, Any], equation: str) -> dict[str, Any]:
    """Remove one reviewed event and rebuild markers from the unchanged raw rationale."""
    raw, marker = record["rationale_raw"], record["policy"]["marker_token"]
    removed = [event for event in record["events"] if raw[event["start"]:event["end"]] == equation]
    if len(removed) != 1:
        raise ValueError(f"fixture override {equation!r} must match exactly one accepted event")
    events = [{**event, "ordinal": ordinal} for ordinal, event in
              enumerate(event for event in record["events"] if event is not removed[0])]
    chunks, previous = [], 0
    for event in events:
        chunks.extend((raw[previous:event["end"]], marker))
        previous = event["end"]
    chunks.append(raw[previous:])
    rejected = sorted(record["rejected_candidates"] + [{**removed[0], "reason": "reviewed_fixture_override"}],
                      key=lambda item: (item["start"], item["end"], item["reason"]))
    return {**record, "events": events, "event_count": len(events), "rationale_marked": "".join(chunks),
            "rejected_candidates": rejected}


def select_indices(count: int, limit: int, seed: int) -> list[int]:
    if count < 1 or limit < 0:
        raise ValueError("source split must be nonempty and limits must be nonnegative")
    return list(range(count)) if not limit or limit >= count else sorted(random.Random(seed).sample(range(count), limit))


def _source_identity(paths: dict[str, Path], pinned: dict[str, Any]) -> tuple[dict[str, Any], bool]:
    sources = {split: {"path": str(paths[split].resolve()), "sha256": file_sha256(paths[split]),
                       "bytes": paths[split].stat().st_size} for split in SPLITS}
    matches = all(sources[split]["sha256"] == pinned["splits"][split]["json_sha256"] for split in SPLITS)
    return sources, matches


def _audited_provenance(sources: dict[str, Any], primary: bool) -> dict[str, Any]:
    if not primary:
        return {"regime": "local_fixture", "benchmark_claim": False,
                "official_membership": "not_established_for_fixture"}
    split_file = REPOSITORY / "provenance/lotus_split_manifest.json"
    membership_file = REPOSITORY / "provenance/lotus_membership_audit.json"
    audit = json.loads(split_file.read_text(encoding="utf-8"))
    for split in SPLITS:
        if sources[split]["sha256"] != audit["splits"][split]["original_file_sha256"]:
            raise ValueError("source does not match the frozen full-split audit")
    if audit["split_overlap_audit"]["status"] != "passed" or not audit["official_test_comparison"]["ordered_question_answer_match"]:
        raise ValueError("frozen corpus audit does not establish disjoint splits and official test membership")
    return {"regime": "lotus_mirror", "benchmark_claim": "source_matches_frozen_audit; derivative_is_implementation_pilot",
            "split_overlap_audit": audit["split_overlap_audit"],
            "official_test_comparison": audit["official_test_comparison"],
            "ordered_evaluation_sha256": audit["ordered_evaluation_sha256"],
            "audit_sha256s": {"provenance/lotus_split_manifest.json": file_sha256(split_file),
                              "provenance/lotus_membership_audit.json": file_sha256(membership_file)}}


def _verify_reuse(output_dir: Path, request: dict[str, Any]) -> dict[str, Any]:
    try:
        manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
        if manifest["request_identity"] != request:
            raise ValueError("requested source/model/policy/loops/seed/subset/code differs")
        for name, expected in manifest["artifacts"].items():
            _verify(output_dir / name, expected)
        if directory_sha256(output_dir / "tokenizer") != manifest["tokenizer_sha256"]:
            raise ValueError("saved tokenizer contents changed")
        expected_identity = stable_json_hash({"request": request, "tokenizer_sha256": manifest["tokenizer_sha256"]})
        if manifest["cache_identity_sha256"] != expected_identity:
            raise ValueError("prepared cache identity changed")
    except (OSError, KeyError, TypeError, json.JSONDecodeError, ValueError) as exc:
        raise ValueError(f"{output_dir}: stale or invalid prepared cache ({exc}); choose a new --output-dir") from exc
    return manifest


def prepare_pilot(*, storage_dir: Path | None, output_dir: Path, source_dir: Path | None = None,
                  train_limit: int = 32, valid_limit: int = 0, seed: int = 7, loops: int = 16,
                  policy: str = "conclusive", tokenizer_model: str = DEFAULT_BACKBONE,
                  revision: str = DEFAULT_REVISION, tiny: bool = False,
                  fixture: str | None = None, steps_join: str = "space") -> dict[str, Any]:
    fixture_plan = load_fixture(fixture) if fixture is not None else None
    if fixture_plan is not None:
        train_limit = fixture_plan["expected_records"]
    if loops < 1 or train_limit < 0 or valid_limit < 0:
        raise ValueError("loops must be positive and split limits nonnegative")
    if policy not in {"conclusive", "verified_only"}:
        raise ValueError("unsupported annotation policy")
    if steps_join not in {"space", "newline"}:
        raise ValueError("steps_join must be space or newline")
    if not re.fullmatch(r"[a-f0-9]{40}", revision):
        raise ValueError("--revision must be an immutable full 40-character commit SHA")
    pinned = load_pinned_manifest()
    if source_dir is not None:
        paths = {split: Path(source_dir) / f"gsm_{split}.json" for split in SPLITS}
    elif storage_dir is not None:
        paths = ensure_corpus(storage_dir)
    else:
        raise ValueError("provide --storage-dir for persistent downloads or --source-dir for existing arrays")
    sources, primary = _source_identity(paths, pinned)
    provenance = _audited_provenance(sources, primary)
    request = {"sources": {split: {key: value for key, value in sources[split].items() if key != "path"} for split in SPLITS},
               "tiny": tiny, "backbone": tokenizer_model, "revision": revision, "policy": policy,
               "loop_budget": loops, "seed": seed, "train_limit": train_limit, "valid_limit": valid_limit,
               "code_sha256s": {name: file_sha256(REPOSITORY / name) for name in CODE_FILES},
               "pinned_download_manifest_sha256": file_sha256(REPOSITORY / "provenance/lotus_download_manifest.json"),
               "audit_provenance": provenance}
    if fixture_plan is not None:
        request["fixture"] = {"name": fixture_plan["name"], "plan_sha256": fixture_plan["plan_sha256"]}
    if steps_join != "space":
        request["steps_join"] = steps_join
    output_dir = Path(output_dir).resolve()
    if output_dir in [path.resolve() for path in paths.values()] or any(output_dir in path.resolve().parents for path in paths.values()):
        raise ValueError("output directory must not contain or replace the source arrays")
    if output_dir.exists() and (not output_dir.is_dir() or any(output_dir.iterdir())):
        return _verify_reuse(output_dir, request)
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.", dir=output_dir.parent))
    try:
        if tiny:
            tokenizer = TinyTokenizer()
        else:
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_model, revision=revision, use_fast=True)
            if tokenizer.pad_token_id is None:
                tokenizer.pad_token = tokenizer.eos_token
        marker_id = register_marker_token(tokenizer)
        tokenizer_dir = staging / "tokenizer"
        tokenizer.save_pretrained(tokenizer_dir)
        tokenizer_files = {str(path.relative_to(tokenizer_dir)): file_sha256(path)
                           for path in sorted(tokenizer_dir.rglob("*")) if path.is_file()}
        tokenizer_identity = {"schema_version": "dsa_pilot_tokenizer_v1", "tokenizer_class": type(tokenizer).__name__,
                              "source": "dsa_tiny_utf8_v1" if tiny else tokenizer_model,
                              "revision": revision, "marker_token": "<SA>", "marker_token_id": marker_id,
                              "tokenizer_vocab_sha256": stable_json_hash(tokenizer.get_vocab()),
                              "tokenizer_file_sha256s": tokenizer_files,
                              "tokenizer_config_sha256": stable_json_hash(tokenizer_files),
                              "pad_token_id": tokenizer.pad_token_id, "eos_token_id": tokenizer.eos_token_id}
        (tokenizer_dir / "tokenizer_identity.json").write_text(_json(tokenizer_identity), encoding="utf-8")
        tokenizer_sha = directory_sha256(tokenizer_dir)
        selections, splits, reports = {}, {}, {}
        fixture_questions: dict[str, set[str]] = {}
        for split in SPLITS:
            rows = load_lotus_array(paths[split])
            if primary:
                expected_count = json.loads((REPOSITORY / "provenance/lotus_split_manifest.json").read_text(encoding="utf-8"))["splits"][split]["rows"]
                if len(rows) != expected_count:
                    raise ValueError(f"pinned {split} record count differs from frozen audit")
            indices = select_indices(len(rows), train_limit if split == "train" else valid_limit if split == "valid" else 0, seed)
            if fixture_plan is not None and split == "train":
                indices = [item["source_row_number"] - 1 for item in fixture_plan["training_rows"]]
            filename = "train.tokenized.jsonl" if split == "train" else f"{split}.jsonl"
            chosen, annotated, tokens = [], [], []
            if not primary:
                fixture_questions[split] = {canonical_record(row)["question_hash"] for row in rows}
                if any(fixture_questions[split] & questions for other, questions in fixture_questions.items() if other != split):
                    raise ValueError("cross-partition question overlap in fixture; no artifacts installed")
            with (staging / filename).open("w", encoding="utf-8") as handle:
                for index in indices:
                    canonical = canonical_record(rows[index], steps_join="space")
                    chosen.append({"source_row_number": index + 1, "sample_id": canonical["sample_id"],
                                   "source_problem_id": canonical["source_problem_id"], "question_hash": canonical["question_hash"]})
                    if split == "train":
                        # Row identity stays space-joined; only the training text takes the requested join.
                        text_source = canonical if steps_join == "space" else canonical_record(rows[index], steps_join=steps_join)
                        record = annotate_record(text_source, mode=policy, steps_join=steps_join)
                        if fixture_plan is not None:
                            reviewed = fixture_plan["training_rows"][len(annotated)]
                            if canonical["sample_id"] != reviewed["sample_id"]:
                                raise ValueError(f"fixture row {index + 1} differs from its reviewed identity")
                            override = fixture_plan["annotation_override"]
                            if index + 1 == override["source_row_number"]:
                                record = drop_event(record, override["remove_equation"])
                        annotated.append(record)
                        tokenized = tokenize_record(record, tokenizer, loops)
                        tokens.append(tokenized)
                        derivative = {"schema_version": "dsa_tokenized_v1", "question_hash": canonical["question_hash"],
                                      "annotation_policy_sha256": stable_json_hash(record["policy"]), **tokenized,
                                      "source_row_number": index + 1, "source_split_sha256": sources[split]["sha256"]}
                    else:
                        derivative = {**canonical, "source_row_number": index + 1, "source_split_sha256": sources[split]["sha256"]}
                    handle.write(json.dumps(derivative, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n")
            selections[split] = {"source_sha256": sources[split]["sha256"], "source_records": len(rows),
                                 "selected_records": len(indices), "selection": chosen,
                                 "ordered_sample_ids_sha256": stable_json_hash([item["sample_id"] for item in chosen]),
                                 "selection_method": (f"reviewed fixture {fixture} from {FIXTURES[fixture]}" if fixture_plan is not None and split == "train"
                                                      else "entire original split in source order" if len(indices) == len(rows)
                                                      else "random.Random(seed).sample without replacement, restored source order")}
            splits[split] = {"path": filename, "sha256": file_sha256(staging / filename), "records": len(indices),
                             "source_sha256": sources[split]["sha256"],
                             "ordered_sample_ids_sha256": selections[split]["ordered_sample_ids_sha256"]}
            if split == "train" and fixture_plan is not None:
                counts = (len(annotated), sum(record["event_count"] for record in annotated),
                          sum(record["event_count"] == 0 for record in annotated))
                if counts != (fixture_plan["expected_records"], fixture_plan["expected_checkpoint_events"],
                              fixture_plan["expected_no_event_controls"]):
                    raise ValueError(f"fixture counts {counts} differ from the reviewed plan; nothing installed")
            if split == "train":
                reasons = Counter(token["controller_mask_reason"] for token in tokens if not token["controller_supervised"])
                reports[split] = {**annotation_report(annotated), "loop_budget": loops, "truncation": False,
                                  "controller_supervised_count": sum(token["controller_supervised"] for token in tokens),
                                  "controller_mask_reasons": dict(reasons), "controller_label_overflow": reasons["event_overflow"],
                                  "tokenizer_mapping_failures": 0,
                                  "max_explicit_length": max(len(token["explicit_input_ids"]) for token in tokens),
                                  "max_question_length": max(len(token["question_input_ids"]) for token in tokens)}
            del rows
        (staging / "selections.json").write_text(_json(selections), encoding="utf-8")
        (staging / "preparation_report.json").write_text(_json(reports), encoding="utf-8")
        artifacts = {name: file_sha256(staging / name) for name in ("train.tokenized.jsonl", "valid.jsonl", "test.jsonl", "selections.json", "preparation_report.json")}
        manifest = {"schema_version": 1, "backbone": tokenizer_model, "revision": revision, "tiny": tiny,
                    "loop_budget": loops, "tokenizer_sha256": tokenizer_sha, "splits": splits,
                    "model": {"id": tokenizer_model, "revision": revision}, "tokenizer": {"path": "tokenizer", **tokenizer_identity},
                    "metadata": {"loop_budget": loops, "policy": policy, "seed": seed,
                                 "training_regime": "implementation_pilot", "benchmark_result": False,
                                 "train_limit": train_limit, "valid_limit": valid_limit, "fixture": fixture,
                                 "steps_join": steps_join,
                                 "test_used_for_selection": False, "source_rows_filtered": False,
                                 "all_no_event_and_overflow_records_retained": True},
                    "sources": sources, "source_provenance": provenance,
                    "request_identity": request, "cache_identity_sha256": stable_json_hash({"request": request, "tokenizer_sha256": tokenizer_sha}),
                    "artifacts": artifacts}
        (staging / "manifest.json").write_text(_json(manifest), encoding="utf-8")
        if output_dir.exists():
            output_dir.rmdir()  # An existing empty directory is safe to replace.
        os.replace(staging, output_dir)
        return manifest
    finally:
        if staging.exists():
            shutil.rmtree(staging)
