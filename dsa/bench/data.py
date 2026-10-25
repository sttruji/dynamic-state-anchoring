"""Pinned sources, normalization, deterministic tagging and fixed splits.

Every output is canonical JSONL (sorted keys, fixed separators, LF), so the same
pinned inputs give byte-identical files on any machine. Hashes are checked
against ``provenance/bench_data_lock.json``; a mismatch stops setup.

Splits follow the proposal (section 4.2):
  training corpora   gsm8k_aug, gsm8k_aug_nl, csqa_cot        (train/*.jsonl)
  development        dev64 (first 64 validation), gsm8k_valid  (eval/*.jsonl)
  in-distribution    gsm8k_test                                (eval/, needs --test)
  out-of-distribution svamp, gsm_hard, multiarith              (eval/, needs --test)
  commonsense         csqa (CommonsenseQA-CoT validation)       (eval/, needs --test)
  overfit fixture    overfit32: the reviewed 32 TRAIN questions from
                     provenance/mps_experiment_plan_20261019.json, with its one
                     reviewed annotation override applied.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

from dsa.data.adapters import canonical_record, normalized_question_hash
from dsa.data.milestones import annotate_record

REPOSITORY = Path(__file__).resolve().parents[2]
LOCK_PATH = REPOSITORY / "provenance" / "bench_data_lock.json"
FIXTURE_PLAN = REPOSITORY / "provenance" / "mps_experiment_plan_20261019.json"
ANNOTATION_POLICY = "conclusive"
LOCK_SCHEMA = "dsa_bench_data_lock_v1"


def default_data_dir() -> Path:
    return Path(os.environ.get("DSA_DATA_DIR", REPOSITORY / "data")).resolve()


# name -> (file under eval/, answer type, needs --test)
EVAL_DATASETS: dict[str, dict[str, Any]] = {
    "overfit32": {"answer_type": "numeric", "test": False,
                  "about": "reviewed 32 TRAIN questions; measures fitting, not generalization"},
    "dev64": {"answer_type": "numeric", "test": False,
              "about": "first 64 GSM8K-Aug validation questions (frozen development set)"},
    "gsm8k_valid": {"answer_type": "numeric", "test": False, "about": "all 500 GSM8K-Aug validation questions"},
    "gsm8k_test": {"answer_type": "numeric", "test": True, "about": "official GSM8K test (1,319), in-distribution"},
    "svamp": {"answer_type": "numeric", "test": True, "about": "SVAMP, out-of-distribution"},
    "gsm_hard": {"answer_type": "numeric", "test": True, "about": "GSM-Hard, out-of-distribution"},
    "multiarith": {"answer_type": "numeric", "test": True, "about": "MultiArith, out-of-distribution"},
    "csqa": {"answer_type": "choice", "test": True, "about": "CommonsenseQA-CoT validation (CODI's test protocol)"},
}
SUITES = {
    "overfit": ["overfit32"],
    "dev": ["dev64"],
    "valid": ["gsm8k_valid"],
    "ood": ["svamp", "gsm_hard", "multiarith"],
    "test": ["gsm8k_test", "svamp", "gsm_hard", "multiarith"],
    "csqa": ["csqa"],
}


def eval_file_key(name: str) -> str:
    return f"eval/{name}.jsonl"


def canonical_line(row: dict) -> str:
    return json.dumps(row, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False) + "\n"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_lock() -> dict:
    if not LOCK_PATH.exists():
        return {"schema": LOCK_SCHEMA, "files": {}}
    return json.loads(LOCK_PATH.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict]:
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> dict:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.part")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(canonical_line(row))
    os.replace(temporary, path)
    return {"sha256": file_sha256(path), "rows": len(rows)}


TRAIN_CORPORA = ("overfit32", "gsm8k_aug", "gsm8k_aug_nl", "csqa_cot")


def load_eval_rows(name: str, data_dir: Path | None = None) -> tuple[list[dict], str]:
    """Rows of one evaluation dataset, after checking its hash against the lock."""
    if name not in EVAL_DATASETS:
        raise ValueError(f"unknown dataset {name!r}; choose from {', '.join(EVAL_DATASETS)}")
    return _load_locked(eval_file_key(name), data_dir)


def load_train_rows(name: str, data_dir: Path | None = None) -> tuple[list[dict], str]:
    """Rows of one training corpus, after checking its hash against the lock."""
    if name not in TRAIN_CORPORA:
        raise ValueError(f"unknown training corpus {name!r}; choose from {', '.join(TRAIN_CORPORA)}")
    return _load_locked(f"train/{name}.jsonl", data_dir)


def _load_locked(key: str, data_dir: Path | None) -> tuple[list[dict], str]:
    path = (data_dir or default_data_dir()) / "bench" / key
    if not path.is_file():
        raise FileNotFoundError(f"{path} is missing; run: python scripts/setup_data.py")
    sha = file_sha256(path)
    expected = load_lock().get("files", {}).get(key, {}).get("sha256")
    if expected is None:
        raise ValueError(f"{key} is not in {LOCK_PATH.name}; results would not be comparable")
    if sha != expected:
        raise ValueError(f"{path} differs from the locked team copy; rerun python scripts/setup_data.py")
    return read_jsonl(path), sha


# ---------------------------------------------------------------- GSM8K-Aug

def ensure_gsm8k_aug(data_dir: Path) -> dict[str, Path]:
    """Download the pinned LOTUS mirror (Deng et al. GSM8K-Aug) into data/gsm_*.json.

    This is the exact layout ``scripts/prepare_dsa_pilot.py --source-dir data``
    already expects, so the existing DSA trainer keeps working unchanged.
    """
    from dsa.data.pilot_prepare import convert_upstream, download_verified, load_pinned_manifest

    manifest = load_pinned_manifest()
    paths = {}
    for split, expected in manifest["splits"].items():
        raw, converted = data_dir / f"gsm_{split}.txt", data_dir / f"gsm_{split}.json"
        if not converted.exists():
            download_verified(expected["url"], raw, expected)
        convert_upstream(raw, converted, expected)  # Verifies the hash when already present.
        paths[split] = converted
    return paths


def _tag(canonical: dict) -> dict:
    """Deterministic <SA> milestone tags; character offsets into ``rationale``."""
    annotated = annotate_record(canonical, mode=ANNOTATION_POLICY, steps_join="space")
    return {"rationale_marked": annotated["rationale_marked"], "event_count": annotated["event_count"],
            "events": [{key: event[key] for key in ("start", "end", "text", "value", "rule", "validation")}
                       for event in annotated["events"]]}


def _drop_events(row: dict, equation: str) -> dict:
    """Remove a reviewed event and rebuild the marked rationale from the raw text."""
    raw = row["rationale"]
    kept = [event for event in row["events"] if raw[event["start"]:event["end"]] != equation]
    if len(kept) != len(row["events"]) - 1:
        raise ValueError(f"override {equation!r} must match exactly one event")
    marked, previous = [], 0
    for event in kept:
        marked += [raw[previous:event["end"]], "<SA>"]
        previous = event["end"]
    marked.append(raw[previous:])
    return {**row, "events": kept, "event_count": len(kept), "rationale_marked": "".join(marked),
            "annotation_override": f"removed {equation} (reviewed given-restatement)"}


def gsm8k_row(source: dict, dataset: str, split: str, number: int, *, tag: bool) -> dict:
    canonical = canonical_record(source, steps_join="space")
    row = {"id": f"{dataset}/{split}/{number:06d}", "dataset": dataset, "split": split,
           "source_row": number, "sample_id": canonical["sample_id"], "question": canonical["question"],
           "rationale": canonical["rationale_raw"], "answer": canonical["final_answer"],
           "answer_type": "numeric"}
    return {**row, **_tag(canonical)} if tag else row


def build_gsm8k_aug(data_dir: Path, *, with_train: bool) -> dict[str, list[dict]]:
    paths = ensure_gsm8k_aug(data_dir)
    out: dict[str, list[dict]] = {}
    loaded = {split: json.loads(paths[split].read_text(encoding="utf-8")) for split in ("valid", "test")}
    valid = [gsm8k_row(row, "gsm8k_aug", "valid", i + 1, tag=False) for i, row in enumerate(loaded["valid"])]
    out["eval/gsm8k_valid.jsonl"] = valid
    out["eval/dev64.jsonl"] = valid[:64]
    out["eval/gsm8k_test.jsonl"] = [gsm8k_row(row, "gsm8k_aug", "test", i + 1, tag=False)
                                    for i, row in enumerate(loaded["test"])]
    train = json.loads(paths["train"].read_text(encoding="utf-8"))
    out["train/overfit32.jsonl"] = build_overfit32(train)
    out["eval/overfit32.jsonl"] = [{key: row[key] for key in ("id", "dataset", "split", "source_row", "sample_id",
                                                               "question", "rationale", "answer", "answer_type")}
                                   for row in out["train/overfit32.jsonl"]]
    if with_train:
        out["train/gsm8k_aug.jsonl"] = [gsm8k_row(row, "gsm8k_aug", "train", i + 1, tag=True)
                                        for i, row in enumerate(train)]
    return out


def build_overfit32(train: list[dict]) -> list[dict]:
    """The reviewed 32-question fixture, with identities and counts asserted."""
    plan = json.loads(FIXTURE_PLAN.read_text(encoding="utf-8"))["fixture"]
    override = plan["annotation_override"]
    rows = []
    for item in plan["training_rows"]:
        number = item["source_row_number"]
        row = gsm8k_row(train[number - 1], "gsm8k_aug", "train", number, tag=True)
        if row["sample_id"] != item["sample_id"]:
            raise ValueError(f"fixture row {number} no longer matches its reviewed identity")
        if number == override["source_row_number"]:
            row = _drop_events(row, override["remove_equation"])
        rows.append({**row, "dataset": "overfit32"})
    events = sum(row["event_count"] for row in rows)
    no_event = sum(row["event_count"] == 0 for row in rows)
    if (len(rows), events, no_event) != (plan["expected_records"], plan["expected_checkpoint_events"],
                                         plan["expected_no_event_controls"]):
        raise ValueError(f"overfit32 counts {(len(rows), events, no_event)} differ from the reviewed plan")
    return rows


# ------------------------------------------------------- Hugging Face sources
# Pinned to immutable dataset commits. Field conventions follow the CODI and
# LOTUS evaluation scripts (SVAMP uses train+test as question_concat; MultiArith
# uses test; GSM-Hard uses all 1,319 rows; CommonsenseQA evaluates validation).

HF_SOURCES: dict[str, dict[str, Any]] = {
    "gsm8k_aug_nl": {"repo": "zen-E/GSM8k-Aug-NL", "revision": "e20551f801620cc7dd968edc941126e493fa8202",
                     "files": ["gsm8k_aug_nl.json"], "train": True},
    "csqa_cot": {"repo": "zen-E/CommonsenseQA-GPT4omini", "revision": "742d45af29380bacb270feb75b6853dda1e64034",
                 "files": ["commonsense_cot_train.json", "commonsense_cot_dev.json"], "train": True},
    "svamp": {"repo": "ChilleD/SVAMP", "revision": "5e0bf1e5e7c0e9c4bc39180d224f41f3f801b7ef",
              "files": ["train.json", "test.json"], "train": False},
    "gsm_hard": {"repo": "reasoning-machines/gsm-hard", "revision": "960448f73503112d4226baeb8eb41d3fb5ae2506",
                 "files": ["gsmhardv2.jsonl"], "train": False},
    "multiarith": {"repo": "ChilleD/MultiArith", "revision": "144d44c3fb87c0b9097ac9593c789e716a282e3e",
                   "files": ["test.json"], "train": False},
}


def _download(data_dir: Path, source: dict, filename: str) -> Any:
    from huggingface_hub import hf_hub_download

    target = data_dir / "raw" / source["repo"].replace("/", "__")
    path = Path(hf_hub_download(source["repo"], filename, repo_type="dataset",
                                revision=source["revision"], local_dir=target))
    text = path.read_text(encoding="utf-8")
    if filename.endswith(".jsonl"):
        return [json.loads(line) for line in text.split("\n") if line.strip()]
    return json.loads(text)


def _number(value: Any) -> str:
    number = float(str(value).replace(",", ""))
    return str(int(number)) if number.is_integer() else repr(number)


def _eval_row(dataset: str, split: str, number: int, question: str, answer: str,
              answer_type: str = "numeric") -> dict:
    return {"id": f"{dataset}/{split}/{number:06d}", "dataset": dataset, "split": split, "source_row": number,
            "question": question, "answer": answer, "answer_type": answer_type}


def _train_row(dataset: str, number: int, question: str, rationale: str, answer: str,
               answer_type: str = "numeric") -> dict:
    canonical = canonical_record({"question": question, "rationale_raw": rationale, "final_answer": answer})
    return {"id": f"{dataset}/train/{number:06d}", "dataset": dataset, "split": "train", "source_row": number,
            "sample_id": canonical["sample_id"], "question": question, "rationale": rationale,
            "answer": answer, "answer_type": answer_type, **_tag(canonical)}


def build_hf(data_dir: Path, name: str, *, with_train: bool) -> dict[str, list[dict]]:
    source = HF_SOURCES[name]
    if source["train"] and not with_train and name != "csqa_cot":
        return {}
    files = [_download(data_dir, source, filename) for filename in source["files"]]
    if name == "gsm8k_aug_nl":
        columns = files[0]
        return {"train/gsm8k_aug_nl.jsonl": [
            _train_row(name, i + 1, q, r, a) for i, (q, r, a) in
            enumerate(zip(columns["question"], columns["cot"], columns["answer"]))]}
    if name == "csqa_cot":
        train, dev = files
        out = {"eval/csqa.jsonl": [_eval_row("csqa", "validation", i + 1, row["question"], row["answer"], "choice")
                                   for i, row in enumerate(dev)]}
        if with_train:
            out["train/csqa_cot.jsonl"] = [_train_row(name, i + 1, row["question"], row["cot"], row["answer"], "choice")
                                           for i, row in enumerate(train)]
        return out
    if name == "svamp":
        rows = [(split, i + 1, row) for split, part in zip(("train", "test"), files) for i, row in enumerate(part)]
        return {"eval/svamp.jsonl": [_eval_row(name, split, number, f"{row['Body'].strip()} {row['Question'].strip()}",
                                               _number(row["Answer"])) for split, number, row in rows]}
    if name == "gsm_hard":
        return {"eval/gsm_hard.jsonl": [_eval_row(name, "train", i + 1, row["input"], _number(row["target"]))
                                        for i, row in enumerate(files[0])]}
    if name == "multiarith":
        return {"eval/multiarith.jsonl": [_eval_row(name, "test", i + 1, row["question"], _number(row["final_ans"]))
                                          for i, row in enumerate(files[0])]}
    raise ValueError(f"no normalizer for {name}")


def build_all(data_dir: Path, *, with_train: bool = True) -> dict[str, list[dict]]:
    outputs = build_gsm8k_aug(data_dir, with_train=with_train)
    for name in HF_SOURCES:
        outputs.update(build_hf(data_dir, name, with_train=with_train))
    return outputs


def contamination_report(outputs: dict[str, list[dict]]) -> dict[str, int]:
    """Count evaluation questions that also appear in any training corpus."""
    train = set()
    for key, rows in outputs.items():
        if key.startswith("train/") and key != "train/overfit32.jsonl":
            train.update(normalized_question_hash(row["question"]) for row in rows)
    return {key: sum(normalized_question_hash(row["question"]) in train for row in rows)
            for key, rows in outputs.items()
            if key.startswith("eval/") and key != "eval/overfit32.jsonl" and train}
