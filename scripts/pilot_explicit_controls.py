#!/usr/bin/env python3
"""Explicit question-to-CoT controls on a fixed small train/valid subset.

The original model is loaded from the existing HF cache only. Trained controls
use the shared backbone from saved DSA checkpoints, without recurrent memory.
This is a diagnostic comparison, not a matched-compute benchmark.
"""
from pathlib import Path
import argparse
import importlib.metadata
import json
import math
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
from dsa.data.adapters import canonical_record
from dsa.evaluation.evaluate import extract_answer, score_answer
from dsa.provenance import file_sha256
from dsa.training.pilot import _device, _seed
from scripts.diagnose_dsa_pilot import decode, read_json
from scripts.evaluate_dsa import (_checkpoint_files, _load_model, _validate_tokenizers,
                                 _validate_training_identity, resolve_checkpoint)
from scripts.smoke_pretrained_dsa import atomic_json


def explicit_controls(prepared_dir, output_dir, *, run_dir=None, device="cuda", limit=8, max_new_tokens=192):
    if limit < 1 or max_new_tokens < 1:
        raise ValueError("Positive control subset and token cap required")
    prepared_dir, output_dir = Path(prepared_dir).resolve(), Path(output_dir).resolve()
    if output_dir.is_relative_to(prepared_dir) or (run_dir and output_dir.is_relative_to(Path(run_dir).resolve())):
        raise ValueError("Control outputs must be separate from inputs/checkpoints")
    manifest = read_json(prepared_dir / "manifest.json")
    device = _device(device)
    torch.set_num_threads(1)
    identity = {"schema_version": "dsa_explicit_controls_v1", "prepared_manifest_sha256": file_sha256(prepared_dir / "manifest.json"),
                "limit_per_split": limit, "max_new_tokens": max_new_tokens, "device": str(device),
                "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
                "torch": torch.__version__, "transformers": importlib.metadata.version("transformers"),
                "script_sha256": file_sha256(Path(__file__)), "route": "question_newline_to_explicit_continuation"}
    if run_dir:
        run_dir = Path(run_dir).resolve()
        checkpoint = resolve_checkpoint(run_dir)
        anchor, contract = _validate_training_identity(run_dir, checkpoint, manifest, identity["prepared_manifest_sha256"])
        tokenizer = _validate_tokenizers(prepared_dir, checkpoint, manifest, anchor)
        identity.update(checkpoint=str(checkpoint), checkpoint_files_sha256=_checkpoint_files(checkpoint))
    else:
        from scripts.evaluate_dsa import _load_tokenizer
        from dsa.training.pilot import tokenizer_directory_sha256
        if tokenizer_directory_sha256(prepared_dir / "tokenizer") != manifest["tokenizer_sha256"]:
            raise ValueError("Prepared tokenizer differs")
        tokenizer = _load_tokenizer(prepared_dir / "tokenizer", manifest["tiny"])
        identity.update(backbone=manifest["backbone"], revision=manifest["revision"], local_files_only=True)
    source_rows = []
    for split in ("train", "valid"):
        info = manifest["splits"][split]
        path = (prepared_dir / info["path"]).resolve()
        if not path.is_relative_to(prepared_dir) or file_sha256(path) != info["sha256"]:
            raise ValueError("Control source checksum/path differs")
        with path.open(encoding="utf-8") as handle:
            rows = [json.loads(line) for line in handle if line.strip()]
        for index, row in enumerate(rows[:limit]):
            if split == "train":
                ids = row["question_input_ids"]
                gold = decode(tokenizer, row["answer_input_ids"][:-1]).replace(",", "").strip()
                question = decode(tokenizer, ids)
            else:
                row = canonical_record(row)
                question = row["question"].strip() + "\n"
                ids = tokenizer.encode(question, add_special_tokens=False)
                gold = row["final_answer"].replace(",", "").strip()
            source_rows.append({"split": split, "index": index, "sample_id": row["sample_id"],
                                "question": question, "question_input_ids": ids, "gold": gold})
    identity["ordered_examples"] = [[r["split"], r["sample_id"]] for r in source_rows]
    output_dir.mkdir(parents=True, exist_ok=True)
    if (output_dir / "identity.json").exists() and read_json(output_dir / "identity.json") != identity:
        raise ValueError("Explicit-control identity changed; use a new review name")
    atomic_json(output_dir / "identity.json", identity)
    report_path = output_dir / "report.json"
    if report_path.exists():
        report = read_json(report_path)
        if report["identity"] != identity or any(file_sha256(output_dir / name) != digest for name, digest in report["artifacts_sha256"].items()):
            raise ValueError("Explicit control cache changed")
        return report
    _seed(7, device)
    if run_dir:
        wrapper = _load_model(checkpoint / "model")
        model = wrapper.backbone.model
    elif manifest["tiny"]:
        from dsa.models.backbone import BackboneAdapter
        model = BackboneAdapter.tiny(len(tokenizer)).model
    else:
        from transformers import AutoModelForCausalLM
        model = AutoModelForCausalLM.from_pretrained(manifest["backbone"], revision=manifest["revision"],
                    local_files_only=True, attn_implementation="eager", torch_dtype=torch.float32)
    model.to(device).eval()
    results = []
    for ordinal, row in enumerate(source_rows):
        path = output_dir / "examples" / f"{ordinal:04d}.json"
        if path.exists():
            result = read_json(path)
            if result["sample_id"] != row["sample_id"] or result["gold"] != row["gold"]:
                raise ValueError("Saved explicit control source differs")
        else:
            inputs = torch.tensor([row["question_input_ids"]], device=device)
            if device.type == "cuda": torch.cuda.synchronize(device)
            started = time.perf_counter()
            with torch.no_grad():
                generated = model.generate(inputs, attention_mask=torch.ones_like(inputs), do_sample=False,
                    max_new_tokens=max_new_tokens, use_cache=False, pad_token_id=tokenizer.eos_token_id,
                    eos_token_id=tokenizer.eos_token_id)
            if device.type == "cuda": torch.cuda.synchronize(device)
            ids = generated[0, inputs.shape[1]:].tolist()
            text = decode(tokenizer, ids, skip_special_tokens=True)
            answer = extract_answer(text)
            try: valid = math.isfinite(float(answer))
            except ValueError: valid = False
            result = {**row, "generated_token_ids": ids, "generated_text": text, "prediction": answer,
                      "correct": score_answer(answer, row["gold"]), "valid_numeric_answer": valid,
                      "latency_seconds": time.perf_counter()-started, "generated_token_count": len(ids),
                      "stop_reason": "eos" if ids[-1] == tokenizer.eos_token_id else "token_cap"}
            atomic_json(path, result)
        results.append(result)
        print(f"Explicit control: {ordinal+1}/{len(source_rows)}", flush=True)
    report = {"identity": identity, "complete": True, "question_only_generation": True, "recurrent_memory_used": False,
              "optimizer_updates": 0, "predictions": results, "splits": {split: {
                  "total": sum(r["split"] == split for r in results),
                  "correct": sum(r["correct"] for r in results if r["split"] == split)} for split in ("train", "valid")},
              "limitations": ["Small fixed subset; explicit decoding has a different token/compute budget from latent decoding.",
                              "The trained teacher route shares student-finetuned weights; the original control uses cached pretrained weights."]}
    report["artifacts_sha256"] = {str(p.relative_to(output_dir)): file_sha256(p) for p in output_dir.rglob("*.json") if p != report_path}
    atomic_json(report_path, report)
    return report


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for key in ("prepared-dir", "output-dir"): p.add_argument("--"+key, required=True, type=Path)
    p.add_argument("--run-dir", type=Path)
    p.add_argument("--device", default="cuda")
    p.add_argument("--limit", type=int, default=8)
    p.add_argument("--max-new-tokens", type=int, default=192)
    args = p.parse_args()
    result = explicit_controls(**vars(args))
    print(json.dumps({"complete": result["complete"], "splits": result["splits"]}, indent=2))
