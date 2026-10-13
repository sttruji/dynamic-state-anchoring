#!/usr/bin/env python3
"""Training-only checkpoint diagnostics; no optimizer or checkpoint writes.

Existing model/data contracts remain unchanged. State hooks measure gradients
through the actual concatenated recurrent inputs, rather than an unused stack of
state copies. Text generation is an outcome measure, not a latent-state decoder.
"""
from contextlib import contextmanager
from dataclasses import replace
import argparse
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import platform
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
from dsa.data.adapters import stable_json_hash
from dsa.data.tokenize import TinyTokenizer
from dsa.evaluation.evaluate import extract_answer, score_answer
from dsa.provenance import file_sha256
from dsa.training.objectives import LossConfig, joint_objective
from dsa.training.pilot import _device, _seed, _state_digest
from scripts.evaluate_dsa import (_checkpoint_files, _load_model, _recover_predictions,
                                 _validate_tokenizers, _validate_training_identity,
                                 resolve_checkpoint)
from scripts.smoke_pretrained_dsa import atomic_json


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def decode(tokenizer, ids, *, skip_special_tokens=False):
    kwargs = {} if isinstance(tokenizer, TinyTokenizer) else {"clean_up_tokenization_spaces": False}
    return tokenizer.decode(ids, skip_special_tokens=skip_special_tokens, **kwargs)


def finite_number(value):
    value = float(value)
    return value if math.isfinite(value) else None


def norm(tensor):
    return finite_number(torch.linalg.vector_norm(tensor.detach(), dtype=torch.float64).item())


def parameter_gradients(model):
    squares, missing, nonfinite = 0., 0, 0
    groups = {}
    for name, parameter in model.named_parameters():
        if parameter.grad is None:
            missing += 1
            continue
        value = norm(parameter.grad)
        group = "backbone" if name.startswith("backbone.") else name.split(".")[0]
        if value is None:
            nonfinite += 1
            groups[group] = None
        else:
            squares += value * value
            if groups.get(group, 0.) is not None:
                groups[group] = groups.get(group, 0.) + value * value
    return {"global_norm_before_clipping": math.sqrt(squares) if not nonfinite else None,
            "nonfinite_parameter_gradients": nonfinite, "parameters_without_gradient": missing,
            "group_norms": {key: math.sqrt(value) if value is not None else None
                            for key, value in groups.items()}}


@contextmanager
def capture_recurrence(model, budget, *, teacher_first=False):
    """Capture original graph tensors; remove instrumentation before backward."""
    original = model.backbone.forward_embeds
    states, workspaces, calls = [], [], [0]

    def forward(inputs_embeds, attention_mask, position_ids):
        calls[0] += 1
        first = 2 if teacher_first else 1
        if first <= calls[0] < first + budget:
            inputs_embeds.retain_grad()
            states.append(inputs_embeds)
        return original(inputs_embeds, attention_mask, position_ids)

    def workspace_hook(module, inputs, output):
        output.retain_grad()
        workspaces.append(output)

    handle = model.workspace_norm.register_forward_hook(workspace_hook)
    model.backbone.forward_embeds = forward
    try:
        yield states, workspaces
        if len(states) != budget or len(workspaces) != budget:
            raise RuntimeError("Diagnostic hooks did not capture exactly one state per latent loop")
    finally:
        model.backbone.forward_embeds = original
        handle.remove()


def state_statistics(states, workspaces, question_width):
    result, previous = [], None
    for i, (state, workspace) in enumerate(zip(states, workspaces)):
        memory = state.detach()[:, question_width:]
        flat = memory.flatten().double()
        length = torch.linalg.vector_norm(flat).item()
        cosine, delta = None, None
        if previous is not None:
            denominator = torch.linalg.vector_norm(previous).item() * length
            if denominator:
                cosine = finite_number(torch.dot(previous, flat).item() / denominator)
            delta = finite_number(torch.linalg.vector_norm(flat - previous).item())
        grad = state.grad
        result.append({"loop": i + 1, "memory_input_rms": finite_number(length / math.sqrt(flat.numel())),
                       "memory_input_change_l2": delta, "memory_input_cosine_to_previous": cosine,
                       "anchor_input_l2": norm(memory[:, :1]), "workspace_input_l2": norm(memory[:, 1:]),
                       "connected_to_loss": grad is not None,
                       "anchor_input_gradient_l2": norm(grad[:, question_width:question_width+1]) if grad is not None else None,
                       "workspace_input_gradient_l2": norm(grad[:, question_width+1:]) if grad is not None else None,
                       "workspace_output_gradient_l2": norm(workspace.grad) if workspace.grad is not None else None})
        previous = flat
    return result


def gradient_probe(model, record, prefix, budget, weights, seed, *, full_objective=False):
    device = model.initial_anchor.device
    _seed(seed, device)
    model.train()  # Preserve training dropout and the straight-through gate.
    model.zero_grad(set_to_none=True)
    with torch.enable_grad(), capture_recurrence(model, budget, teacher_first=full_objective) as captured:
        if full_objective:
            if budget != model.config.loop_budget:
                raise ValueError("Full objective uses the original training budget and labels")
            output = joint_objective(model, record, prefix, weights)
            losses = {"answer": output["answer"], "cot": weights.cot * output["cot"],
                      "gate": weights.gate * output["gate"], "anchor": weights.anchor * output["anchor"],
                      "total": output["total"]}
        else:
            def ids(values):
                return torch.tensor([values], device=device, dtype=torch.long)
            question = ids(record["question_input_ids"])
            loop = model(question, torch.ones_like(question), loop_budget=budget)
            suffix = ids(record["latent_suffix_input_ids"])
            logits = model.answer_logits(loop, suffix, torch.ones_like(suffix))
            loss = torch.nn.functional.cross_entropy(logits[:, :-1].reshape(-1, logits.shape[-1]),
                    ids(record["latent_suffix_labels"])[:, 1:].reshape(-1), ignore_index=-100)
            losses = {"answer": loss}
    states, workspaces = captured
    profiles = {}
    try:
        for i, (name, loss) in enumerate(losses.items()):
            model.zero_grad(set_to_none=True)
            for tensor in states + workspaces:
                tensor.grad = None
            loss.backward(retain_graph=i < len(losses) - 1)
            profiles[name] = {"weighted_loss": finite_number(loss.detach().item()),
                              "parameter_gradients": parameter_gradients(model),
                              "states": state_statistics(states, workspaces, len(record["question_input_ids"]))}
        return {"sample_id": record["sample_id"], "loop_budget": budget,
                "kind": "joint_loss_contributions" if full_objective else "answer_only_depth_probe",
                "seed": seed, "training_mode": True, "gradient_clipping_applied": False,
                "controller_supervised": record["controller_supervised"], "losses": profiles}
    finally:
        model.zero_grad(set_to_none=True)


@contextmanager
def inference_condition(model, mode):
    original = model.config
    if mode not in {"normal", "no_writes", "all_writes"}:
        raise ValueError("Unknown diagnostic inference condition")
    override = {"normal": original.write_mode, "no_writes": "static", "all_writes": "every_loop"}[mode]
    model.config = replace(original, write_mode=override)
    try:
        yield
    finally:
        model.config = original


def predict_record(model, tokenizer, record, row, index, prefix, budget, cap):
    device = model.initial_anchor.device
    question = torch.tensor([record["question_input_ids"]], device=device, dtype=torch.long)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    generated = model.generate(question, torch.ones_like(question), answer_prefix_ids=prefix,
                               loop_budget=budget, max_new_tokens=cap)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    token_ids = generated.token_ids[0].tolist()
    text = decode(tokenizer, token_ids, skip_special_tokens=True)
    prediction = extract_answer(text)
    try:
        valid = math.isfinite(float(prediction))
    except ValueError:
        valid = False
    return {"index": index, "sample_id": row["sample_id"], "question_hash": row["question_hash"], "question": row["question"],
            "gold": row["final_answer"], "generated_text": text, "prediction": prediction,
            "correct": score_answer(prediction, row["final_answer"]), "valid_numeric_answer": valid,
            "latency_seconds": elapsed, "recurrent_forwards": budget,
            "gate_probabilities": generated.loop.gate_logits[0].sigmoid().tolist(),
            "writes": generated.loop.writes[0].tolist(), "generated_token_ids": token_ids,
            "generated_token_count": len(token_ids), "stop_reason": "eos" if token_ids[-1] == model.config.eos_token_id else "token_cap"}


def summarize_predictions(saved, budget, records):
    result = {"total": len(saved), "correct": sum(r["correct"] for r in saved),
              "accuracy": sum(r["correct"] for r in saved) / len(saved),
              "invalid_answer_rate": sum(not r["valid_numeric_answer"] for r in saved) / len(saved),
              "mean_write_rate": sum(sum(r["writes"]) for r in saved) / (len(saved) * budget),
              "mean_latency_seconds": sum(r["latency_seconds"] for r in saved) / len(saved),
              "stop_reasons": {kind: sum(r["stop_reason"] == kind for r in saved) for kind in ("eos", "token_cap")}}
    supervised = [(prediction, record) for prediction, record in zip(saved, records)
                  if record["controller_supervised"] and len(record["gate_targets"]) == budget]
    if supervised:
        pairs = [(int(write), int(target)) for prediction, record in supervised
                 for write, target in zip(prediction["writes"], record["gate_targets"])]
        tp = sum(write == target == 1 for write, target in pairs)
        fp = sum(write == 1 and target == 0 for write, target in pairs)
        fn = sum(write == 0 and target == 1 for write, target in pairs)
        result["controller_training_labels"] = {
            "supervised_records": len(supervised), "decisions": len(pairs),
            "target_write_rate": sum(target for _, target in pairs) / len(pairs),
            "executed_write_rate": sum(write for write, _ in pairs) / len(pairs),
            "true_positives": tp, "false_positives": fp, "false_negatives": fn,
            "precision": tp / (tp + fp) if tp + fp else None,
            "recall": tp / (tp + fn) if tp + fn else None}
    return result


def prediction_journal(model, tokenizer, records, rows, path, prefix, budget, cap):
    saved = _recover_predictions(path, rows, budget)
    for row in saved:
        ids = row["generated_token_ids"]
        if (not ids or len(ids) > cap or row["generated_token_count"] != len(ids)
                or decode(tokenizer, ids, skip_special_tokens=True) != row["generated_text"]
                or row["stop_reason"] != ("eos" if ids[-1] == model.config.eos_token_id else "token_cap")
                or (row["stop_reason"] == "token_cap" and len(ids) != cap)):
            raise ValueError("Saved raw generation metadata disagrees")
        probabilities = row["gate_probabilities"]
        mode = model.config.write_mode
        expected = ([0.] * budget if mode == "static" else [1.] * budget if mode == "every_loop"
                    else [float(k % model.config.periodic_interval == 0) for k in range(1, budget + 1)] if mode == "periodic"
                    else [float(p >= model.config.gate_threshold) for p in probabilities])
        if row["writes"] != expected:
            raise ValueError("Saved diagnostic write decisions disagree")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as journal:
        for index in range(len(saved), len(rows)):
            result = predict_record(model, tokenizer, records[index], rows[index], index, prefix, budget, cap)
            journal.write(json.dumps(result, ensure_ascii=False, allow_nan=False) + "\n")
            journal.flush()
            saved.append(result)
            print(f"{path.parent.name}: {index+1}/{len(rows)} correct={result['correct']}", flush=True)
    return summarize_predictions(saved, budget, records)


def diagnose(prepared_dir, run_dir, output_dir, *, device="cuda", gradient_limit=2,
             probe_limit=4, depth_grid=(1, 2, 4, 8, 16), max_new_tokens=64):
    if gradient_limit < 1 or probe_limit < 1 or max_new_tokens < 1 or not depth_grid or any(k < 1 for k in depth_grid):
        raise ValueError("Positive diagnostic limits and depth budgets required")
    if len(set(depth_grid)) != len(depth_grid):
        raise ValueError("Depth budgets must be unique")
    prepared_dir, run_dir, output_dir = (Path(p).resolve() for p in (prepared_dir, run_dir, output_dir))
    if any(output_dir == p or output_dir.is_relative_to(p) or p.is_relative_to(output_dir)
           for p in (prepared_dir, run_dir)):
        raise ValueError("Diagnostic outputs must be separate from prepared data and training checkpoints")
    device = _device(device)
    torch.set_num_threads(1)
    manifest = read_json(prepared_dir / "manifest.json")
    manifest_sha = file_sha256(prepared_dir / "manifest.json")
    checkpoint = resolve_checkpoint(run_dir)
    anchor, contract = _validate_training_identity(run_dir, checkpoint, manifest, manifest_sha)
    for name, expected in contract["runtime"]["implementation_sha256"].items():
        path = (ROOT / name).resolve()
        if not path.is_relative_to(ROOT) or file_sha256(path) != expected:
            raise ValueError(f"Implementation differs from the original training run: {name}")
    tokenizer = _validate_tokenizers(prepared_dir, checkpoint, manifest, anchor)
    train = manifest["splits"]["train"]
    train_path = (prepared_dir / train["path"]).resolve()
    if not train_path.is_relative_to(prepared_dir) or file_sha256(train_path) != train["sha256"]:
        raise ValueError("Training data checksum/path differs from manifest")
    with train_path.open(encoding="utf-8") as handle:
        records = [json.loads(line) for line in handle if line.strip()]
    if len(records) != train["records"] or len(records) != contract["record_count"]:
        raise ValueError("Training denominator differs from manifest")
    if stable_json_hash([r["sample_id"] for r in records]) != train["ordered_sample_ids_sha256"]:
        raise ValueError("Ordered training identities differ from manifest")
    if len(records) > 512:
        raise ValueError("This diagnostic is for small pilots (at most 512 training records)")
    rows = [{"sample_id": r["sample_id"], "question_hash": r["question_hash"],
             "question": decode(tokenizer, r["question_input_ids"]),
             "final_answer": decode(tokenizer, r["answer_input_ids"][:-1]).replace(",", "").strip()}
            for r in records]
    prefix = tokenizer.encode("\n### ", add_special_tokens=False)
    if any(r["latent_answer_prefix_ids"] != prefix or r["answer_input_ids"][-1] != tokenizer.eos_token_id
           for r in records):
        raise ValueError("Prepared training prefix or answer boundary differs")
    settings = contract["settings"]
    weights = LossConfig(cot=settings["cot_weight"], gate=settings["gate_weight"],
                         anchor=settings["anchor_weight"], gate_positive_weight=settings["gate_positive_weight"])
    identity = {"schema_version": "dsa_checkpoint_diagnostics_v1", "checkpoint": str(checkpoint),
                "checkpoint_files_sha256": _checkpoint_files(checkpoint),
                "prepared_manifest_sha256": manifest_sha, "train_sha256": train["sha256"],
                "script_sha256": file_sha256(Path(__file__)), "contract": contract,
                "diagnostic_helpers_sha256": {name: file_sha256(ROOT / name) for name in
                    ("scripts/evaluate_dsa.py", "scripts/smoke_pretrained_dsa.py")},
                "gradient_limit": min(gradient_limit, len(rows)), "probe_limit": min(probe_limit, len(rows)),
                "depth_grid": list(depth_grid), "max_new_tokens": max_new_tokens,
                "runtime": {"device": str(device), "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
                            "python": platform.python_version(), "torch": torch.__version__,
                            "cuda": torch.version.cuda, "cudnn": torch.backends.cudnn.version(),
                            "packages": {name: importlib.metadata.version(name) for name in
                                         ("transformers", "tokenizers", "numpy")}}}
    output_dir.mkdir(parents=True, exist_ok=True)
    identity_path = output_dir / "identity.json"
    if identity_path.exists() and read_json(identity_path) != identity:
        raise ValueError("Diagnostic identity changed; use a new output directory")
    atomic_json(identity_path, identity)
    report_path = output_dir / "report.json"
    if report_path.exists():
        report = read_json(report_path)
        if report.get("identity") != identity or not report.get("complete"):
            raise ValueError("Invalid completed diagnostic report")
        for name, expected in report["artifacts_sha256"].items():
            path = (output_dir / name).resolve()
            if not path.is_relative_to(output_dir) or file_sha256(path) != expected:
                raise ValueError("Completed diagnostic artifact checksum differs")
        return {**report, "reused_completed_diagnostic": True}
    _seed(settings["seed"], device)
    model = _load_model(checkpoint / "model").to(device)
    model.backbone.model.config.use_cache = False
    if settings.get("gradient_checkpointing") and not manifest["tiny"]:
        model.backbone.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    before = _state_digest(model)
    model.eval()
    prediction_reports = {}
    budget = anchor["loop_budget"]
    prediction_reports["normal_training_set"] = prediction_journal(model, tokenizer, records, rows,
        output_dir / "normal-training-set/predictions.jsonl", prefix, budget, max_new_tokens)
    count = identity["probe_limit"]
    baseline = _recover_predictions(output_dir / "normal-training-set/predictions.jsonl", rows, budget)[:count]
    prediction_reports[f"normal-k{budget}-first{count}"] = summarize_predictions(baseline, budget, records[:count])
    conditions = [("normal", k) for k in depth_grid if k != budget] + [("no_writes", budget), ("all_writes", budget)]
    for mode, k in conditions:
        name = f"{mode}-k{k}-first{count}"
        with inference_condition(model, mode):
            prediction_reports[name] = prediction_journal(model, tokenizer, records[:count], rows[:count],
                output_dir / name / "predictions.jsonl", prefix, k, max_new_tokens)
        condition_rows = _recover_predictions(output_dir / name / "predictions.jsonl", rows[:count], k)
        prediction_reports[name]["paired_with_normal_same_questions"] = {
            "normal_correct": sum(row["correct"] for row in baseline),
            "condition_correct": sum(row["correct"] for row in condition_rows),
            "extracted_answer_changes": sum(a["prediction"] != b["prediction"] for a, b in zip(baseline, condition_rows)),
            "correct_to_wrong": sum(a["correct"] and not b["correct"] for a, b in zip(baseline, condition_rows)),
            "wrong_to_correct": sum(not a["correct"] and b["correct"] for a, b in zip(baseline, condition_rows))}
    gradient_reports = []
    for index, record in enumerate(records[:identity["gradient_limit"]]):
        cases = [(k, False) for k in depth_grid] + [(budget, True)]
        for k, full in cases:
            name = f"sample{index:04d}-k{k}-{'joint' if full else 'answer'}.json"
            path = output_dir / "gradients" / name
            if path.exists():
                result = read_json(path)
                if result.get("identity_sha256") != stable_json_hash(identity) or result.get("sample_id") != record["sample_id"]:
                    raise ValueError("Saved gradient diagnostic identity differs")
            else:
                print(f"Gradient probe: sample {index}, K={k}, {'joint components' if full else 'answer only'}", flush=True)
                result = gradient_probe(model, record, prefix, k, weights, settings["seed"] + index,
                                        full_objective=full)
                result["identity_sha256"] = stable_json_hash(identity)
                path.parent.mkdir(parents=True, exist_ok=True)
                atomic_json(path, result)
            gradient_reports.append({"path": str(path.relative_to(output_dir)), **result})
    after = _state_digest(model)
    if before != after:
        raise RuntimeError("Diagnostic changed model state; no training checkpoint was saved")
    artifact_hashes = {str(p.relative_to(output_dir)): file_sha256(p)
                      for p in sorted(output_dir.rglob("*")) if p.is_file() and p != report_path}
    report = {"identity": identity, "complete": True, "split": "train_diagnostic",
              "training_examples": len(records), "question_only_generation": True,
              "gold_used_only_for_scoring_and_teacher_forced_gradient_probe": True,
              "training_subset_order": "prepared source order; first rows for probes",
              "test_or_validation_answers_read": False, "optimizer_updates": 0,
              "model_state_unchanged": True, "model_state_sha256": before,
              "prediction_reports": prediction_reports, "gradient_reports": gradient_reports,
              "artifacts_sha256": artifact_hashes,
              "limitations": ["Inference interventions can shift the state distribution; effects do not prove semantic reasoning.",
                              "Depth probes use the saved checkpoint, not separately trained matched-depth models.",
                              "Full objective gradients use training dropout and straight-through gates; each loss uses the same forward graph.",
                              "State similarities and teacher alignment are diagnostics, not translations of latent thoughts."]}
    atomic_json(report_path, report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("prepared-dir", "run-dir", "output-dir"):
        parser.add_argument("--" + name, required=True, type=Path)
    parser.add_argument("--device", default="cuda", choices=("auto", "cuda", "cpu", "mps"))
    parser.add_argument("--gradient-limit", type=int, default=2)
    parser.add_argument("--probe-limit", type=int, default=4)
    parser.add_argument("--depth-grid", type=int, nargs="+", default=[1, 2, 4, 8, 16])
    parser.add_argument("--max-new-tokens", type=int, default=64)
    args = parser.parse_args()
    report = diagnose(**vars(args))
    print(json.dumps({"complete": report["complete"], "optimizer_updates": report["optimizer_updates"],
                      "model_state_unchanged": report["model_state_unchanged"],
                      "predictions": report["prediction_reports"], "report": str(args.output_dir / "report.json")}, indent=2), flush=True)


if __name__ == "__main__":
    main()
