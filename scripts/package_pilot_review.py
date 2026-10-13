#!/usr/bin/env python3
"""Build one bounded review ZIP and a transparent pre-scale evidence summary."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys
from zipfile import ZipFile, ZIP_DEFLATED

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.smoke_pretrained_dsa import atomic_json


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def jsonl(path, errors=None):
    if not path.is_file(): return []
    rows = []
    with path.open(encoding="utf-8", errors="replace") as handle:
        for index, line in enumerate(handle, 1):
            if not line.strip(): continue
            try: rows.append(json.loads(line))
            except json.JSONDecodeError:
                if errors is None: raise
                errors.append({"path": str(path), "line": index, "error": "invalid/incomplete JSONL; raw bytes retained in bundle"})
    return rows


def package_review(project_dir, run_name, review_name, prepared_dir, output=None, arms=("static", "learned")):
    project_dir, prepared_dir = Path(project_dir).resolve(), Path(prepared_dir).resolve()
    if Path(run_name).name != run_name or Path(review_name).name != review_name:
        raise ValueError("Run/review names must be single path components")
    runs = project_dir / "runs" / run_name
    reports = project_dir / "reports" / run_name
    review = reports / review_name
    review.mkdir(parents=True, exist_ok=True)
    output = Path(output).resolve() if output else project_dir / "bundles" / f"{run_name}-{review_name}.zip"
    evidence = {"purpose": "pilot engineering evidence; not an automatic approval to scale or a benchmark claim",
                "run_name": run_name, "review_name": review_name, "arms": {}, "blocking": [], "questions_remaining": []}
    manifests, journals, artifact_errors = {}, {}, []
    required = []
    for arm in arms:
        if Path(arm).name != arm: raise ValueError("Invalid arm name")
        manifest = runs / arm / "run_manifest.json"
        diag = review / arm / "diagnostics" / "report.json"
        control = review / arm / "explicit-control" / "report.json"
        required += [manifest, diag, control]
        if manifest.exists(): manifests[arm] = load(manifest)
        metrics = jsonl(runs / arm / "metrics.jsonl", artifact_errors)
        journals[arm] = metrics
        arm_evidence = {"training_steps_logged": len(metrics)}
        if metrics:
            n = manifests.get(arm, {}).get("contract", {}).get("record_count", 32)
            windows = {"first_complete_pass": metrics[:n], "last_complete_window": metrics[-n:]}
            arm_evidence["loss_windows"] = {name: {key: statistics.mean(row[key] for row in rows)
                for key in ("total", "answer", "cot", "gate", "anchor", "write_rate")} for name, rows in windows.items()}
            norms = [row["gradient_norm"] for row in metrics]
            arm_evidence["training_gradient_norms_before_clip"] = {"median": statistics.median(norms), "max": max(norms)}
            seconds = sum(row["seconds"] for row in metrics)
            arm_evidence["measured_updates_per_second"] = len(metrics) / seconds if seconds else None
            arm_evidence["peak_cuda_allocated_bytes"] = max(row.get("peak_cuda_allocated_bytes") or 0 for row in metrics)
        if diag.exists():
            report = load(diag)
            arm_evidence["training_generation"] = report["prediction_reports"]["normal_training_set"]
            arm_evidence["interventions"] = {k: v for k, v in report["prediction_reports"].items() if k != "normal_training_set"}
            arm_evidence["model_state_unchanged"] = report["model_state_unchanged"]
            nonfinite = sum(loss["parameter_gradients"]["nonfinite_parameter_gradients"]
                for probe in report["gradient_reports"] for loss in probe["losses"].values())
            arm_evidence["nonfinite_parameter_gradient_observations"] = nonfinite
            if nonfinite: evidence["blocking"].append(f"{arm}: nonfinite gradient observations")
            # This is an explicit engineering heuristic, not a statistical threshold.
            if arm_evidence["training_generation"]["accuracy"] < .8:
                evidence["blocking"].append(f"{arm}: training accuracy below the declared 80% memorization diagnostic heuristic")
        if control.exists(): arm_evidence["explicit_control"] = load(control)["splits"]
        valid_reports = sorted((review / arm).glob("validation-*/report.json"))
        if not valid_reports:
            evidence["blocking"].append(f"{arm}: requested validation report missing")
            required.append(review / arm / "validation-missing" / "report.json")
        else:
            if len(valid_reports) != 1:
                raise ValueError("Review contains multiple validation settings; use separate review names")
            r = load(valid_reports[-1])
            arm_evidence["validation"] = {key: r[key] for key in ("total", "correct", "accuracy", "invalid_answer_rate", "average_latency_seconds")}
        evidence["arms"][arm] = arm_evidence
    required += [review / "original-explicit-control" / "report.json", review / "protocol.json", review / "session_status.json"]
    missing = [str(p.relative_to(project_dir)) for p in required if not p.is_file()]
    evidence["missing_required_artifacts"] = missing
    evidence["blocking"] += ["missing evidence: " + name for name in missing]
    if len(manifests) == len(arms):
        first = manifests[arms[0]]
        evidence["matched_controls"] = {}
        for arm in arms[1:]:
            other = manifests[arm]
            differences = {k: [first["contract"]["settings"][k], other["contract"]["settings"][k]]
                for k in first["contract"]["settings"] if first["contract"]["settings"][k] != other["contract"]["settings"][k]}
            actual_order_equal = (bool(journals[arms[0]]) and
                [r["sample_ids"] for r in journals[arms[0]]] == [r["sample_ids"] for r in journals[arm]])
            matched = (actual_order_equal and first["initial_state_sha256"] == other["initial_state_sha256"]
                and first["contract"]["prepared_manifest_sha256"] == other["contract"]["prepared_manifest_sha256"]
                and first["contract"]["record_order_sha256"] == other["contract"]["record_order_sha256"]
                and first["contract"]["runtime"] == other["contract"]["runtime"]
                and set(differences) <= {"write_mode"})
            evidence["matched_controls"][arm] = {"matched": matched, "logged_example_sequences_equal": actual_order_equal,
                                                  "settings_differences": differences}
            if not matched: evidence["blocking"].append(f"{arm}: matched-control identity differs")
    if all((review / arm).is_dir() for arm in ("static", "learned")):
        paired = {}
        for arm in ("static", "learned"):
            paths = sorted((review / arm).glob("validation-*/predictions.jsonl"))
            paired[arm] = jsonl(paths[-1], artifact_errors) if paths else []
        a, b = paired["static"], paired["learned"]
        if a and [r["sample_id"] for r in a] == [r["sample_id"] for r in b]:
            evidence["paired_validation"] = {"n": len(a),
                "both_correct": sum(x["correct"] and y["correct"] for x,y in zip(a,b)),
                "static_only_correct": sum(x["correct"] and not y["correct"] for x,y in zip(a,b)),
                "learned_only_correct": sum(not x["correct"] and y["correct"] for x,y in zip(a,b)),
                "both_wrong": sum(not x["correct"] and not y["correct"] for x,y in zip(a,b))}
    evidence["artifact_errors"] = artifact_errors
    if artifact_errors: evidence["blocking"].append("incomplete/corrupt journals; inspect the retained raw evidence")
    evidence["questions_remaining"] = ["Review per-loop gradient growth and competing controller loss directions before choosing a stabilization change.",
        "Intervention effects and teacher alignment do not prove that latent states encode human-readable reasoning.",
        "One small training subset and one seed cannot establish an anchoring advantage; a larger run needs predeclared data, updates, controls and repeated seeds.",
        "The held-out test remains reserved until the larger protocol is frozen."]
    evidence["next_action"] = "resolve pilot blockers before increasing training scale" if evidence["blocking"] else "review evidence and freeze a larger exploratory protocol; no automatic scale-up"
    atomic_json(review / "scale_readiness.json", evidence)
    entries = {}
    def add(path, name):
        path = Path(path)
        if not path.is_file(): return
        if path.is_symlink(): raise ValueError("Bundle source is a symlink")
        if path.stat().st_size > 50_000_000: raise ValueError(f"Review artifact unexpectedly exceeds 50 MB: {path.name}")
        if name in entries: raise ValueError("Duplicate bundle member")
        entries[name] = path
    for arm in arms:
        for name in ("run_manifest.json", "metrics.jsonl", "train_report.json", "latest.json"):
            add(runs / arm / name, f"runs/{arm}/{name}")
        pointer = runs / arm / "latest.json"
        if pointer.exists():
            relative = load(pointer)["path"]
            checkpoint = (runs / arm / relative).resolve()
            if not checkpoint.is_relative_to((runs / arm).resolve()): raise ValueError("Checkpoint escapes run")
            for name in ("COMPLETE.json", "model/anchor_config.json", "model/backbone/config.json"):
                add(checkpoint / name, f"runs/{arm}/checkpoint-metadata/{name}")
        for path in sorted((reports / arm).rglob("*")):
            if any(part == "test" or part.startswith("test-") for part in path.relative_to(reports / arm).parts): continue
            if path.suffix in {".json", ".jsonl"}: add(path, f"previous-reports/{arm}/{path.relative_to(reports / arm)}")
    for name in ("manifest.json", "selections.json", "preparation_report.json", "train.tokenized.jsonl", "valid.jsonl"):
        add(prepared_dir / name, f"prepared/{name}")
    for name in ("notebook-summary.json", "runtime.json", "unmodified-checkpoint-smoke.json"):
        add(reports / name, f"previous-reports/{name}")
    for path in sorted(review.rglob("*")):
        if any(part == "test" or part.startswith("test-") for part in path.relative_to(review).parts): continue
        if path.suffix in {".json", ".jsonl", ".log"}:
            add(path, f"review/{path.relative_to(review)}")
    for name in ("notebooks/DSA_Colab.ipynb", "scripts/diagnose_dsa_pilot.py", "scripts/pilot_explicit_controls.py", "scripts/package_pilot_review.py"):
        add(ROOT / name, f"source/{name}")
    inventory = {name: {"bytes": path.stat().st_size, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()} for name, path in entries.items()}
    if sum(v["bytes"] for v in inventory.values()) > 150_000_000: raise ValueError("Review bundle exceeds 150 MB")
    status = load(review / "session_status.json") if (review / "session_status.json").exists() else {}
    bundle_manifest = {"schema_version": "dsa_pilot_review_bundle_v1", "run_name": run_name, "review_name": review_name,
        "evidence_complete": not missing and not artifact_errors and all(v.get("status") == "passed" for v in status.values()) and bool(status),
        "scale_readiness_is_a_heuristic": True, "included": inventory,
        "excluded": ["model/optimizer weights", "Hugging Face cache", "raw full training corpus", "test questions/answers"],
        "next_action": evidence["next_action"]}
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".zip.part")
    with ZipFile(temporary, "w", ZIP_DEFLATED) as archive:
        archive.writestr("bundle_manifest.json", json.dumps(bundle_manifest, indent=2)+"\n")
        for name, path in sorted(entries.items()): archive.write(path, name)
    temporary.replace(output)
    print(json.dumps({"zip": str(output), "evidence_complete": bundle_manifest["evidence_complete"],
                      "missing_required_artifacts": missing, "next_action": evidence["next_action"]}, indent=2))
    return bundle_manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("project-dir", "run-name", "review-name", "prepared-dir"): parser.add_argument("--"+name, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--arms", nargs="+", default=["static", "learned"])
    args = vars(parser.parse_args())
    package_review(**args)
