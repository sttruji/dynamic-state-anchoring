#!/usr/bin/env python3
"""Audit LOTUS arrays and create lossless JSONL and two annotation policies."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

REPOSITORY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY))

from dsa.data.milestones import annotate_record, annotation_report
from dsa.provenance import SPLITS, build_split_manifest, file_sha256, load_lotus_array


def prepare_data(split_paths, output_dir, *, lotus_sha, preprocessing_paths, official_test_path=None):
    # Finish audit and every annotation before creating any output. Bad records
    # and split collisions therefore cannot leave a partially prepared corpus.
    manifest = build_split_manifest(split_paths, lotus_sha=lotus_sha,
                                    preprocessing_paths=preprocessing_paths,
                                    official_test_path=official_test_path)
    rows = {split: load_lotus_array(split_paths[split]) for split in SPLITS}
    annotated = {
        mode: {split: [annotate_record(row, mode=mode, steps_join="space") for row in rows[split]]
               for split in SPLITS}
        for mode in ("conclusive", "verified_only")
    }
    output = Path(output_dir)
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"{output}: output directory must be empty to preserve existing artifacts")
    output.mkdir(parents=True, exist_ok=True)
    artifacts = {}
    for split in SPLITS:
        path = output / f"{split}.original.jsonl"
        path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows[split]), encoding="utf-8")
        artifacts[path.name] = file_sha256(path)
        for mode in annotated:
            path = output / f"{split}.{mode}.jsonl"
            records = annotated[mode][split]
            path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in records), encoding="utf-8")
            artifacts[path.name] = file_sha256(path)
            report_path = output / f"{split}.{mode}.report.json"
            report_path.write_text(json.dumps(annotation_report(records), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
            artifacts[report_path.name] = file_sha256(report_path)
    manifest["derivative_sha256"] = artifacts
    manifest["annotation_code_sha256"] = file_sha256(REPOSITORY / "dsa/data/milestones.py")
    manifest["dsa_preparation_sha256"] = {
        name: file_sha256(REPOSITORY / name)
        for name in ("scripts/prepare_dsa_data.py", "dsa/provenance.py",
                     "dsa/data/adapters.py", "dsa/data/milestones.py")
    }
    manifest["tokenization_status"] = "not_run: tokenizer mapping and loop-overflow audit pending"
    (output / "split_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, default=REPOSITORY / "data/gsm_train.json")
    parser.add_argument("--valid", type=Path, default=REPOSITORY / "data/gsm_valid.json")
    parser.add_argument("--test", type=Path, default=REPOSITORY / "data/gsm_test.json")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--official-test", type=Path)
    parser.add_argument("--lotus-sha", help="Defaults to this checkout's actual HEAD")
    args = parser.parse_args(argv)
    try:
        lotus_sha = args.lotus_sha or subprocess.check_output(["git", "-C", str(REPOSITORY), "rev-parse", "HEAD"], text=True).strip()
        manifest = prepare_data({split: getattr(args, split) for split in SPLITS}, args.output_dir,
                                lotus_sha=lotus_sha,
                                preprocessing_paths={name: REPOSITORY / "preprocessing" / name
                                                     for name in ("gsm_icot.bash", "gsm_icot.py")},
                                official_test_path=args.official_test)
    except (OSError, TypeError, ValueError, subprocess.CalledProcessError) as exc:
        parser.exit(2, f"preparation stopped: {exc}\n")
    print(json.dumps({"manifest": str((args.output_dir / "split_manifest.json").resolve()),
                      "rows": {split: manifest["splits"][split]["rows"] for split in SPLITS}}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
