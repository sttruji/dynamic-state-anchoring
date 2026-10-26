#!/usr/bin/env python3
"""Download LOTUS once to persistent storage and prepare an auditable pilot."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dsa.data.pilot_prepare import DEFAULT_BACKBONE, DEFAULT_REVISION, prepare_pilot


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--storage-dir", type=Path, help="persistent corpus directory outside this repository; use mounted Drive in Colab")
    parser.add_argument("--source-dir", type=Path, help="existing gsm_train.json/gsm_valid.json/gsm_test.json arrays; unpinned inputs are labeled fixtures")
    parser.add_argument("--output-dir", type=Path, required=True, help="persistent prepared pilot directory; same identity is reused, stale caches refused")
    parser.add_argument("--train-limit", type=int, default=32, help="train-only deterministic pilot sample; 0 selects complete split")
    parser.add_argument("--valid-limit", type=int, default=0, help="0 preserves the complete validation split")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--loops", type=int, default=16)
    parser.add_argument("--policy", choices=("conclusive", "verified_only"), default="conclusive")
    parser.add_argument("--tokenizer-model", default=DEFAULT_BACKBONE)
    parser.add_argument("--revision", default=DEFAULT_REVISION)
    parser.add_argument("--fixture", choices=("overfit32",), help="reviewed TRAIN fixture instead of a seeded sample; overrides --train-limit")
    parser.add_argument("--steps-join", choices=("space", "newline"), default="space",
                        help="how LOTUS steps join in the explicit training text; newline matches the CoT checkpoint's format")
    parser.add_argument("--tiny", action="store_true", help="use download-free TinyTokenizer for implementation checks; not the GPT-2 experiment")
    args = parser.parse_args(argv)
    if args.storage_dir is None and args.source_dir is None:
        parser.error("provide --storage-dir or --source-dir")
    try:
        manifest = prepare_pilot(**vars(args))
    except (OSError, ValueError, TypeError, KeyError, ImportError) as exc:
        parser.exit(2, f"pilot preparation stopped: {exc}\n")
    print(json.dumps({"manifest": str((args.output_dir / "manifest.json").resolve()),
                      "regime": manifest["source_provenance"]["regime"],
                      "records": {split: info["records"] for split, info in manifest["splits"].items()},
                      "cache_identity_sha256": manifest["cache_identity_sha256"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
