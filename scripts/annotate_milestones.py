#!/usr/bin/env python3
"""Annotate JSONL using the newly reconstructed revision 4 arithmetic rules."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dsa.data.milestones import annotate_record, annotation_report, map_events_to_loops


def _reject_constant(value: str) -> None:
    raise ValueError(f"nonstandard JSON numeric constant: {value}")


def _stage(path: Path, contents: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    staged = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(contents)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        staged.unlink(missing_ok=True)
        raise
    return staged


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="source JSONL; JSON arrays must be converted explicitly")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--mode", choices=("conclusive", "verified_only"), default="conclusive")
    parser.add_argument("--steps-join", choices=("space", "newline"), default="space")
    parser.add_argument("--marker-token", default="<SA>")
    parser.add_argument("--include-terminal", dest="exclude_terminal", action="store_false",
                        help="retain a last conclusion equal to the final answer")
    parser.set_defaults(exclude_terminal=True)
    args = parser.parse_args(argv)
    destinations = [args.output] + ([args.report] if args.report else [])
    if len({path.resolve() for path in [args.input, *destinations]}) != len(destinations) + 1:
        parser.error("input, output and report must be distinct files; source files are preserved")
    try:
        original_bytes = args.input.read_bytes()
        text = original_bytes.decode("utf-8")
    except (OSError, UnicodeError) as exc:
        parser.error(str(exc))
    annotated = []
    # Unicode line/paragraph separators are valid inside JSON strings. JSONL
    # record boundaries are actual newline characters, not str.splitlines().
    for line_number, line in enumerate(io.StringIO(text), start=1):
        try:
            record = json.loads(line, parse_constant=_reject_constant)
            annotated.append(annotate_record(record, mode=args.mode, exclude_terminal=args.exclude_terminal,
                                             marker_token=args.marker_token, steps_join=args.steps_join))
        except (ValueError, TypeError, OverflowError) as exc:
            parser.error(f"{args.input}: line {line_number}: {exc}; no output files were replaced")
    report = annotation_report(annotated)
    report["input_file_sha256"] = hashlib.sha256(original_bytes).hexdigest()
    report["input_path"] = str(args.input)
    report["policy"] = {"marker_token": args.marker_token, "mode": args.mode,
                        "exclude_terminal": args.exclude_terminal, "steps_join": args.steps_join}
    output_text = "".join(json.dumps(record, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n"
                          for record in annotated)
    report_text = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n"
    staged: list[tuple[Path, Path]] = []
    try:
        # Validate the entire corpus and stage every artifact before replacing any.
        staged.append((_stage(args.output, output_text), args.output))
        if args.report:
            staged.append((_stage(args.report, report_text), args.report))
        for temporary, destination in staged:
            os.replace(temporary, destination)
    except OSError as exc:
        parser.error(f"could not write annotation artifacts: {exc}")
    finally:
        for temporary, _ in staged:
            temporary.unlink(missing_ok=True)
    print(json.dumps({"record_count": report["record_count"], "event_count": report["event_count"],
                      "no_event_rate": report["no_event_rate"], "mode": args.mode}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
