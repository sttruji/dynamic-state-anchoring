"""The only contract between a model and the evaluation harness."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


class BenchModel:
    """Subclass this to plug a new model into ``scripts/evaluate.py``.

    The harness calls ``generate`` with question strings only (never gold
    answers, rationales or milestones) and scores the returned raw text with
    one shared scorer. ``identity`` must change whenever outputs could change;
    the harness uses it to reuse finished results and to refuse mixing runs.
    """

    name = "unnamed"
    batch_size = 1

    def identity(self) -> dict:
        raise NotImplementedError

    def generate(self, questions: list[str]) -> list[str]:
        raise NotImplementedError


def directory_identity(path: str | Path) -> str:
    """Content hash of a local checkpoint, independent of where it lives on disk."""
    root = Path(path)
    entries = []
    for file in sorted(item for item in root.rglob("*") if item.is_file()):
        digest = hashlib.sha256()
        with file.open("rb") as handle:
            for block in iter(lambda: handle.read(1 << 20), b""):
                digest.update(block)
        entries.append([file.relative_to(root).as_posix(), digest.hexdigest()])
    return hashlib.sha256(json.dumps(entries, separators=(",", ":")).encode()).hexdigest()
