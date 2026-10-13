#!/usr/bin/env python3
"""Download and reproduce the pinned LOTUS corpus without deleting raw sources."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dsa.provenance import DATA_SOURCE_COMMIT, SPLITS, file_sha256, verify_pinned_download_script


def expected_payload(source_repo, split):
    blob = subprocess.check_output(["git", "-C", str(source_repo), "show",
                                    f"{DATA_SOURCE_COMMIT}:data/gsm8k/{split}.txt"])
    if blob.startswith(b"version https://git-lfs.github.com/spec/v1"):
        text = blob.decode("ascii")
        oid = re.search(r"^oid sha256:([0-9a-f]{64})$", text, re.M)
        size = re.search(r"^size (\d+)$", text, re.M)
        if oid is None or size is None:
            raise ValueError(f"Malformed pinned {split} LFS pointer")
        return {"sha256": oid.group(1), "bytes": int(size.group(1)), "verification": "pinned_git_lfs_oid"}
    return {"sha256": hashlib.sha256(blob).hexdigest(), "bytes": len(blob),
            "verification": "pinned_git_blob"}


def download_verified(url, destination, expected):
    destination = Path(destination)
    if destination.exists():
        if destination.stat().st_size != expected["bytes"] or file_sha256(destination) != expected["sha256"]:
            raise ValueError(f"Existing source differs from pinned payload: {destination}; preserved without replacement")
        return {**expected, "downloaded": False, "reused_verified_file": True}
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=f".{destination.name}.", suffix=".partial", dir=destination.parent)
    staged = Path(name)
    digest, count = hashlib.sha256(), 0
    try:
        request = urllib.request.Request(url, headers={"User-Agent": "DSA-corpus-audit/1.0"})
        with os.fdopen(descriptor, "wb") as output, urllib.request.urlopen(request, timeout=120) as response:
            if response.status != 200:
                raise ValueError(f"Unexpected HTTP status {response.status} for {url}")
            while chunk := response.read(1024 * 1024):
                count += len(chunk)
                if count > expected["bytes"]:
                    raise ValueError(f"Payload exceeds pinned byte count: {url}")
                digest.update(chunk)
                output.write(chunk)
            output.flush()
            os.fsync(output.fileno())
        if count != expected["bytes"] or digest.hexdigest() != expected["sha256"]:
            raise ValueError(f"Payload checksum/size differs from pinned source: {url}")
        os.replace(staged, destination)
    finally:
        staged.unlink(missing_ok=True)
    return {**expected, "downloaded": True, "reused_verified_file": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-repo", type=Path, default=ROOT.parent / "dsa-sources/internalize-cot")
    args = parser.parse_args()
    source = args.source_repo.resolve()
    download = verify_pinned_download_script(ROOT / "preprocessing/gsm_icot.bash")
    manifest = {"schema_version": "dsa_lotus_download_v1", "data_source_commit": DATA_SOURCE_COMMIT,
                "lotus_sha": subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip(),
                "preprocessing_sha256": {name: file_sha256(ROOT / "preprocessing" / name)
                                         for name in ("gsm_icot.bash", "gsm_icot.py")},
                "raw_sources_preserved": True, "splits": {}}
    data = ROOT / "data"
    data.mkdir(parents=True, exist_ok=True)
    for split in SPLITS:
        raw = data / f"gsm_{split}.txt"
        expected = expected_payload(source, split)
        result = download_verified(download["urls"][split], raw, expected)
        print(json.dumps({"split": split, "raw_checksum_verified": True, **result}), flush=True)
        # Reproduce the upstream conversion exactly, preserving existing JSON arrays.
        # Run in a temporary layout so even an interrupted conversion cannot alter sources.
        with tempfile.TemporaryDirectory(prefix=".lotus-convert-", dir=data) as temp:
            staging = Path(temp)
            (staging / "data").mkdir()
            os.symlink(raw, staging / "data" / raw.name)
            subprocess.run([sys.executable, str(ROOT / "preprocessing/gsm_icot.py"), split], cwd=staging, check=True)
            converted = staging / "data" / f"gsm_{split}.json"
            target = data / converted.name
            new_sha = file_sha256(converted)
            if target.exists():
                if file_sha256(target) != new_sha:
                    raise ValueError(f"Existing JSON differs from upstream conversion: {target}; preserved")
            else:
                os.replace(converted, target)
            manifest["splits"][split] = {"url": download["urls"][split], "raw_path": str(raw.relative_to(ROOT)),
                                         **result, "json_path": str(target.relative_to(ROOT)),
                                         "json_sha256": new_sha, "json_bytes": target.stat().st_size}
        print(json.dumps({"split": split, "upstream_conversion_verified": True,
                          "json_bytes": manifest["splits"][split]["json_bytes"]}), flush=True)
    path = data / "download_manifest.json"
    path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Manifest: {path}")


if __name__ == "__main__":
    main()
