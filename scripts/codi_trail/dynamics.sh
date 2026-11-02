#!/bin/bash
# Post hoc dynamics run (results/codi_trail/DYNAMICS_PLAN.md): validation in full, then the content test on SVAMP.
# Three shards each, merged back into dataset order. Needs the latents from probe.py. DEVICE overrides mps.
set -e
cd "$(dirname "$0")/../.."
P=results/codi_trail/probe
D=results/codi_trail/dynamics
for k in 0 1 2; do
  python scripts/codi_trail/dynamics.py --dataset gsm8k_valid --latents $P/gsm8k_valid.latents.pt --device "${DEVICE:-mps}" \
    --output $D/gsm8k_valid.shard$k.jsonl --shard $k --shards 3 &
done
wait
for k in 0 1 2; do
  python scripts/codi_trail/dynamics.py --dataset svamp --latents $P/svamp.latents.pt --device "${DEVICE:-mps}" --content-only \
    --output $D/svamp.shard$k.jsonl --shard $k --shards 3 &
done
wait
python - <<'PY'
import json
from pathlib import Path
for name in ("gsm8k_valid", "svamp"):
    order = [json.loads(line)["id"] for line in Path(f"results/codi_trail/probe/{name}.jsonl").read_text().splitlines()]
    lines = {}
    for path in sorted(Path("results/codi_trail/dynamics").glob(f"{name}.shard*.jsonl")):
        lines.update((json.loads(line)["id"], line) for line in path.read_text().splitlines())
        path.unlink()
    Path(f"results/codi_trail/dynamics/{name}.jsonl").write_text("".join(lines[i] + "\n" for i in order))
PY
