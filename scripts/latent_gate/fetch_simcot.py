"""Download SIM-CoT's CODI GPT-2 at a pinned revision and write it in the layout CODIGPT2(checkpoint=...) loads.

The checkpoint holds the CODI model and projection plus SIM-CoT's training-time step decoder (`decoder.*`), which inference
does not use and which is dropped here.

    python scripts/latent_gate/fetch_simcot.py --output checkpoints/simcot-gpt2
"""
import argparse
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

REPO, REVISION = "internlm/SIM_COT-GPT2-CODI", "8685b42870510ba33f7caae30f7a24a2020d2e50"
SHA256 = "9eaa159155d1db8e8fa3024b959b1e7640cbd5f78b738a3759b2aa61b80b409d"

parser = argparse.ArgumentParser()
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
target = args.output / "pytorch_model.bin"
if target.exists():
    print(f"already there: {target}")
    raise SystemExit
path = Path(hf_hub_download(REPO, "model-00001-of-00001.safetensors", revision=REVISION))
import hashlib
digest = hashlib.sha256(path.read_bytes()).hexdigest()
assert digest == SHA256, f"unexpected checkpoint {digest}"
state = {k: v for k, v in load_file(str(path)).items() if not k.startswith("decoder.")}
args.output.mkdir(parents=True, exist_ok=True)
torch.save(state, target)
print(f"{REPO}@{REVISION[:7]}: {len(state)} tensors -> {target}")
