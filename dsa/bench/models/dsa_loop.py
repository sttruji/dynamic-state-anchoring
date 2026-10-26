"""DSA (AnchoredLoop) checkpoints and their write-policy ablations.

Accepts a saved model directory, a ``checkpoint-NNNNNNNN`` directory or a run
directory with ``latest.json``. ``loops`` overrides the trained budget for the
proposal's 2x/4x robustness checks; ``write_mode`` swaps the inference policy
(static, every_loop, periodic, learned) without retraining. ``route="explicit"``
decodes the trained backbone's ordinary CoT route instead (question + newline,
greedy, 192 tokens), the explicit-retention check from docs/FINDINGS.md.
"""
from __future__ import annotations

from dataclasses import asdict, replace
import json
from pathlib import Path

import torch

from dsa.bench.models.base import BenchModel, directory_identity
from dsa.devices import resolve_device
from dsa.evaluation.evaluate import predict_question


def resolve_model_directory(path: str | Path) -> Path:
    path = Path(path).resolve()
    if (path / "latest.json").is_file():
        latest = json.loads((path / "latest.json").read_text(encoding="utf-8"))
        path = (path / latest["path"]).resolve()
    if (path / "model" / "anchor_config.json").is_file():
        path = path / "model"
    if not (path / "anchor_config.json").is_file():
        raise ValueError(f"{path} is not a DSA model, checkpoint or run directory")
    return path


class DSAModel(BenchModel):
    def __init__(self, *, checkpoint: str, loops: int | None = None, write_mode: str | None = None,
                 max_new_tokens: int | None = None, device: str = "auto", name: str | None = None,
                 route: str = "latent"):
        from transformers import AutoModelForCausalLM

        from dsa.models.anchored_loop import AnchorConfig, AnchoredLoop
        from dsa.models.backbone import BackboneAdapter

        if route not in {"latent", "explicit"}:
            raise ValueError("route must be latent or explicit")
        directory = resolve_model_directory(checkpoint)
        config = AnchorConfig(**json.loads((directory / "anchor_config.json").read_text(encoding="utf-8")))
        trained = asdict(config)
        self.route = route
        if route == "explicit":
            if loops is not None or write_mode is not None:
                raise ValueError("loops and write_mode apply to the latent route only")
            if (directory / "tokenizer" / "tiny_tokenizer.json").is_file():
                raise ValueError("the explicit route needs a Hugging Face tokenizer")
            from dsa.bench.models.hf_causal import HFCausalLM
            self.explicit = HFCausalLM(str(directory / "backbone"), tokenizer_id=str(directory / "tokenizer"),
                                       device=device, max_new_tokens=192 if max_new_tokens is None else max_new_tokens)
            self.name = name or f"dsa-{config.recurrence_mode}-{config.write_mode}-explicit"
            self.batch_size = self.explicit.batch_size
            self._identity = {"route": "explicit", "checkpoint_sha256": directory_identity(directory),
                              "trained_config": trained, "explicit": self.explicit.identity()}
            return
        max_new_tokens = 64 if max_new_tokens is None else max_new_tokens
        if write_mode is not None:
            config = replace(config, write_mode=write_mode)
        self.loops = int(loops) if loops is not None else config.loop_budget
        self.max_new_tokens = int(max_new_tokens)
        self.device = resolve_device(device)
        backbone = BackboneAdapter(AutoModelForCausalLM.from_pretrained(
            directory / "backbone", attn_implementation="eager", torch_dtype=torch.float32,
            local_files_only=True))
        self.model = AnchoredLoop(backbone, config)
        memory = torch.load(directory / "anchor.pt", map_location="cpu", weights_only=True)
        missing, unexpected = self.model.load_state_dict(memory, strict=False)
        if unexpected or any(not key.startswith("backbone.") for key in missing):
            raise ValueError("incomplete or incompatible anchor checkpoint")
        self.model.to(self.device).eval()
        tokenizer_dir = directory / "tokenizer"
        if (tokenizer_dir / "tiny_tokenizer.json").is_file():
            from dsa.data.tokenize import TinyTokenizer
            self.tokenizer = TinyTokenizer.from_pretrained(tokenizer_dir)
        else:
            from transformers import AutoTokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir, use_fast=True, local_files_only=True)
        self.name = name or f"dsa-{config.recurrence_mode}-{config.write_mode}-k{self.loops}"
        self._identity = {"checkpoint_sha256": directory_identity(directory), "trained_config": trained,
                          "write_mode": config.write_mode, "loops": self.loops,
                          "max_new_tokens": self.max_new_tokens, "decoding": "greedy"}

    def identity(self) -> dict:
        return dict(self._identity)

    def generate(self, questions: list[str]) -> list[str]:
        if self.route == "explicit":
            return self.explicit.generate(questions)
        return [predict_question(self.model, self.tokenizer, question, loop_budget=self.loops,
                                 max_new_tokens=self.max_new_tokens)["generated_text"]
                for question in questions]
