"""Baseline 4: LOTUS GPT-2 (Fan et al., 2026), using the repository's own Lotus class.

No LOTUS GPT-2 weights are public (only Llama-3B LOTUS models are on the hub),
so ``checkpoint`` is required: a ``checkpoint_final``/``model.pt`` saved by
``scripts/run.py`` with args/gsm8k_lotus_gpt2.yaml. Setup mirrors scripts/eval.py:
three latent tokens appended to the GPT-2 tokenizer, prompt question + "\\n" +
<start> + 6x13 <latent> + <end>, 6 looped iterations, greedy, 64 new tokens.
Upstream generate() calls torch.cuda.synchronize(); on MPS/CPU that call is
redirected to the matching device synchronize only while generating.
"""
from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
import sys

import torch

from dsa.bench.data import file_sha256
from dsa.bench.models.base import BenchModel
from dsa.data.pilot_prepare import DEFAULT_BACKBONE, DEFAULT_REVISION
from dsa.devices import resolve_device, synchronize

REPOSITORY = Path(__file__).resolve().parents[3]


@contextmanager
def _device_synchronize(device: torch.device):
    if device.type == "cuda":
        yield
        return
    original = torch.cuda.synchronize
    torch.cuda.synchronize = lambda *args, **kwargs: synchronize(device)
    try:
        yield
    finally:
        torch.cuda.synchronize = original


class LOTUSGPT2(BenchModel):
    def __init__(self, *, checkpoint: str | None = None, base: str = DEFAULT_BACKBONE,
                 base_revision: str | None = DEFAULT_REVISION, looped_iterations: int = 6, c_thought: int = 13,
                 max_new_tokens: int = 64, device: str = "auto", name: str = "lotus",
                 allow_untrained: bool = False):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        if checkpoint is None and not allow_untrained:
            raise ValueError("LOTUS GPT-2 needs checkpoint=<checkpoint_final or dir with model.pt>; "
                             "no public GPT-2 LOTUS weights exist (see docs/BENCHMARK.md)")
        sys.path.insert(0, str(REPOSITORY / "scripts"))
        from lotus import Lotus

        self.name, self.device = name, resolve_device(device)
        self.looped_iterations, self.c_thought = int(looped_iterations), int(c_thought)
        self.max_new_tokens = int(max_new_tokens)
        self.tokenizer = AutoTokenizer.from_pretrained(base, revision=base_revision)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        for token in ("<|start-latent|>", "<|end-latent|>", "<|latent|>"):
            self.tokenizer.add_tokens(token)
        self.start_id, self.end_id, self.latent_id = (self.tokenizer.convert_tokens_to_ids(token) for token in
                                                      ("<|start-latent|>", "<|end-latent|>", "<|latent|>"))
        backbone = AutoModelForCausalLM.from_pretrained(base, revision=base_revision, torch_dtype=torch.float32)
        weights = None
        if checkpoint is not None:
            path = Path(checkpoint)
            path = path / "model.pt" if path.is_dir() else path
            # Upstream checkpoints are full pickled state dicts written by run.py.
            weights = torch.load(path, map_location="cpu", weights_only=False)
            rows = next(value.shape[0] for key, value in weights.items() if key.endswith("wte.weight"))
            backbone.resize_token_embeddings(rows)
        elif backbone.get_input_embeddings().weight.shape[0] < len(self.tokenizer):
            backbone.resize_token_embeddings(len(self.tokenizer))
        self.model = Lotus(backbone, latent_token_id=self.latent_id, start_latent_id=self.start_id,
                           end_latent_id=self.end_id, eos_token_id=self.tokenizer.eos_token_id,
                           pad_token_id=self.tokenizer.pad_token_id, c_thought=self.c_thought)
        if weights is not None:
            result = self.model.load_state_dict(weights, strict=False)
            missing = [key for key in result.missing_keys if not key.startswith("teacher")]
            if missing or result.unexpected_keys:
                raise ValueError(f"LOTUS checkpoint mismatch: missing={missing} unexpected={result.unexpected_keys}")
        self.model.to(self.device).to(torch.float32).eval()
        self._identity = {"checkpoint_sha256": file_sha256(path) if checkpoint else "untrained",
                          "base": base, "base_revision": base_revision,
                          "looped_iterations": self.looped_iterations, "c_thought": self.c_thought,
                          "max_new_tokens": self.max_new_tokens, "decoding": "greedy", "dtype": "float32"}

    def identity(self) -> dict:
        return dict(self._identity)

    @torch.no_grad()
    def generate(self, questions: list[str]) -> list[str]:
        latent = [self.start_id] + [self.latent_id] * (self.looped_iterations * self.c_thought) + [self.end_id]
        texts = []
        with _device_synchronize(self.device):
            for question in questions:
                prompt = self.tokenizer.encode(question.strip() + "\n", add_special_tokens=True) + latent
                inputs = torch.tensor([prompt], device=self.device)
                output = self.model.generate(input_ids=inputs, attention_mask=torch.ones_like(inputs),
                                             max_new_tokens=self.max_new_tokens,
                                             n_looped_iters=self.looped_iterations)
                texts.append(self.tokenizer.decode(output[0][len(prompt):], skip_special_tokens=True))
        return texts
