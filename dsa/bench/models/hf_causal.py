"""Explicit-route decoding for any Hugging Face causal LM (vanilla and SFT GPT-2)."""
from __future__ import annotations

from pathlib import Path

import torch

from dsa.bench.models.base import BenchModel, directory_identity
from dsa.devices import resolve_device


class HFCausalLM(BenchModel):
    """Greedy question-to-continuation decoding, batched with left padding."""

    def __init__(self, model_id: str, *, revision: str | None = None, device: str = "auto",
                 prompt: str = "{question}\n", max_new_tokens: int = 192, batch_size: int = 16,
                 name: str = "hf-causal-lm"):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        if "{question}" not in prompt:
            raise ValueError("prompt must contain {question}")
        self.name, self.prompt = name, prompt
        self.max_new_tokens, self.batch_size = int(max_new_tokens), int(batch_size)
        self.device = resolve_device(device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision, use_fast=True)
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        # Eager attention and FP32 match the earlier explicit controls on every device.
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id, revision=revision, torch_dtype=torch.float32, attn_implementation="eager",
        ).to(self.device).eval()
        local = Path(model_id).is_dir()
        self._identity = {
            "model_id": None if local else model_id,
            "revision": None if local else getattr(self.model.config, "_commit_hash", None) or revision,
            "checkpoint_sha256": directory_identity(model_id) if local else None,
            "prompt": prompt, "max_new_tokens": self.max_new_tokens,
            "decoding": "greedy", "batch_size": self.batch_size,
        }

    def identity(self) -> dict:
        return dict(self._identity)

    @torch.no_grad()
    def generate(self, questions: list[str]) -> list[str]:
        prompts = [self.prompt.format(question=question.strip()) for question in questions]
        encoded = self.tokenizer(prompts, return_tensors="pt", padding=True, add_special_tokens=False)
        encoded = {key: value.to(self.device) for key, value in encoded.items()}
        eos = self.tokenizer.eos_token_id
        output = self.model.generate(
            **encoded, do_sample=False, num_beams=1, max_new_tokens=self.max_new_tokens,
            pad_token_id=self.tokenizer.pad_token_id, eos_token_id=eos,
            temperature=None, top_p=None, top_k=None,
        )
        new_tokens = output[:, encoded["input_ids"].shape[1]:]
        return self.tokenizer.batch_decode(new_tokens, skip_special_tokens=True)
