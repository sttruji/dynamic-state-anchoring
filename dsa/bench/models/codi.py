"""Baseline 3: CODI GPT-2 (Shen et al., 2025), the published zen-E/CODI-gpt2 weights.

A dependency-free port of CODI's test.py inference (commit 2c2314662c63):
question + <bot>, six projected latent steps fed back through the KV cache,
then <eot> and greedy decoding with only <eot> masked. The LoRA adapters
(r=128, alpha=32 on c_attn/c_proj/c_fc) are merged into the GPT-2 weights at
load time, so peft is not needed. Upstream ran bf16 on CUDA; this runs FP32 on
every device by default, so scores can differ slightly from the paper.
"""
from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

from dsa.bench.models.base import BenchModel
from dsa.bench.models.gpt2_vanilla import GPT2_REVISION
from dsa.devices import resolve_device

CODI_REPO = "zen-E/CODI-gpt2"
CODI_REVISION = "fd641b3d3edc59e4f534b55588e906588c9e36bb"
LORA_ALPHA = 32
_PREFIX = "codi.base_model.model."


def merge_codi_state(state: dict[str, torch.Tensor], lora_alpha: int = LORA_ALPHA) -> tuple[dict, dict]:
    """Split a CODI checkpoint into merged GPT-2 weights and projection weights."""
    backbone, projection = {}, {}
    for key, value in state.items():
        if key.startswith("prj."):
            projection[key.removeprefix("prj.")] = value.float()
        elif key.startswith(_PREFIX) and ".lora_" not in key:
            name = key.removeprefix(_PREFIX).replace(".base_layer.", ".")
            if name != "lm_head.weight":  # GPT-2 ties the head to wte, as CODI does after loading.
                backbone[name] = value.float()
    for key in state:
        if key.endswith(".lora_A.default.weight"):
            module = key.removeprefix(_PREFIX).removesuffix(".lora_A.default.weight")
            lora_a = state[key].float()
            lora_b = state[key.replace(".lora_A.", ".lora_B.")].float()
            # Conv1D stores [in, out]; LoRA computes x @ A.T @ B.T, i.e. (B @ A).T.
            name = f"{module}.weight"  # Out of place: .float() aliases tensors that are already FP32.
            backbone[name] = backbone[name] + (lora_alpha / lora_a.shape[0]) * (lora_b @ lora_a).T
    return backbone, projection


class CODIGPT2(BenchModel):
    def __init__(self, *, checkpoint: str | None = None, latent_iterations: int = 6, max_new_tokens: int = 256,
                 device: str = "auto", name: str = "codi"):
        from huggingface_hub import hf_hub_download
        from transformers import GPT2Config, GPT2LMHeadModel, GPT2Tokenizer

        self.name, self.device = name, resolve_device(device)
        self.latent_iterations, self.max_new_tokens = int(latent_iterations), int(max_new_tokens)
        path = (Path(checkpoint) / "pytorch_model.bin" if checkpoint else
                Path(hf_hub_download(CODI_REPO, "pytorch_model.bin", revision=CODI_REVISION)))
        backbone, projection = merge_codi_state(torch.load(path, map_location="cpu", weights_only=True))

        self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2", revision=GPT2_REVISION)
        config = GPT2Config.from_pretrained("gpt2", revision=GPT2_REVISION)
        self.vocab_size = backbone["transformer.wte.weight"].shape[0]
        config.vocab_size = self.vocab_size
        config._attn_implementation = "eager"
        self.model = GPT2LMHeadModel(config)
        missing, unexpected = self.model.load_state_dict(backbone, strict=False)
        if unexpected or set(missing) - {"lm_head.weight"}:
            raise ValueError(f"CODI checkpoint does not match GPT-2: missing={missing} unexpected={unexpected}")
        self.model.tie_weights()
        width = config.n_embd
        self.projection = nn.Sequential(nn.Dropout(0.0), nn.Linear(width, width), nn.GELU(), nn.Linear(width, width))
        self.projection.add_module("ln", nn.LayerNorm(width))
        self.projection.load_state_dict(projection)
        self.model.to(self.device).eval()
        self.projection.to(self.device).eval()
        original = self.vocab_size - 3  # CODI appends pad, <bot>, <eot> after the GPT-2 vocabulary.
        self.bot_id, self.eot_id = original + 1, original + 2
        self._identity = {"weights": None if checkpoint else f"{CODI_REPO}@{CODI_REVISION}",
                          "checkpoint": str(path) if checkpoint else None, "lora_merged_fp32": True,
                          "latent_iterations": self.latent_iterations, "max_new_tokens": self.max_new_tokens,
                          "decoding": "greedy", "dtype": "float32"}

    def identity(self) -> dict:
        return dict(self._identity)

    @torch.no_grad()
    def _answer(self, question: str) -> str:
        ids = self.tokenizer.encode(question.strip().replace("  ", " ")) + [self.bot_id]
        output = self.model(input_ids=torch.tensor([ids], device=self.device), use_cache=True,
                            output_hidden_states=True)
        cache = output.past_key_values
        latent = self.projection(output.hidden_states[-1][:, -1:, :])
        for _ in range(self.latent_iterations):
            output = self.model(inputs_embeds=latent, past_key_values=cache, use_cache=True,
                                output_hidden_states=True)
            cache = output.past_key_values
            latent = self.projection(output.hidden_states[-1][:, -1:, :])
        embed = self.model.get_input_embeddings()
        step = embed(torch.tensor([[self.eot_id]], device=self.device))
        tokens = []
        for _ in range(self.max_new_tokens):
            output = self.model(inputs_embeds=step, past_key_values=cache, use_cache=True)
            cache = output.past_key_values
            token = int(output.logits[0, -1, :self.vocab_size - 1].argmax())
            tokens.append(token)
            if token == self.tokenizer.eos_token_id:
                break
            step = embed(torch.tensor([[token]], device=self.device))
        return self.tokenizer.decode([t for t in tokens if t < len(self.tokenizer)], skip_special_tokens=True)

    def generate(self, questions: list[str]) -> list[str]:
        return [self._answer(question) for question in questions]
