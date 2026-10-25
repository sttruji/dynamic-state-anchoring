"""Baseline 1: pretrained GPT-2 small, no fine-tuning.

Uses the same prompt and decoding as the SFT baseline so the only difference is
the weights. Expect near-zero accuracy; this row is a floor and a pipeline check.
"""
from dsa.bench.models.hf_causal import HFCausalLM

GPT2_REVISION = "607a30d783dfa663caf39e06633721c8d4cfcd7e"


class VanillaGPT2(HFCausalLM):
    def __init__(self, *, model_id: str = "gpt2", revision: str = GPT2_REVISION, **kwargs):
        kwargs.setdefault("name", "gpt2-vanilla")
        super().__init__(model_id, revision=revision, **kwargs)
