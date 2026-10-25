"""Baseline 2: GPT-2 supervised fine-tuned on explicit GSM8K-Aug chain of thought.

Default weights are the pinned LOTUS CoT checkpoint that DSA initializes from
(it scored 25/64 on dev64 in docs/FINDINGS.md). Pass ``checkpoint=<dir>`` to
evaluate a model produced by ``scripts/train_sft.py`` instead.
"""
from dsa.bench.models.hf_causal import HFCausalLM
from dsa.data.pilot_prepare import DEFAULT_BACKBONE, DEFAULT_REVISION


class SFTGPT2(HFCausalLM):
    def __init__(self, *, checkpoint: str | None = None, model_id: str = DEFAULT_BACKBONE,
                 revision: str = DEFAULT_REVISION, **kwargs):
        kwargs.setdefault("name", "gpt2-sft" if checkpoint is None else "gpt2-sft-local")
        if checkpoint is not None:
            model_id, revision = checkpoint, None
        super().__init__(model_id, revision=revision, **kwargs)
