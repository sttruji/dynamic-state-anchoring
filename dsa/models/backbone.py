"""Hidden-only adapters for the two explicitly supported causal backbones."""
from __future__ import annotations

import torch
from torch import nn


class BackboneAdapter(nn.Module):
    def __init__(self, model: nn.Module):
        super().__init__()
        kind = model.config.model_type
        if kind not in {"gpt2", "llama"}:
            raise ValueError(f"Unsupported backbone: {kind}; expected gpt2 or llama")
        self.model = model
        self.kind = kind
        self.hidden_size = model.get_input_embeddings().embedding_dim

    @property
    def embeddings(self):
        return self.model.get_input_embeddings()

    def forward_embeds(self, inputs_embeds, attention_mask, position_ids):
        """Full recomputation, without KV caches or vocabulary-sized logits."""
        body = self.model.transformer if self.kind == "gpt2" else self.model.model
        return body(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            use_cache=False,
            past_key_values=None,
            return_dict=True,
        ).last_hidden_state

    def logits(self, hidden):
        return self.model.get_output_embeddings()(hidden)

    def forward_memory_embeds(self, inputs_embeds, attention_mask, position_ids, base_inputs_embeds, question_width):
        from .flt_memory import forward_memory
        return forward_memory(self, inputs_embeds, attention_mask, position_ids, base_inputs_embeds, question_width)

    @classmethod
    def tiny(cls, vocab_size: int, hidden_size: int = 32, max_positions: int = 1024):
        from transformers import GPT2Config, GPT2LMHeadModel
        config = GPT2Config(
            vocab_size=vocab_size, n_embd=hidden_size, n_head=4, n_layer=2,
            n_positions=max_positions, n_ctx=max_positions,
            resid_pdrop=0.0, embd_pdrop=0.0, attn_pdrop=0.0,
            use_cache=False, bos_token_id=1, eos_token_id=1,
        )
        config._attn_implementation = "eager"
        return cls(GPT2LMHeadModel(config))


def position_ids(attention_mask: torch.Tensor) -> torch.Tensor:
    return (attention_mask.long().cumsum(dim=-1) - 1).clamp_min(0)
