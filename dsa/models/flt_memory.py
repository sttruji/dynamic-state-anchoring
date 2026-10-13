"""FLT-inspired query injection for the fixed five-vector GPT-2 memory.

This is an adaptation, not a reproduction of arXiv:2605.18797v1. Question
positions retain ordinary causal self-attention. Memory queries at every block
come from the previous loop's anchor/workspace, while keys/values and the
residual stream come from a freshly rebuilt current pass. All projections and
normalizations are existing GPT-2 parameters. The teacher/readout paths are
unchanged; no states are detached and no cross-attention module is added.
"""
from functools import partial

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


def memory_block(block, hidden, previous_memory, attention_mask, question_width):
    current = block.ln_1(hidden)
    previous = block.ln_1(previous_memory)
    attention = block.attn
    query, key, value = attention.c_attn(current).split(attention.split_size, dim=2)
    # GPT-2 Conv1D stores [input, output] weights. Reuse only its Q slice;
    # projecting unused previous-memory K/V would add unnecessary work.
    memory_query = F.linear(previous, attention.c_attn.weight[:, :attention.split_size].T,
                            attention.c_attn.bias[:attention.split_size])
    query = torch.cat((query[:, :question_width], memory_query), dim=1)
    query, key, value = [attention._split_heads(x, attention.num_heads, attention.head_dim)
                         for x in (query, key, value)]
    attend = attention._upcast_and_reordered_attn if attention.reorder_and_upcast_attn else attention._attn
    output, _ = attend(query, key, value, attention_mask, None)
    output = attention._merge_heads(output, attention.num_heads, attention.head_dim)
    hidden = hidden + attention.resid_dropout(attention.c_proj(output))
    return hidden + block.mlp(block.ln_2(hidden))


def forward_memory(backbone, inputs_embeds, attention_mask, position_ids, base_inputs_embeds, question_width):
    if backbone.kind != "gpt2":
        raise ValueError("FLT-inspired memory injection currently supports GPT-2 only")
    body = backbone.model.transformer
    if body.config._attn_implementation != "eager" or body.config.add_cross_attention:
        raise ValueError("FLT memory requires the pinned eager GPT-2 self-attention implementation")
    if inputs_embeds.shape != base_inputs_embeds.shape or not 0 < question_width < inputs_embeds.shape[1]:
        raise ValueError("Current and base embeddings must have matching question-plus-memory shapes")
    # Reset the current residual stream each iteration. Previous memory is used
    # only through attention queries, never added to this residual stream.
    hidden = body.drop(base_inputs_embeds + body.wpe(position_ids))
    previous = inputs_embeds[:, question_width:]
    mask = (1.0 - attention_mask[:, None, None, :].to(hidden.dtype)) * torch.finfo(hidden.dtype).min
    for block in body.h:
        # Bind the block and pass previous memory explicitly: checkpoint
        # recomputation must never capture the next loop's mutable state.
        run_block = partial(memory_block, block, question_width=question_width)
        if body.gradient_checkpointing and body.training:
            hidden = checkpoint(run_block, hidden, previous, mask, use_reentrant=False)
        else:
            hidden = run_block(hidden, previous, mask)
    return body.ln_f(hidden)
