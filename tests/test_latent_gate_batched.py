"""Checks E2 and E3 of results/latent_gate/heldout_scale/PLAN.md on tiny random models: the hard gate's training cut agrees
with inference eviction, the LLaMA LoRA merge, the latent bias's equalities, and batched loops agreeing with single rows."""
import math
import random
import sys
import unittest
from pathlib import Path

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts/latent_gate"))

from batched import POLICIES, Loop, ask, forward, run  # noqa: E402
from dsa.bench.models.codi_llama import merge_llama_state  # noqa: E402
from gate import Gate, install_bias, top_visible  # noqa: E402

WIDTH, VOCAB = 32, 40


def streamed(scores: list[float], budget: int) -> list[int]:
    """stage1_eval's rule: after each new entry, above `budget` drop the lowest score, a tie dropping the newest."""
    kept = []
    for j, s in enumerate(scores):
        kept.append(j)
        if len(kept) > budget:
            values = torch.tensor([scores[i] for i in kept])
            del kept[len(values) - 1 - int(values.flip(0).argmin())]
    return kept


def tiny(llama: bool) -> Loop:
    torch.manual_seed(0)
    if llama:
        from transformers import LlamaConfig, LlamaForCausalLM

        config = LlamaConfig(hidden_size=WIDTH, intermediate_size=64, num_hidden_layers=2, num_attention_heads=4,
                             num_key_value_heads=2, head_dim=8, vocab_size=VOCAB, max_position_embeddings=512,
                             rope_theta=500000.0, tie_word_embeddings=True,
                             rope_scaling={"factor": 32.0, "high_freq_factor": 4.0, "low_freq_factor": 1.0,
                                           "original_max_position_embeddings": 64, "rope_type": "llama3"})
        config._attn_implementation = "eager"
        model = LlamaForCausalLM(config).eval()
    else:
        from transformers import GPT2Config, GPT2LMHeadModel

        config = GPT2Config(n_embd=WIDTH, n_layer=2, n_head=4, vocab_size=VOCAB, n_positions=256)
        config._attn_implementation = "eager"
        model = GPT2LMHeadModel(config).eval()
        install_bias(model)
    projection = nn.Sequential(nn.Linear(WIDTH, WIDTH), nn.GELU(), nn.Linear(WIDTH, WIDTH), nn.LayerNorm(WIDTH)).eval()
    gate = Gate(WIDTH).eval()
    nn.init.normal_(gate.net[2].weight)
    return Loop(model, projection, None, bot_id=VOCAB - 2, eot_id=VOCAB - 1, pad_id=0, eos_id=5, vocab_size=VOCAB,
                llama=llama, max_new_tokens=4, gate=gate)


QUESTIONS = [[3, 9, 11, 7, 38], [4, 12, 38], [8, 8, 15, 21, 30, 2, 38]]


class TopVisible(unittest.TestCase):
    def test_training_cut_matches_inference_eviction(self):
        rng = random.Random(0)
        for dtype in (torch.float32, torch.bfloat16):
            for budget in (1, 3, 6):
                for _ in range(300):
                    scores = torch.tensor([rng.choice([-1.0, 0.0, 0.5, 2.0]) + rng.choice([0.0, 1e-3])
                                           for _ in range(rng.randint(1, 14))]).to(dtype).float()
                    for t in range(1, len(scores) + 1):
                        cut = top_visible(scores[:t][None], budget)[0]
                        self.assertEqual(cut.nonzero().flatten().tolist(), streamed(scores[:t].tolist(), budget))
                        self.assertEqual(int(cut.sum()), min(t, budget))


class LlamaMerge(unittest.TestCase):
    def test_lora_is_merged_as_alpha_over_r_times_b_a(self):
        torch.manual_seed(0)
        base, a, b = torch.randn(8, 4).bfloat16(), torch.randn(2, 4), torch.randn(8, 2)
        embed = torch.randn(10, 4).bfloat16()
        prefix = "codi.base_model.model."
        state = {f"{prefix}model.layers.0.self_attn.q_proj.base_layer.weight": base,
                 f"{prefix}model.layers.0.self_attn.q_proj.lora_A.default.weight": a,
                 f"{prefix}model.layers.0.self_attn.q_proj.lora_B.default.weight": b,
                 f"{prefix}model.embed_tokens.weight": embed, f"{prefix}lm_head.weight": embed.clone(),
                 "prj.1.weight": torch.randn(4, 4)}
        backbone, projection = merge_llama_state(state, lora_alpha=32)
        expected = base.float() + 16 * (b @ a)
        self.assertTrue(torch.equal(backbone["model.layers.0.self_attn.q_proj.weight"], expected))
        self.assertNotIn("lm_head.weight", backbone)
        self.assertEqual(set(projection), {"1.weight"})
        state[f"{prefix}lm_head.weight"] = torch.randn(10, 4).bfloat16()
        with self.assertRaises(ValueError):
            merge_llama_state(state)


class Bias(unittest.TestCase):
    def latents(self, loop, steps=3):
        asked = ask(loop, QUESTIONS, "cpu")
        cache, latent = asked.cache, asked.first
        batch, question = asked.mask.shape
        for step in range(1, steps + 1):
            mask = torch.cat([asked.mask, torch.ones(batch, step, dtype=asked.mask.dtype)], 1)
            out = forward(loop, embeds=latent, cache=cache, mask=mask, positions=(asked.bot + step)[:, None], hidden=True)
            cache, latent = out.past_key_values, loop.projection(out.hidden_states[-1][:, -1:, :])
        return asked, cache, question

    def next_logits(self, loop, asked, cache, n, bias=None, position=None):
        batch, question = asked.mask.shape
        mask = torch.cat([asked.mask, torch.ones(batch, n + 1, dtype=asked.mask.dtype)], 1)
        step = loop.model.get_input_embeddings()(torch.full((batch, 1), loop.eot_id))
        where = asked.bot + (n if position is None else position) + 1
        return forward(loop, embeds=step, cache=cache, mask=mask, positions=where[:, None], bias=bias).logits[:, -1]

    @torch.no_grad()
    def test_zero_bias_is_the_plain_mask(self):
        loop = tiny(llama=True)
        asked, cache, question = self.latents(loop)
        batch = asked.mask.shape[0]
        zero = torch.zeros(batch, question + 4)
        mask = torch.cat([asked.mask, torch.ones(batch, 4, dtype=asked.mask.dtype)], 1)
        step = loop.model.get_input_embeddings()(torch.full((batch, 1), loop.eot_id))
        plain = loop.model(inputs_embeds=step, past_key_values=cache, attention_mask=mask,
                           position_ids=(asked.bot + 4)[:, None], use_cache=True).logits[:, -1]
        self.assertTrue(torch.equal(self.next_logits(loop, asked, cache, 3, zero), plain))

    @torch.no_grad()
    def test_k_copies_at_minus_log_k_are_one_entry(self):
        for llama in (True, False):
            loop = tiny(llama)
            asked, cache, question = self.latents(loop)
            batch, k, column = asked.mask.shape[0], 4, question + 1  # latent entry 2
            original = self.next_logits(loop, asked, cache, 3)
            copied = tuple((torch.cat([key] + [key[:, :, column:column + 1]] * (k - 1), 2),
                            torch.cat([value] + [value[:, :, column:column + 1]] * (k - 1), 2)) for key, value in cache)
            bias = torch.zeros(batch, question + 3 + k - 1 + 1)
            bias[:, [column] + list(range(question + 3, question + 3 + k - 1))] = -math.log(k)
            again = self.next_logits(loop, asked, copied, 3 + k - 1, bias, position=3)  # same query position: RoPE
            self.assertLess(float((again - original).abs().max()), 1e-5, f"llama={llama}")


class Batched(unittest.TestCase):
    @torch.no_grad()
    def test_a_batch_matches_its_rows_one_at_a_time(self):
        for llama in (True, False):
            loop = tiny(llama)
            together = ask(loop, QUESTIONS, "cpu")
            alone = [ask(loop, [q], "cpu") for q in QUESTIONS]
            for policy in POLICIES:
                batch = run(loop, policy, together, [2, 4], 2, [0, 1, 2], attention=True)
                for row, single in enumerate(alone):
                    one = run(loop, policy, single, [2, 4], 2, [row], attention=True)[0]
                    self.assertEqual(sorted(one), sorted(batch[row]), f"{policy} llama={llama}")
                    for depth in one:
                        self.assertEqual(one[depth]["tokens"], batch[row][depth]["tokens"], f"{policy}@{depth}")
                        self.assertEqual(one[depth]["kept"], batch[row][depth]["kept"], f"{policy}@{depth}")
                        for part in ("latent_1_6", "latent_later"):
                            for x, y in zip(one[depth]["attn"][part], batch[row][depth]["attn"][part]):
                                self.assertAlmostEqual(x, y, places=4)

    @torch.no_grad()
    def test_policies_read_out_where_the_plan_says(self):
        loop = tiny(llama=True)
        asked = ask(loop, QUESTIONS[:1], "cpu")
        depths = {policy: sorted(run(loop, policy, asked, [2, 4, 6], 2, [0])[0], key=int) for policy in POLICIES}
        self.assertEqual(depths["plain"], ["2", "4", "6"])
        self.assertEqual(depths["shift"], ["2"])
        for policy in ("keep_first", "window", "random", "logn", "dedup_soft", "copy_last", "copy_pair", "gate", "gate_soft"):
            self.assertEqual(depths[policy], ["4", "6"], policy)
        kept = {policy: run(loop, policy, asked, [6], 2, [0])[0]["6"]["kept"] for policy in ("keep_first", "window")}
        self.assertEqual(kept, {"keep_first": [1, 2], "window": [5, 6]})


if __name__ == "__main__":
    unittest.main()
