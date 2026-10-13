import copy
import inspect
import tempfile
import unittest
from dataclasses import replace

import torch
from transformers import LlamaConfig, LlamaForCausalLM

from dsa.models.backbone import BackboneAdapter
from dsa.models.anchored_loop import AnchorConfig, AnchoredLoop
from dsa.training.objectives import normalized_alignment


torch.set_num_threads(1)


class ModelTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.model = AnchoredLoop(BackboneAdapter.tiny(259), AnchorConfig(loop_budget=4, marker_token_id=2))
        self.model.eval()
        self.question = torch.tensor([[10, 11, 12]])
        self.mask = torch.ones_like(self.question)

    def test_question_inputs_immutable_and_fixed_budget(self):
        before = self.question.clone()
        output = self.model(self.question, self.mask)
        self.assertTrue(torch.equal(self.question, before))
        self.assertEqual(output.gate_logits.shape, (1, 4))
        self.assertTrue(torch.equal(output.question_embeds, self.model.backbone.embeddings(before)))

    def test_padding_and_batch_independence(self):
        first = self.model(self.question, self.mask)
        batch = torch.tensor([[10, 11, 12, 0, 0], [10, 21, 31, 41, 51]])
        mask = torch.tensor([[1, 1, 1, 0, 0], [1, 1, 1, 1, 1]])
        together = self.model(batch, mask)
        torch.testing.assert_close(first.anchor[0], together.anchor[0], atol=2e-6, rtol=2e-5)
        torch.testing.assert_close(first.workspace[0], together.workspace[0], atol=2e-6, rtol=2e-5)
        left = self.model(torch.tensor([[0, 0, 10, 11, 12]]), torch.tensor([[0, 0, 1, 1, 1]]))
        torch.testing.assert_close(first.workspace, left.workspace, atol=2e-6, rtol=2e-5)

    def test_closed_gates_preserve_anchor_exactly(self):
        with torch.no_grad():
            self.model.gate_head.weight.zero_()
            self.model.gate_head.bias.fill_(-100)
        self.model.train()
        output = self.model(self.question, self.mask)
        self.assertTrue(torch.equal(output.anchors[:, 0], output.anchors[:, -1]))
        self.assertEqual(output.writes.detach().sum().item(), 0)

    def test_open_gate_writes_candidate(self):
        with torch.no_grad():
            self.model.gate_head.weight.zero_()
            self.model.gate_head.bias.fill_(100)
        output = self.model(self.question, self.mask)
        self.assertTrue(torch.equal(output.anchor[:, 0], output.candidates[:, -1]))

    def test_disabled_learned_matches_static_control(self):
        with torch.no_grad():
            self.model.gate_head.weight.zero_()
            self.model.gate_head.bias.fill_(-100)
        static = copy.deepcopy(self.model)
        static.config = replace(static.config, write_mode="static")
        learned = self.model(self.question, self.mask)
        baseline = static(self.question, self.mask)
        self.assertTrue(torch.equal(learned.workspace, baseline.workspace))
        self.assertTrue(torch.equal(learned.anchor, baseline.anchor))

    def test_per_example_gate_is_independent(self):
        model = self.model
        class SplitGate(torch.nn.Module):
            def forward(self, x):
                return x.new_tensor([[-100.0], [100.0]])
        model.gate_head = SplitGate()
        output = model(self.question.expand(2, -1), self.mask.expand(2, -1))
        self.assertTrue(torch.equal(output.anchors[0, 0], output.anchor[0]))
        self.assertTrue(torch.equal(output.anchor[1, 0], output.candidates[1, -1]))

    def test_periodic_writes_and_matched_parameters(self):
        counts = []
        for mode in ("static", "every_loop", "periodic", "learned"):
            model = copy.deepcopy(self.model)
            model.config = replace(model.config, write_mode=mode)
            counts.append(sum(p.numel() for p in model.parameters()))
            out = model(self.question, self.mask)
            if mode == "periodic":
                self.assertEqual(out.writes.tolist(), [[0.0, 0.0, 0.0, 1.0]])
            if mode == "every_loop":
                self.assertEqual(out.writes.tolist(), [[1.0] * 4])
        self.assertEqual(len(set(counts)), 1)

    def test_straight_through_gradient_from_continuation(self):
        self.model.train()
        with torch.no_grad():
            self.model.gate_head.bias.fill_(-1)
        output = self.model(self.question, self.mask)
        output.anchor.square().mean().backward()
        self.assertGreater(self.model.gate_head.weight.grad.norm().item(), 0)
        self.assertTrue(set(output.writes.detach().flatten().tolist()) <= {0.0, 1.0})

    def test_alignment_detaches_targets(self):
        candidates = torch.randn(2, 32, requires_grad=True)
        targets = torch.randn(2, 32, requires_grad=True)
        normalized_alignment(candidates, targets).backward()
        self.assertIsNone(targets.grad)
        self.assertGreater(candidates.grad.norm().item(), 0)

    def test_recurrence_does_not_project_logits_or_cache(self):
        calls = []
        body = self.model.backbone.model.transformer
        hook = body.register_forward_pre_hook(lambda module, args, kwargs: calls.append(kwargs), with_kwargs=True)
        head_calls = []
        head_hook = self.model.backbone.model.lm_head.register_forward_hook(lambda *args: head_calls.append(True))
        self.model(self.question, self.mask)
        hook.remove()
        head_hook.remove()
        self.assertEqual(len(calls), 4)
        self.assertEqual(head_calls, [])
        self.assertTrue(all(call["use_cache"] is False and call["past_key_values"] is None for call in calls))

    def test_question_only_api_and_marker_rejection(self):
        parameters = set(inspect.signature(self.model.forward).parameters)
        self.assertEqual(parameters, {"question_input_ids", "attention_mask", "loop_budget"})
        with self.assertRaises(ValueError):
            self.model(torch.tensor([[2]]), torch.tensor([[1]]))
        with self.assertRaises(TypeError):
            self.model(self.question, self.mask, rationale="gold")

    def test_generation_rebuilds_final_memory_and_has_fixed_cap(self):
        generated = self.model.generate(self.question, self.mask, answer_prefix_ids=[15, 16], max_new_tokens=3)
        self.assertLessEqual(generated.token_ids.shape[1], 3)
        self.assertEqual(generated.loop.candidates.shape[1], 4)
        changed = copy.deepcopy(generated.loop)
        changed.anchor = changed.anchor + torch.randn_like(changed.anchor)
        suffix = torch.tensor([[15, 16]])
        original_logits = self.model.answer_logits(generated.loop, suffix, torch.ones_like(suffix))
        changed_logits = self.model.answer_logits(changed, suffix, torch.ones_like(suffix))
        self.assertFalse(torch.allclose(original_logits, changed_logits))

    def test_llama_adapter_padding(self):
        config = LlamaConfig(vocab_size=259, hidden_size=32, intermediate_size=64,
                             num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
                             attention_dropout=0.0, max_position_embeddings=128)
        config._attn_implementation = "eager"
        model = AnchoredLoop(BackboneAdapter(LlamaForCausalLM(config)), AnchorConfig(loop_budget=2))
        model.eval()
        unpadded = model(self.question, self.mask)
        padded = model(torch.tensor([[10, 11, 12, 0, 0]]), torch.tensor([[1, 1, 1, 0, 0]]))
        torch.testing.assert_close(unpadded.workspace, padded.workspace, atol=2e-6, rtol=2e-5)


if __name__ == "__main__":
    unittest.main()
