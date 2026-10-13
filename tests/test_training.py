import copy
from dataclasses import replace
import tempfile
from pathlib import Path
import unittest

import torch

from dsa.data.milestones import annotate_record
from dsa.data.tokenize import TinyTokenizer, tokenize_record
from dsa.models.backbone import BackboneAdapter
from dsa.models.backbone import position_ids
from dsa.models.anchored_loop import AnchorConfig, AnchoredLoop
from dsa.training.objectives import joint_objective, LossConfig
from dsa.training.trainer import seed_everything, train_steps, save_training_state, load_training_state


torch.set_num_threads(1)


class TrainingTests(unittest.TestCase):
    def setUp(self):
        seed_everything(13)
        self.tokenizer = TinyTokenizer()
        self.prefix = self.tokenizer.encode("\n### ")
        self.raw = {"question": "What is twice the sum of 2 and 3?",
                    "answer": "<<2+3=5>>\n<<5*2=10>>\n#### 10"}
        self.record = tokenize_record(annotate_record(self.raw), self.tokenizer, 4)
        self.model = AnchoredLoop(BackboneAdapter.tiny(len(self.tokenizer)), AnchorConfig(loop_budget=4, marker_token_id=2))

    def test_all_controls_receive_candidate_gate_and_explicit_supervision(self):
        for mode in ("static", "every_loop", "periodic", "learned"):
            model = copy.deepcopy(self.model)
            model.config = replace(model.config, write_mode=mode)
            losses = joint_objective(model, self.record, self.prefix)
            losses["total"].backward()
            self.assertGreater(losses["gate"].item(), 0)
            self.assertGreater(losses["anchor"].item(), 0)
            self.assertGreater(model.anchor_projection.weight.grad.norm().item(), 0)
            self.assertGreater(model.gate_head.weight.grad.norm().item(), 0)
            self.assertGreater(model.backbone.embeddings.weight.grad[2].norm().item(), 0)

    def test_no_event_retains_answer_and_explicit_losses(self):
        raw = {"question": "How many?", "answer": "Unannotated reasoning.\n#### 10"}
        record = tokenize_record(annotate_record(raw), self.tokenizer, 4)
        losses = joint_objective(self.model, record, self.prefix)
        self.assertEqual(losses["gate"].item(), 0)
        self.assertEqual(losses["anchor"].item(), 0)
        self.assertGreater(losses["answer"].item(), 0)
        self.assertGreater(losses["cot"].item(), 0)
        losses["total"].backward()
        self.assertGreater(self.model.backbone.embeddings.weight.grad.norm().item(), 0)

    def test_overflow_keeps_answer_supervision(self):
        raw = {"question": "How many?", "answer": "<<1+1=2>>\n<<2+1=3>>\n<<3+1=4>>\n#### 5"}
        record = tokenize_record(annotate_record(raw), self.tokenizer, 2)
        model = copy.deepcopy(self.model)
        model.config = replace(model.config, loop_budget=2)
        losses = joint_objective(model, record, self.prefix)
        self.assertFalse(record["controller_supervised"])
        self.assertEqual(losses["gate"].item(), 0)
        self.assertGreater(losses["answer"].item(), 0)

    def test_gold_changes_do_not_change_question_inference(self):
        changed = {"question": self.raw["question"], "answer": "<<8+8=16>>\n#### 500"}
        second = tokenize_record(annotate_record(changed), self.tokenizer, 4)
        self.assertEqual(self.record["question_input_ids"], second["question_input_ids"])
        self.model.eval()
        inputs = torch.tensor([self.record["question_input_ids"]])
        first = self.model.generate(inputs, torch.ones_like(inputs), answer_prefix_ids=self.prefix, max_new_tokens=3)
        other = torch.tensor([second["question_input_ids"]])
        last = self.model.generate(other, torch.ones_like(other), answer_prefix_ids=self.prefix, max_new_tokens=3)
        self.assertTrue(torch.equal(first.token_ids, last.token_ids))

    def test_checkpoint_targets_cannot_see_future_rationale_or_answer(self):
        self.model.eval()
        tokens = torch.tensor([self.record["explicit_input_ids"]])
        checkpoint = self.record["checkpoint_positions"][0]
        altered = tokens.clone()
        altered[:, checkpoint + 1:] = 25
        mask = torch.ones_like(tokens)
        backbone = self.model.backbone
        original = backbone.forward_embeds(backbone.embeddings(tokens), mask, position_ids(mask))
        changed = backbone.forward_embeds(backbone.embeddings(altered), mask, position_ids(mask))
        self.assertTrue(torch.equal(original[:, :checkpoint + 1], changed[:, :checkpoint + 1]))

    def test_checkpoint_preserves_tying_tokenizer_and_generation(self):
        self.model.eval()
        inputs = torch.tensor([self.record["question_input_ids"]])
        original = self.model.generate(inputs, torch.ones_like(inputs), answer_prefix_ids=self.prefix, max_new_tokens=3)
        with tempfile.TemporaryDirectory() as temp:
            self.model.save_pretrained(temp, self.tokenizer)
            restored = AnchoredLoop.from_pretrained(temp).eval()
            tokenizer = TinyTokenizer.from_pretrained(Path(temp) / "tokenizer")
            self.assertEqual(tokenizer.encode("é<SA>"), self.tokenizer.encode("é<SA>"))
            self.assertIs(restored.backbone.model.lm_head.weight, restored.backbone.embeddings.weight)
            generated = restored.generate(inputs, torch.ones_like(inputs), answer_prefix_ids=self.prefix, max_new_tokens=3)
            self.assertTrue(torch.equal(original.token_ids, generated.token_ids))

    def test_resume_reproduces_next_update(self):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=0.001)
        train_steps(self.model, [self.record], self.prefix, optimizer, 1)
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "resume.pt"
            save_training_state(path, self.model, optimizer, 1, records=[self.record])
            expected_metrics = train_steps(self.model, [self.record], self.prefix, optimizer, 1, start_step=1)
            resumed = copy.deepcopy(self.model)
            resumed_optimizer = torch.optim.AdamW(resumed.parameters(), lr=0.001)
            step = load_training_state(path, resumed, resumed_optimizer, records=[self.record])
            actual_metrics = train_steps(resumed, [self.record], self.prefix, resumed_optimizer, 1, start_step=step)
            self.assertEqual(expected_metrics, actual_metrics)
            for key, expected in self.model.state_dict().items():
                self.assertTrue(torch.equal(expected, resumed.state_dict()[key]), key)

    def test_resume_rejects_changed_experiment(self):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=0.001)
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "resume.pt"
            save_training_state(path, self.model, optimizer, 0, records=[self.record])
            changed = copy.deepcopy(self.model)
            changed.config = replace(changed.config, write_mode="static")
            with self.assertRaisesRegex(ValueError, "model settings"):
                load_training_state(path, changed, optimizer, records=[self.record])
            with self.assertRaisesRegex(ValueError, "loss settings"):
                load_training_state(path, self.model, optimizer, loss_config=LossConfig(gate=0.2), records=[self.record])
            with self.assertRaisesRegex(ValueError, "data or record order"):
                load_training_state(path, self.model, optimizer, records=[])


if __name__ == "__main__":
    unittest.main()
