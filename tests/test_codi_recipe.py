"""Batched training, CODI answer distillation, LoRA and new-module learning rates."""
import copy
from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest

import torch

from dsa.data.milestones import annotate_record
from dsa.data.tokenize import TinyTokenizer, answer_alignment, tokenize_record
from dsa.models.anchored_loop import AnchorConfig, AnchoredLoop
from dsa.models.backbone import BackboneAdapter
from dsa.models.lora import LoRAConv1D, apply_lora, merged_state_dict
from dsa.training.objectives import LossConfig, joint_objective
from dsa.training.pilot import (PilotConfig, _parameter_groups, sha256_file,
                                tokenizer_directory_sha256, train_pilot)


torch.set_num_threads(1)
RAW = [{"question": "Add 2 and 3, then double it.", "answer": "<<2+3=5>>\n<<5*2=10>>\n#### 10"},
       {"question": "Add 3 and 4, then 1.", "answer": "<<3+4=7>>\n<<7+1=8>>\n#### 8"},
       {"question": "Unannotated question here?", "answer": "Just count them.\n#### 16"}]


def tiny_model(recurrence="standard", mode="learned"):
    return AnchoredLoop(BackboneAdapter.tiny(259), AnchorConfig(
        loop_budget=4, marker_token_id=2, write_mode=mode, recurrence_mode=recurrence))


def objective(model, batches, prefix, config):
    """Losses and gradients after accumulating per-batch objectives with equal weight."""
    model.zero_grad(set_to_none=True)
    totals = {key: 0.0 for key in ("total", "answer", "cot", "gate", "anchor", "distill")}
    for batch in batches:
        losses = joint_objective(model, batch, prefix, config)
        (losses["total"] * len(batch) / sum(map(len, batches))).backward()
        for key in totals:
            totals[key] += losses[key].item() * len(batch) / sum(map(len, batches))
    return totals, {name: parameter.grad.clone() for name, parameter in model.named_parameters()
                    if parameter.grad is not None}


class BatchedObjectiveTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(5)
        self.tokenizer = TinyTokenizer()
        self.prefix = self.tokenizer.encode("\n### ")
        self.records = [tokenize_record(annotate_record(row), self.tokenizer, 4) for row in RAW]
        for record in self.records:
            record["answer_alignment"] = answer_alignment(record, self.tokenizer)
        self.assertEqual(len({len(record["explicit_input_ids"]) for record in self.records}), 3)
        self.assertEqual([record["controller_supervised"] for record in self.records], [True, True, False])

    def test_padded_batch_equals_accumulating_its_records(self):
        for recurrence in ("standard", "flt_memory"):
            for mode in ("learned", "static"):
                for distill in (0.0, 1.0):
                    model = tiny_model(recurrence, mode)
                    config = LossConfig(answer_distill=distill)
                    expected, expected_grads = objective(model, [[record] for record in self.records],
                                                         self.prefix, config)
                    actual, actual_grads = objective(model, [self.records], self.prefix, config)
                    label = f"{recurrence}/{mode}/distill={distill}"
                    for key in expected:
                        self.assertAlmostEqual(expected[key], actual[key], places=5, msg=f"{label} {key}")
                    self.assertEqual(expected_grads.keys(), actual_grads.keys(), label)
                    for name, grad in expected_grads.items():
                        torch.testing.assert_close(actual_grads[name], grad, rtol=1e-4, atol=1e-6,
                                                   msg=f"{label} {name}")

    def test_single_record_list_matches_the_record(self):
        model = tiny_model()
        left = joint_objective(model, self.records[0], self.prefix)
        right = joint_objective(model, [self.records[0]], self.prefix)
        for key in ("total", "answer", "cot", "gate", "anchor", "distill"):
            self.assertTrue(torch.equal(left[key], right[key]), key)

    def test_answer_alignment_pairs_the_token_before_the_answer(self):
        for record in self.records:
            teacher, student = record["answer_alignment"]
            explicit, suffix = record["explicit_input_ids"], record["latent_suffix_input_ids"]
            self.assertEqual(explicit[teacher], suffix[student])
            # The next teacher token starts the answer; the student's next is its first answer token.
            answer = self.tokenizer.decode(record["answer_input_ids"][:-1])
            self.assertTrue(self.tokenizer.decode(explicit[teacher + 1:-1]).strip() == answer)
            self.assertEqual(suffix[student + 1], record["answer_input_ids"][0])
            self.assertEqual(student, len(record["latent_answer_prefix_ids"]) - 1)

    def test_answer_alignment_rejects_a_changed_explicit_tail(self):
        record = copy.deepcopy(self.records[0])
        record["explicit_input_ids"][-2] = self.tokenizer.encode("9")[0]
        with self.assertRaisesRegex(ValueError, "does not end with"):
            answer_alignment(record, self.tokenizer)

    def test_distillation_is_weighted_into_the_total_and_trains_the_student(self):
        model = tiny_model("flt_memory", "static")
        without = joint_objective(model, self.records, self.prefix)
        self.assertEqual(without["distill"].item(), 0.0)
        losses = joint_objective(model, self.records, self.prefix, LossConfig(answer_distill=2.0))
        self.assertGreater(losses["distill"].item(), 0)
        self.assertAlmostEqual(losses["total"].item(), without["total"].item() + 2 * losses["distill"].item(), places=5)
        losses["distill"].backward()
        self.assertGreater(model.initial_workspace.grad.norm().item(), 0)
        unaligned = [{key: value for key, value in record.items() if key != "answer_alignment"}
                     for record in self.records]
        with self.assertRaisesRegex(ValueError, "answer_alignment"):
            joint_objective(model, unaligned, self.prefix, LossConfig(answer_distill=1.0))


class LoRATests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(9)
        self.tokenizer = TinyTokenizer()
        self.prefix = self.tokenizer.encode("\n### ")
        self.record = tokenize_record(annotate_record(RAW[0]), self.tokenizer, 4)
        self.question = torch.tensor([self.record["question_input_ids"]])
        self.base = tiny_model("flt_memory")

    def adapted(self, perturb=True):
        model = apply_lora(copy.deepcopy(self.base), rank=4, alpha=8.0, dropout=0.0)
        if perturb:
            with torch.no_grad():
                for module in model.modules():
                    if isinstance(module, LoRAConv1D):
                        module.lora_b.normal_(std=0.05)
        return model

    def test_adapters_start_at_the_base_model_and_freeze_it(self):
        model = self.adapted(perturb=False).eval()
        base, adapted = self.base.eval()(self.question, torch.ones_like(self.question)), model(
            self.question, torch.ones_like(self.question))
        self.assertTrue(torch.equal(base.workspace, adapted.workspace))
        trainable = {name for name, parameter in model.named_parameters() if parameter.requires_grad}
        backbone = {name for name in trainable if name.startswith("backbone.")}
        self.assertEqual({name.rsplit(".", 1)[-1] for name in backbone - {"backbone.model.transformer.wte.weight"}},
                         {"lora_a", "lora_b"})
        self.assertIn("backbone.model.transformer.wte.weight", backbone)
        self.assertEqual(len(backbone) - 1, 2 * 4 * len(model.backbone.model.transformer.h))
        self.assertTrue({"initial_anchor", "initial_workspace", "gate_head.weight"} <= trainable)

    def test_flt_memory_queries_use_the_adapted_projection(self):
        model = self.adapted().eval()
        merged = copy.deepcopy(self.base).eval()
        merged.backbone.model.load_state_dict(merged_state_dict(model.backbone.model))
        mask = torch.ones_like(self.question)
        torch.testing.assert_close(model(self.question, mask).workspace, merged(self.question, mask).workspace,
                                   rtol=1e-5, atol=1e-5)

    def test_saved_checkpoint_folds_adapters_into_plain_gpt2(self):
        model = self.adapted().eval()
        mask = torch.ones_like(self.question)
        expected = model.generate(self.question, mask, answer_prefix_ids=self.prefix, max_new_tokens=4)
        with tempfile.TemporaryDirectory() as temp:
            model.save_pretrained(temp, self.tokenizer)
            restored = AnchoredLoop.from_pretrained(temp).eval()
            self.assertFalse(any(isinstance(module, LoRAConv1D) for module in restored.modules()))
            self.assertIs(restored.backbone.model.lm_head.weight, restored.backbone.embeddings.weight)
            actual = restored.generate(self.question, mask, answer_prefix_ids=self.prefix, max_new_tokens=4)
            torch.testing.assert_close(actual.loop.workspace, expected.loop.workspace, rtol=1e-5, atol=1e-5)
            self.assertTrue(torch.equal(actual.token_ids, expected.token_ids))

    def test_update_changes_only_adapters_marker_row_and_new_modules(self):
        model = self.adapted(perturb=False)
        before = copy.deepcopy(model.state_dict())
        config = PilotConfig("unused", "unused", tiny=True, lora_rank=4, learning_rate=0.01,
                             new_module_learning_rate=0.1, weight_decay=0.1)
        groups = _parameter_groups(model, config)
        self.assertEqual([group.get("lr") for group in groups], [None, 0.1, 0.1])
        optimizer = torch.optim.AdamW(groups, lr=config.learning_rate, weight_decay=config.weight_decay)
        joint_objective(model, self.record, self.prefix)["total"].backward()
        optimizer.step()
        after = model.state_dict()
        embedding = "backbone.model.transformer.wte.weight"
        changed = before[embedding] != after[embedding]
        self.assertEqual(changed.any(dim=1).nonzero().flatten().tolist(), [2])
        for name, tensor in after.items():
            if ".base." in name or name.endswith(("wpe.weight", "ln_f.weight", "ln_1.bias")):
                self.assertTrue(torch.equal(tensor, before[name]), name)
        self.assertFalse(torch.equal(after["initial_workspace"], before["initial_workspace"]))
        self.assertFalse(torch.equal(after["backbone.model.transformer.h.0.attn.c_attn.lora_b"],
                                     before["backbone.model.transformer.h.0.attn.c_attn.lora_b"]))


class RecipePilotTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        prepared = self.root / "prepared"
        tokenizer = TinyTokenizer()
        tokenizer.save_pretrained(prepared / "tokenizer")
        records = [tokenize_record(annotate_record(row), tokenizer, 4) for row in RAW]
        for index, record in enumerate(records):
            record["sample_id"] = str(index)
        (prepared / "train.tokenized.jsonl").write_text("".join(json.dumps(row) + "\n" for row in records))
        for split in ("valid", "test"):
            (prepared / f"{split}.jsonl").write_text(json.dumps(RAW[0]) + "\n")
        manifest = {"schema_version": 1, "backbone": "tiny-gpt2", "revision": "random", "tiny": True,
                    "loop_budget": 4, "tokenizer_sha256": tokenizer_directory_sha256(prepared / "tokenizer"),
                    "splits": {split: {"path": path, "sha256": sha256_file(prepared / path)}
                               for split, path in (("train", "train.tokenized.jsonl"),
                                                   ("valid", "valid.jsonl"), ("test", "test.jsonl"))}}
        (prepared / "manifest.json").write_text(json.dumps(manifest))
        self.config = PilotConfig(prepared, self.root / "run", tiny=True, loops=4, steps=4, save_every=1,
                                  learning_rate=0.003, device="cpu", recurrence_mode="flt_memory",
                                  batch_size=2, answer_distill_weight=1.0, lora_rank=4, lora_alpha=8.0,
                                  lora_dropout=0.0, new_module_learning_rate=0.01, warmup_steps=2)

    def metrics(self, output):
        return [json.loads(line) for line in (Path(output) / "metrics.jsonl").read_text().splitlines()]

    def test_lora_distill_batch_run_resumes_exactly(self):
        whole = train_pilot(self.config)
        interrupted = replace(self.config, output_dir=self.root / "resumed", steps=2)
        train_pilot(interrupted)
        resumed = train_pilot(replace(interrupted, steps=4))
        self.assertIsNotNone(resumed["resumed_from"])
        self.assertEqual(whole["initial_state_sha256"], resumed["initial_state_sha256"])
        left, right = self.metrics(self.config.output_dir), self.metrics(interrupted.output_dir)
        self.assertEqual([row["learning_rate"] for row in left], [0.0015, 0.003, 0.003, 0.003])
        for a, b in zip(left, right, strict=True):
            for key in ("total", "answer", "cot", "gate", "anchor", "distill", "sample_ids", "gradient_norm"):
                self.assertEqual(a[key], b[key], key)
        self.assertTrue(all(row["distill"] > 0 and len(row["sample_ids"]) == 2 for row in left))
        expected = AnchoredLoop.from_pretrained(Path(whole["checkpoint"]) / "model")
        actual = AnchoredLoop.from_pretrained(Path(resumed["checkpoint"]) / "model")
        for key, tensor in expected.state_dict().items():
            self.assertTrue(torch.equal(tensor, actual.state_dict()[key]), key)

    def test_batch_matches_accumulation_order_and_losses(self):
        batched = train_pilot(replace(self.config, lora_rank=0, steps=2, warmup_steps=0))
        accumulated = train_pilot(replace(self.config, output_dir=self.root / "accumulated", lora_rank=0,
                                          steps=2, warmup_steps=0, batch_size=1, gradient_accumulation=2))
        self.assertEqual(batched["initial_state_sha256"], accumulated["initial_state_sha256"])
        for a, b in zip(self.metrics(self.config.output_dir), self.metrics(self.root / "accumulated"), strict=True):
            self.assertEqual(a["sample_ids"], b["sample_ids"])
            for key in ("total", "answer", "cot", "gate", "anchor", "distill", "gradient_norm"):
                self.assertAlmostEqual(a[key], b[key], places=4, msg=key)

    def test_cosine_schedule_binds_the_step_total(self):
        config = replace(self.config, lr_schedule="cosine", steps=2, warmup_steps=0)
        train_pilot(config)
        rates = [row["learning_rate"] for row in self.metrics(config.output_dir)]
        self.assertAlmostEqual(rates[0], 0.0015)
        self.assertAlmostEqual(rates[1], 0.0)
        with self.assertRaisesRegex(ValueError, "settings differ"):
            train_pilot(replace(config, steps=3))

    def test_invalid_recipe_settings_are_refused(self):
        for change in ({"batch_size": 0}, {"lora_rank": -1}, {"lora_dropout": 1.0}, {"warmup_steps": 5},
                       {"lr_schedule": "linear"}, {"new_module_learning_rate": 0.0},
                       {"answer_distill_weight": -1.0}):
            with self.assertRaises(ValueError, msg=str(change)):
                train_pilot(replace(self.config, **change))


if __name__ == "__main__":
    unittest.main()
