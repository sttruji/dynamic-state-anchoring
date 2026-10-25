import csv
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import torch

from dsa.bench import data as bench_data
from dsa.bench import harness
from dsa.bench.models import BenchModel, build_model
from dsa.bench.models.codi import merge_codi_state
from dsa.bench.scoring import score, strict_answer
from dsa.data.milestones import annotate_record
from dsa.data.pilot_prepare import drop_event

ROOT = Path(__file__).resolve().parents[1]


class EchoModel(BenchModel):
    """Answers with the last number in the question; counts calls for reuse checks."""

    def __init__(self, tag="v1", device="cpu", batch_size=2):
        self.name, self.tag, self.batch_size, self.calls = "echo", tag, batch_size, 0

    def identity(self):
        return {"tag": self.tag}

    def generate(self, questions):
        self.calls += len(questions)
        return [f"<<x>>\n### {question.split()[-1]}" for question in questions]


class ScoringTests(unittest.TestCase):
    def test_lotus_format_and_codi_format(self):
        self.assertTrue(score("<<16-3-4=9>>\n<<9*2=18>>\n### 18", "18")["strict_correct"])
        self.assertTrue(score("The answer is: 1,400", "1400")["strict_correct"])
        self.assertTrue(score("18", "18")["strict_correct"])  # DSA latent continuation after "### ".

    def test_incidental_credit_kept_for_comparability_but_not_strict(self):
        for text in ("iced.5", "10*5"):
            result = score(text, "5")
            self.assertTrue(result["correct"], text)
            self.assertFalse(result["strict_correct"], text)

    def test_non_integer_gold_uses_tolerance_and_integers_stay_exact(self):
        self.assertTrue(score("### 10.6667", "10.666666666666666")["correct"])
        self.assertFalse(score("### 10.7", "10.666666666666666")["correct"])
        self.assertTrue(score("### 18.0", "18")["correct"])
        self.assertFalse(score("### 18.00001", "18")["correct"])

    def test_choice_answers(self):
        self.assertTrue(score("Because ... The answer is: C", "C", "choice")["correct"])
        self.assertFalse(score("The answer is: B", "C", "choice")["correct"])

    def test_strict_answer_rejects_expressions(self):
        self.assertIsNone(strict_answer("### 3+4"))
        self.assertEqual(strict_answer("### $12.50.\n\nmore"), "12.50")


class DataTests(unittest.TestCase):
    def test_canonical_jsonl_is_byte_stable(self):
        rows = [{"b": "é", "a": 1}, {"a": 2, "b": "x"}]
        with tempfile.TemporaryDirectory() as tmp:
            first = bench_data.write_jsonl(Path(tmp) / "a.jsonl", rows)
            second = bench_data.write_jsonl(Path(tmp) / "b.jsonl", [dict(reversed(row.items())) for row in rows])
        self.assertEqual(first, second)

    def test_number_normalization(self):
        self.assertEqual(bench_data._number(145.0), "145")
        self.assertEqual(bench_data._number("2,000"), "2000")
        self.assertEqual(bench_data._number(10.6666666667), "10.6666666667")

    def test_drop_event_rebuilds_markers_and_rejects_ambiguity(self):
        record = annotate_record({"question": "q", "answer": "a <<2*3=6>> <<6+1=7>> b #### 9"})
        trimmed = drop_event(record, "<<2*3=6>>")
        self.assertEqual(trimmed["event_count"], 1)
        self.assertEqual(trimmed["rationale_marked"].count("<SA>"), 1)
        self.assertEqual(trimmed["rationale_marked"].replace("<SA>", ""), record["rationale_raw"])
        self.assertEqual(trimmed["events"][0]["ordinal"], 0)
        with self.assertRaises(ValueError):
            drop_event(record, "<<1=1>>")

    @unittest.skipUnless((ROOT / "data/gsm_train.json").is_file(), "run scripts/setup_data.py first")
    def test_overfit32_matches_reviewed_plan(self):
        train = json.loads((ROOT / "data/gsm_train.json").read_text(encoding="utf-8"))
        rows = bench_data.build_overfit32(train)
        self.assertEqual(len(rows), 32)
        self.assertEqual(sum(row["event_count"] for row in rows), 58)
        override = next(row for row in rows if row["source_row"] == 112564)
        self.assertNotIn("<<120=120>><SA>", override["rationale_marked"])

    @unittest.skipUnless((ROOT / "data/gsm_train.json").is_file(), "run scripts/setup_data.py first")
    def test_prepare_pilot_accepts_reviewed_fixture(self):
        from dsa.data.pilot_prepare import prepare_pilot
        with tempfile.TemporaryDirectory() as tmp:
            manifest = prepare_pilot(storage_dir=None, source_dir=ROOT / "data", output_dir=Path(tmp) / "prepared",
                                     fixture="overfit32", loops=8, tiny=True)
            report = json.loads((Path(tmp) / "prepared/preparation_report.json").read_text())["train"]
        self.assertEqual(manifest["splits"]["train"]["records"], 32)
        self.assertEqual(report["controller_supervised_count"], 29)
        self.assertEqual(report["controller_mask_reasons"], {"no_events": 3})
        self.assertEqual(manifest["request_identity"]["fixture"]["name"], "overfit32")

    def test_suites_only_name_known_datasets(self):
        for names in bench_data.SUITES.values():
            self.assertTrue(set(names) <= set(bench_data.EVAL_DATASETS))


class HarnessTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        self.data_dir, self.results = root / "data", root / "results"
        rows = [{"id": f"t/{i}", "question": f"what is {i}", "answer": str(i), "answer_type": "numeric"}
                for i in range(5)]
        info = bench_data.write_jsonl(self.data_dir / "bench/eval/dev64.jsonl", rows)
        self.lock = root / "lock.json"
        self.lock.write_text(json.dumps({"schema": bench_data.LOCK_SCHEMA, "files": {"eval/dev64.jsonl": info}}))
        patcher = mock.patch.object(bench_data, "LOCK_PATH", self.lock)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.addCleanup(self.tmp.cleanup)

    def run_echo(self, model, **kwargs):
        return harness.evaluate(model, "test:Echo", ["dev64"], data_dir=self.data_dir,
                                results_dir=self.results, **kwargs)

    def test_evaluate_saves_result_predictions_and_leaderboard(self):
        summary = self.run_echo(EchoModel())[0]
        self.assertEqual((summary["correct"], summary["n"]), (5, 5))
        self.assertTrue((self.results / "runs/echo/dev64.json").is_file())
        lines = (self.results / "predictions/echo/dev64.jsonl").read_text().splitlines()
        self.assertEqual(len(lines), 6)  # Identity header plus one line per question.
        with (self.results / "leaderboard.csv").open() as handle:
            table = list(csv.DictReader(handle))
        self.assertEqual(table[0]["model"], "echo")
        self.assertEqual(table[0]["dev64_acc"], "100.00")
        self.assertIn("echo", harness.format_table(self.results))

    def test_identical_identity_is_reused_and_changed_identity_refused(self):
        self.run_echo(EchoModel())
        again = EchoModel()
        self.run_echo(again)
        self.assertEqual(again.calls, 0)
        with self.assertRaisesRegex(ValueError, "model_identity"):
            self.run_echo(EchoModel(tag="v2"))
        replaced = EchoModel(tag="v2")
        self.run_echo(replaced, overwrite=True)
        self.assertEqual(replaced.calls, 5)

    def test_interrupted_run_resumes_and_ignores_partial_line(self):
        model = EchoModel()
        self.run_echo(model)
        path = self.results / "predictions/echo/dev64.jsonl"
        lines = path.read_text().splitlines()
        path.write_text("\n".join(lines[:3]) + "\n" + lines[3][:10])  # Two rows plus a torn write.
        (self.results / "runs/echo/dev64.json").unlink()
        resumed = EchoModel()
        summary = self.run_echo(resumed)[0]
        self.assertEqual(resumed.calls, 3)
        self.assertEqual(summary["n"], 5)

    def test_changed_dataset_is_refused(self):
        path = self.data_dir / "bench/eval/dev64.jsonl"
        path.write_text(path.read_text().replace("what is 4", "what is 44"))
        with self.assertRaisesRegex(ValueError, "differs from the locked"):
            self.run_echo(EchoModel())

    def test_smoke_limit_saves_nothing(self):
        summary = self.run_echo(EchoModel(), limit=2)[0]
        self.assertEqual(summary["n"], 2)
        self.assertFalse(self.results.exists())

    def test_stale_scorer_results_are_excluded(self):
        self.run_echo(EchoModel())
        path = self.results / "runs/echo/dev64.json"
        result = json.loads(path.read_text())
        result["identity"]["scorer_version"] = "old"
        path.write_text(json.dumps(result))
        current, stale = harness.collect_results(self.results)
        self.assertEqual(current, [])
        self.assertEqual(len(stale), 1)

    def test_bad_names_rejected(self):
        with self.assertRaises(ValueError):
            self.run_echo(EchoModel(), name="../escape")


class RegistryTests(unittest.TestCase):
    def test_module_class_plugins(self):
        model = build_model("tests.test_bench:EchoModel", tag="x")
        self.assertEqual(model.identity(), {"tag": "x"})
        with self.assertRaises(TypeError):
            build_model("tests.test_bench:ScoringTests")
        with self.assertRaises(ValueError):
            build_model("not-a-model")


class CODIMergeTests(unittest.TestCase):
    def test_lora_merge_matches_unmerged_conv1d_forward(self):
        torch.manual_seed(0)
        prefix = "codi.base_model.model.transformer.h.0.attn.c_attn"
        base, bias = torch.randn(6, 9), torch.randn(9)
        lora_a, lora_b = torch.randn(4, 6), torch.randn(9, 4)
        state = {f"{prefix}.base_layer.weight": base, f"{prefix}.base_layer.bias": bias,
                 f"{prefix}.lora_A.default.weight": lora_a, f"{prefix}.lora_B.default.weight": lora_b,
                 "codi.base_model.model.lm_head.weight": torch.randn(3, 6), "prj.1.weight": torch.ones(2, 2)}
        backbone, projection = merge_codi_state(state, lora_alpha=8)
        x = torch.randn(3, 6)
        expected = x @ base + bias + (8 / 4) * (x @ lora_a.T @ lora_b.T)
        merged = x @ backbone["transformer.h.0.attn.c_attn.weight"] + backbone["transformer.h.0.attn.c_attn.bias"]
        torch.testing.assert_close(merged, expected)
        self.assertNotIn("lm_head.weight", backbone)
        self.assertIn("1.weight", projection)


class HFCausalTests(unittest.TestCase):
    def test_batched_left_padded_greedy_equals_one_at_a_time(self):
        try:
            from tokenizers import Tokenizer, models, pre_tokenizers, trainers
            from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast
        except ImportError:
            self.skipTest("transformers/tokenizers unavailable")
        from dsa.bench.models.gpt2_sft import SFTGPT2

        backend = Tokenizer(models.BPE(unk_token="<UNK>"))
        backend.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        backend.train_from_iterator(["what is 12 plus 30\n<<12+30=42>>\n### 42"] * 20,
                                    trainers.BpeTrainer(vocab_size=120, special_tokens=["<EOS>", "<UNK>"],
                                                        show_progress=False))
        tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, eos_token="<EOS>", unk_token="<UNK>")
        torch.manual_seed(0)
        model = GPT2LMHeadModel(GPT2Config(vocab_size=len(tokenizer), n_embd=32, n_layer=2, n_head=4,
                                           n_positions=128, eos_token_id=tokenizer.eos_token_id))
        questions = ["what is 12", "what is 12 plus 30 plus 12 plus 30", "plus"]
        with tempfile.TemporaryDirectory() as tmp:
            model.save_pretrained(tmp)
            tokenizer.save_pretrained(tmp)
            wrapper = SFTGPT2(checkpoint=tmp, device="cpu", max_new_tokens=6, batch_size=3)
            batched = wrapper.generate(questions)
            single = [wrapper.generate([question])[0] for question in questions]
        self.assertEqual(batched, single)


if __name__ == "__main__":
    unittest.main()
