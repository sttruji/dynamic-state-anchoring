import tempfile
import unittest

from dsa.data.milestones import annotate_record
from dsa.data.tokenize import TinyTokenizer, collate_records, register_marker_token, tokenize_record


def annotated(question="What is next?", raw="First <<1+2=3>>3.\nx = 3.\nThen <<3*2=6>>6.\n", answer="6"):
    return annotate_record({"question": question, "answer": raw + "#### " + answer})


class TinyTokenizerTests(unittest.TestCase):
    def setUp(self):
        self.tokenizer = TinyTokenizer()

    def test_unicode_roundtrip_offsets_and_fixed_vocabulary(self):
        text = "Café 🧮 × 2<SA>!"
        encoded = self.tokenizer(text, return_offsets_mapping=True)
        self.assertEqual(self.tokenizer.decode(encoded["input_ids"]), text)
        self.assertEqual(len(self.tokenizer), 259)
        self.assertEqual(self.tokenizer.encode("<SA>"), [2])
        emoji = text.index("🧮")
        self.assertEqual(encoded["offset_mapping"].count((emoji, emoji + 1)), 4)
        self.assertEqual(register_marker_token(self.tokenizer), 2)

    def test_save_reload_retains_token_ids(self):
        with tempfile.TemporaryDirectory() as directory:
            self.tokenizer.save_pretrained(directory)
            loaded = TinyTokenizer.from_pretrained(directory)
            self.assertEqual(loaded.encode("é<SA>😀"), self.tokenizer.encode("é<SA>😀"))

    def test_full_explicit_sequence_and_question_mask(self):
        record = annotated(question="Café?")
        result = tokenize_record(record, self.tokenizer, 8)
        text = record["question"] + "\n" + record["rationale_marked"] + "\n### " + record["final_answer"]
        self.assertEqual(result["explicit_input_ids"], self.tokenizer.encode(text) + [1])
        self.assertEqual(result["question_input_ids"], self.tokenizer.encode("Café?\n"))
        self.assertEqual(result["answer_input_ids"], self.tokenizer.encode("6") + [1])
        self.assertEqual(result["latent_answer_prefix_ids"], self.tokenizer.encode("\n### "))
        self.assertEqual(result["latent_suffix_input_ids"], self.tokenizer.encode("\n### 6") + [1])
        self.assertEqual(result["latent_suffix_labels"], [-100] * 5 + self.tokenizer.encode("6") + [1])
        prefix_length = len("Café?".encode("utf-8"))
        self.assertEqual(result["explicit_labels"][:prefix_length], [-100] * prefix_length)
        self.assertEqual(result["explicit_labels"][prefix_length:], result["explicit_input_ids"][prefix_length:])
        self.assertEqual([result["explicit_input_ids"][index + 1] for index in result["checkpoint_positions"]], [2, 2])

    def test_unicode_event_ends_use_python_chars_and_raw_tokens(self):
        record = annotated(raw="🧮 café <<1+2=3>>3.\nx = 3.\n", answer="9")
        result = tokenize_record(record, self.tokenizer, 8)
        expected = [len(record["rationale_raw"][:event["end"]].encode("utf-8")) for event in record["events"]]
        self.assertEqual(result["raw_event_token_ends"], expected)
        self.assertEqual(sum(result["gate_targets"]), len(record["events"]))
        self.assertTrue(all(1 <= loop <= 8 for loop in result["event_loops"]))

    def test_no_events_and_overflow_mask_controller_without_dropping_record(self):
        empty = tokenize_record(annotated(raw="Unsupported prose.\n"), self.tokenizer, 2)
        overflow = tokenize_record(annotated(raw="x = 1\ny = 2\nz = 3\n", answer="9"), self.tokenizer, 2)
        for result in (empty, overflow):
            self.assertFalse(result["controller_supervised"])
            self.assertEqual(result["gate_targets"], [0, 0])
            self.assertEqual(result["event_loops"], [])
            self.assertTrue(result["explicit_input_ids"])
            self.assertTrue(result["answer_input_ids"])
        self.assertEqual(len(overflow["checkpoint_positions"]), 3)

    def test_literal_marker_collisions_and_stale_offsets_rejected(self):
        for field in ("question", "rationale_raw", "final_answer"):
            record = annotated()
            record[field] += "<SA>"
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, "literal"):
                tokenize_record(record, self.tokenizer, 8)
        record = annotated()
        record["events"][0]["end"] -= 1
        with self.assertRaisesRegex(ValueError, "exactly preserve"):
            tokenize_record(record, self.tokenizer, 8)
        record = annotated()
        record["events"].reverse()
        with self.assertRaisesRegex(ValueError, "strictly increasing"):
            tokenize_record(record, self.tokenizer, 8)

    def test_collation_pads_question_and_masks_explicit_padding(self):
        records = [tokenize_record(annotated(question=q), self.tokenizer, 8) for q in ("Q", "Longer question")]
        batch = collate_records(records, 0)
        width = len(batch["question_input_ids"][0])
        self.assertEqual(width, len(batch["question_input_ids"][1]))
        self.assertEqual(sum(batch["question_attention_mask"][0]), len(records[0]["question_input_ids"]))
        self.assertEqual(batch["explicit_labels"][0][-1], -100)
        self.assertEqual(batch["checkpoint_positions"][0], records[0]["checkpoint_positions"])


class HuggingFaceTokenizerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            from tokenizers import Tokenizer, models, trainers
            from transformers import PreTrainedTokenizerFast
        except ImportError:
            raise unittest.SkipTest("optional transformers/tokenizers dependencies unavailable")
        cls.backend_class = Tokenizer
        cls.models = models
        cls.trainers = trainers
        cls.fast_class = PreTrainedTokenizerFast

    def tokenizer(self):
        backend = self.backend_class(self.models.BPE(unk_token="<UNK>"))
        corpus = ["Question\nFirst <<1+2=3>>3.\nx = 3.\nThen <<3*2=6>>6.\n\n### 6", "🧮 café <<1+2=3>>3.\nx = 3.\n"]
        backend.train_from_iterator(corpus * 30, self.trainers.BpeTrainer(vocab_size=220, special_tokens=["<PAD>", "<EOS>", "<UNK>"], show_progress=False))
        return self.fast_class(tokenizer_object=backend, pad_token="<PAD>", eos_token="<EOS>", unk_token="<UNK>")

    def test_real_bpe_full_sequence_offsets_and_registration(self):
        tokenizer = self.tokenizer()
        marker_id = register_marker_token(tokenizer)
        count = len(tokenizer)
        self.assertEqual(register_marker_token(tokenizer), marker_id)
        self.assertEqual(len(tokenizer), count)
        record = annotated(question="Question")
        result = tokenize_record(record, tokenizer, 8)
        text = record["question"] + "\n" + record["rationale_marked"] + "\n### " + record["final_answer"]
        self.assertEqual(result["explicit_input_ids"], tokenizer.encode(text, add_special_tokens=False) + [tokenizer.eos_token_id])
        separate = tokenizer.encode(record["question"], add_special_tokens=False) + tokenizer.encode("\n" + record["rationale_marked"] + "\n### 6", add_special_tokens=False)
        self.assertNotEqual(result["explicit_input_ids"][:-1], separate)
        raw = tokenizer(record["rationale_raw"], return_offsets_mapping=True, add_special_tokens=False)
        self.assertTrue(any(end - start > 1 for start, end in raw["offset_mapping"]))
        for event, end in zip(record["events"], result["raw_event_token_ends"]):
            start, stop = raw["offset_mapping"][end - 1]
            self.assertLess(start, event["end"])
            self.assertGreaterEqual(stop, event["end"])
        for position in result["checkpoint_positions"]:
            self.assertEqual(result["explicit_input_ids"][position + 1], marker_id)
        suffix = tokenizer("\n### 6", return_offsets_mapping=True, add_special_tokens=False)
        self.assertTrue(any(start < 5 < end for start, end in suffix["offset_mapping"]))
        prefix_ids = tokenizer.encode("\n### ", add_special_tokens=False)
        answer_ids = tokenizer.encode("6", add_special_tokens=False) + [tokenizer.eos_token_id]
        self.assertNotEqual(result["latent_suffix_input_ids"], suffix["input_ids"] + [tokenizer.eos_token_id])
        self.assertEqual(result["latent_answer_prefix_ids"], prefix_ids)
        self.assertEqual(result["latent_suffix_input_ids"], prefix_ids + answer_ids)
        self.assertEqual(result["latent_suffix_labels"], [-100] * len(prefix_ids) + answer_ids)

    def test_ordinary_vocabulary_collision_is_rejected(self):
        tokenizer = self.tokenizer()
        # The fixture always includes Q in its base alphabet. Vocabulary map
        # iteration order can otherwise select whitespace instead of a collision.
        token = "Q"
        self.assertIn(token, tokenizer.get_vocab())
        self.assertNotIn(token, tokenizer.all_special_tokens)
        with self.assertRaisesRegex(ValueError, "collides"):
            register_marker_token(tokenizer, marker_token=token)

    def test_preexisting_normalizing_marker_is_rejected(self):
        from tokenizers import AddedToken
        tokenizer = self.tokenizer()
        tokenizer.add_special_tokens({"additional_special_tokens": [AddedToken("<SA>", normalized=True, lstrip=True)]})
        with self.assertRaisesRegex(ValueError, "unsafe"):
            register_marker_token(tokenizer)

    def test_resize_preserves_tied_lm_weights_and_trains_new_row(self):
        try:
            import torch
            from transformers import GPT2Config, GPT2LMHeadModel
        except ImportError:
            self.skipTest("torch unavailable")
        tokenizer = self.tokenizer()
        model = GPT2LMHeadModel(GPT2Config(vocab_size=len(tokenizer), n_embd=16, n_layer=1, n_head=2, n_positions=32))
        old_rows = model.get_input_embeddings().weight.detach().clone()
        marker_id = register_marker_token(tokenizer, model)
        self.assertIs(model.get_input_embeddings().weight, model.get_output_embeddings().weight)
        self.assertTrue(model.get_input_embeddings().weight.requires_grad)
        torch.testing.assert_close(model.get_input_embeddings().weight[:len(old_rows)], old_rows)
        model(input_ids=torch.tensor([[marker_id, tokenizer.eos_token_id]]), labels=torch.tensor([[marker_id, tokenizer.eos_token_id]])).loss.backward()
        self.assertGreater(model.get_input_embeddings().weight.grad[marker_id].abs().sum().item(), 0)


if __name__ == "__main__":
    unittest.main()
