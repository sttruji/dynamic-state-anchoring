import importlib.util
from pathlib import Path
import unittest

from dsa.data.tokenize import TinyTokenizer


spec = importlib.util.spec_from_file_location(
    "audit_lotus_tokenizer", Path(__file__).resolve().parents[1] / "scripts/audit_lotus_tokenizer.py")
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


class AuditTokenizer(TinyTokenizer):
    def decode(self, token_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False):
        return super().decode(token_ids, skip_special_tokens=skip_special_tokens)


class CorpusTokenizerAuditTests(unittest.TestCase):
    def test_frozen_vocabulary_and_identical_policy_mapping_reused(self):
        from unittest.mock import patch
        tokenizer = AuditTokenizer()
        record = {"question": "A sum?", "steps": ["<<1+2=3>>", "<<3*2=6>>"], "answer": "6"}
        with patch.object(tokenizer, "get_vocab", wraps=tokenizer.get_vocab) as vocabulary:
            with patch.object(audit, "tokenize_record", wraps=audit.tokenize_record) as mapping:
                result, _ = audit.audit_records([record], tokenizer, split="train")
        self.assertEqual(vocabulary.call_count, 1)
        self.assertEqual(mapping.call_count, 1)
        self.assertEqual(result["policies"]["conclusive"], result["policies"]["verified_only"])

    def test_nearest_rank_quantiles_and_strict_context_overflow(self):
        from collections import Counter
        result = audit.length_summary(Counter({10: 1, 1024: 8, 1025: 1}), 1024)
        self.assertEqual(result["record_count"], 10)
        self.assertEqual(result["quantiles"]["0.5"], 1024)
        self.assertEqual(result["quantiles"]["0.9"], 1024)
        self.assertEqual(result["quantiles"]["0.95"], 1025)
        self.assertEqual(result["over_context"], 1)
        self.assertEqual(result["at_context"], 8)

    def test_no_event_and_overflow_rows_retained_and_supervised_when_budget_fits(self):
        records = [
            {"question": "No event?", "steps": ["Just prose."], "answer": "9"},
            {"question": "Five events?", "steps": ["x=1\ny=2\nz=3\na=4\nb=5"], "answer": "9"},
        ]
        result, samples = audit.audit_records(records, AuditTokenizer(), split="train")
        self.assertEqual(result["record_count"], 2)
        for mode in ("conclusive", "verified_only"):
            policy = result["policies"][mode]
            self.assertEqual(policy["tokenized_records"], 2)
            self.assertEqual(policy["tokenizer_mapping_failures"], 0)
            self.assertEqual(policy["lengths"]["explicit_including_eos"]["record_count"], 2)
        conclusive = result["policies"]["conclusive"]
        self.assertEqual(conclusive["loop_budgets"]["4"], {"controller_supervised": 0, "no_events": 1, "event_overflow": 1})
        self.assertEqual(conclusive["loop_budgets"]["8"], {"controller_supervised": 1, "no_events": 1, "event_overflow": 0})
        self.assertEqual(result["policies"]["verified_only"]["loop_budgets"]["32"]["no_events"], 2)
        self.assertEqual(len(samples), 4)

    def test_source_collisions_and_invalid_records_counted_without_becoming_valid(self):
        records = [{"question": "<SA><SA>?", "steps": ["x=1"], "answer": "9"},
                   {"question": "broken", "steps": [7], "answer": "9"}]
        result, samples = audit.audit_records(records, AuditTokenizer(), split="valid")
        self.assertEqual(result["record_count"], 2)
        self.assertEqual(result["format_failures"], 1)
        self.assertEqual(result["records_with_literal_control_tokens"], 1)
        self.assertEqual(result["literal_control_token_occurrences"]["<SA>"], 2)
        for policy in result["policies"].values():
            self.assertEqual(policy["annotation_failures"], 1)
            self.assertEqual(policy["tokenized_records"], 0)
        self.assertEqual(len(samples), 3)

    def test_held_out_test_does_not_annotate_tokenize_or_save_examples(self):
        from unittest.mock import patch
        records = [{"question": "Held out?", "steps": ["x=1"], "answer": "1"}]
        with patch.object(audit, "annotate_record", side_effect=AssertionError("test annotation forbidden")):
            result, samples = audit.audit_records(records, AuditTokenizer(), split="test")
        self.assertEqual(result["record_count"], 1)
        self.assertEqual(result["policies"], {})
        self.assertEqual(samples, [])


if __name__ == "__main__":
    unittest.main()
