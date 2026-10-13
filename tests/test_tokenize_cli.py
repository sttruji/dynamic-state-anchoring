import contextlib
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest

from dsa.data.milestones import annotate_record
from dsa.data.tokenize import TinyTokenizer
from scripts.tokenize_dsa_data import main


class TokenizeCliTests(unittest.TestCase):
    def paths(self, root):
        return root / "annotated.jsonl", root / "tokens.jsonl", root / "report.json", root / "tokenizer"

    def run_cli(self, source, output, report, tokenizer, *extra):
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            return main(["--input", str(source), "--output", str(output), "--report", str(report),
                         "--tokenizer-dir", str(tokenizer), "--loops", "2", *extra])

    def write_records(self, source, records):
        source.write_text("".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records), encoding="utf-8")

    def test_valid_and_overflow_retained_with_identity_and_reloaded_ids(self):
        with tempfile.TemporaryDirectory() as directory:
            source, output, report, tokenizer = self.paths(Path(directory))
            records = [annotate_record({"question": "Q 🧮\u2028\u2029", "answer": "x = 1\n#### 9"}),
                       annotate_record({"question": "Q2", "answer": "x = 1\ny = 2\nz = 3\n#### 9"})]
            self.write_records(source, records)
            original = source.read_bytes()
            self.assertEqual(self.run_cli(source, output, report, tokenizer), 0)
            tokens = [json.loads(line) for line in output.read_text().splitlines()]
            summary = json.loads(report.read_text())
            identity = json.loads((tokenizer / "tokenizer_identity.json").read_text())
            self.assertEqual(len(tokens), 2)
            self.assertTrue(tokens[0]["controller_supervised"])
            self.assertFalse(tokens[1]["controller_supervised"])
            self.assertEqual(len(tokens[1]["checkpoint_positions"]), 3)
            self.assertEqual(summary["controller_label_overflow"], 1)
            self.assertEqual(summary["controller_masked_count"], 1)
            self.assertEqual(summary["tokenizer_mapping_failures"], 0)
            self.assertFalse(summary["truncation"])
            self.assertEqual(identity["input_sha256"], hashlib.sha256(original).hexdigest())
            self.assertEqual(identity["tokenized_output_sha256"], hashlib.sha256(output.read_bytes()).hexdigest())
            self.assertEqual(source.read_bytes(), original)
            loaded = TinyTokenizer.from_pretrained(tokenizer)
            self.assertEqual(tokens[0]["question_input_ids"], loaded.encode("Q 🧮\u2028\u2029\n"))
            self.assertEqual(identity["marker_token_id"], loaded.marker_token_id)
            # A repeat replaces the complete saved tokenizer consistently.
            self.run_cli(source, output, report, tokenizer)
            self.assertEqual(json.loads((tokenizer / "tokenizer_identity.json").read_text()), identity)

    def test_invalid_late_line_preserves_existing_artifacts(self):
        with tempfile.TemporaryDirectory() as directory:
            source, output, report, tokenizer = self.paths(Path(directory))
            self.write_records(source, [annotate_record({"question": "Q", "answer": "x = 1\n#### 9"})])
            source.write_text(source.read_text() + "{}\n")
            output.write_text("existing tokens")
            report.write_text("existing report")
            with self.assertRaises(SystemExit):
                self.run_cli(source, output, report, tokenizer)
            self.assertEqual(output.read_text(), "existing tokens")
            self.assertEqual(report.read_text(), "existing report")
            self.assertFalse(tokenizer.exists())

    def test_invalid_token_mapping_creates_no_partial_outputs(self):
        with tempfile.TemporaryDirectory() as directory:
            source, output, report, tokenizer = self.paths(Path(directory))
            valid = annotate_record({"question": "Q", "answer": "x = 1\n#### 9"})
            invalid = annotate_record({"question": "Q2", "answer": "x = 1\n#### 9"})
            invalid["events"][0]["end"] -= 1
            self.write_records(source, [valid, invalid])
            with self.assertRaises(SystemExit):
                self.run_cli(source, output, report, tokenizer)
            self.assertFalse(output.exists())
            self.assertFalse(report.exists())
            self.assertFalse(tokenizer.exists())

    def test_no_event_record_retained_and_unsafe_paths_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            source, output, report, tokenizer = self.paths(Path(directory))
            self.write_records(source, [annotate_record({"question": "Q", "answer": "Unsupported prose.\n#### 9"})])
            self.run_cli(source, output, report, tokenizer)
            summary = json.loads(report.read_text())
            self.assertEqual(summary["record_count"], 1)
            self.assertEqual(summary["no_event_count"], 1)
            self.assertEqual(summary["controller_mask_reasons"], {"no_events": 1})
            with self.assertRaises(SystemExit):
                self.run_cli(source, source, report, tokenizer)
            with self.assertRaises(SystemExit):
                self.run_cli(source, output, report, tokenizer, "--tokenizer", "unavailable-online-model")


if __name__ == "__main__":
    unittest.main()
