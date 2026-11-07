"""Checks of Stage 0 of step-level selection (docs/brainstorm/STEP_SELECTION.md) on tiny random GPT-2s: dropout reruns at
rate 0 are the model's own step, MC dropout is switched back off, batched readout matches one row at a time, the fixed
rule, and eval.py then report.py end to end."""
import contextlib
import io
import json
import os
import runpy
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn
from transformers import GPT2Config, GPT2LMHeadModel

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts/step_select"))

from stepsel import (choose, expand, features, mc_dropout, numbers, pick, read, readout, sample_step,  # noqa: E402
                     step)
import looped  # noqa: E402  (stepsel puts scripts/inner_loop on the path)
from dsa.bench.models import codi as codi_module  # noqa: E402
from tests.test_inner_loop_scripts import FakeCODI, fake_decoder  # noqa: E402

WIDTH, VOCAB = 32, 60


def tiny(seed=0):
    torch.manual_seed(seed)
    config = GPT2Config(n_layer=4, n_embd=WIDTH, n_head=4, vocab_size=VOCAB, n_positions=64)
    config._attn_implementation = "eager"
    return GPT2LMHeadModel(config).eval()


def projection():
    return nn.Sequential(nn.Dropout(0.1), nn.Linear(WIDTH, WIDTH), nn.LayerNorm(WIDTH)).eval()


def asked(model, length=5, seed=1):
    torch.manual_seed(seed)
    out = model(input_ids=torch.randint(0, VOCAB, (1, length)), use_cache=True, output_hidden_states=True)
    return out.past_key_values, out.hidden_states[-1][:, -1:, :]


class CandidateTest(unittest.TestCase):
    def test_dropout_is_restored(self):
        model, proj = tiny(), projection()
        with mc_dropout((model, proj), 0.3):
            drops = [m for m in list(model.modules()) + list(proj.modules()) if isinstance(m, nn.Dropout)]
            self.assertTrue(drops and all(m.training and m.p == 0.3 for m in drops))
        self.assertTrue(all(not m.training and m.p == 0.1 for m in drops))

    def test_rate_zero_reruns_are_the_own_step(self):
        model, proj = tiny(), projection()
        cache, latent = asked(model)
        with torch.no_grad():
            own, own_cache = step(model, proj, latent, cache, 5)
            reruns, rerun_cache = sample_step(model, proj, latent, cache, 5, 3, 0.0)
            noisy, _ = sample_step(model, proj, latent, cache, 5, 3, 0.5)
        for row in range(3):
            torch.testing.assert_close(reruns[row:row + 1], own, rtol=0, atol=1e-6)
            for (a, b), (c, d) in zip(pick(rerun_cache, row), own_cache):
                torch.testing.assert_close(a, c, rtol=0, atol=1e-6)
                torch.testing.assert_close(b, d, rtol=0, atol=1e-6)
        self.assertGreater(float((noisy[0] - noisy[1]).abs().max()), 1e-4)  # each row draws its own masks

    def test_batched_readout_matches_rows(self):
        model = tiny()
        codi = SimpleNamespace(eot_id=VOCAB - 1, max_new_tokens=5, vocab_size=VOCAB,
                               tokenizer=SimpleNamespace(eos_token_id=3))
        first, _ = asked(model, seed=1)
        second, _ = asked(model, seed=2)
        both = tuple((torch.cat([a, c]), torch.cat([b, d])) for (a, b), (c, d) in zip(first, second))
        with torch.no_grad():
            batched = readout(model, codi, both, 5, 2)
            alone = [readout(model, codi, first, 5, 1)[0], readout(model, codi, second, 5, 1)[0]]
            copies = readout(model, codi, expand(first, 3), 5, 3)
        self.assertEqual(batched, alone)
        self.assertEqual(copies, [alone[0]] * 3)

    def test_read_gives_confidence(self):
        decoder = tiny(seed=3)
        ids, logprobs = read(decoder, decoder.get_input_embeddings(), torch.randn(4, WIDTH), eos=7, max_tokens=4)
        self.assertEqual(len(ids), 4)
        self.assertTrue(all(len(x) <= 4 and 7 not in x for x in ids))
        self.assertTrue(all(lp <= 0.0 for lp in logprobs))


class RuleTest(unittest.TestCase):
    def test_features(self):
        known = numbers("Tom has 3 boxes of 12 eggs.")
        good = features(" <<3*12=36>>", -0.2, known)
        self.assertEqual((good["valid"], good["grounded"], good["result"]), (True, True, 36.0))
        self.assertFalse(features("<<3*12=35>>", -0.2, known)["valid"])
        self.assertFalse(features("<<5*12=60>>", -0.2, known)["grounded"])  # 5 is in neither the question nor earlier steps
        self.assertFalse(features("no step here", -0.2, known)["parsed"])

    def test_choose(self):
        def f(valid, grounded, logprob):
            return {"valid": valid, "grounded": grounded, "logprob": logprob}
        self.assertEqual(choose([f(True, True, -3.0), f(True, True, -0.1)]), 0)  # candidate 0 keeps a tie
        self.assertEqual(choose([f(False, True, -0.1), f(True, False, -2.0), f(True, False, -1.0)]), 2)
        self.assertEqual(choose([f(True, False, -0.1), f(True, True, -2.0)]), 1)


class ScriptsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        os.environ["HF_HUB_OFFLINE"] = "1"
        try:
            FakeCODI()
        except OSError as error:
            raise unittest.SkipTest(f"GPT-2 tokenizer not cached: {error}")
        cls.saved = (codi_module.CODIGPT2, looped.load_step_decoder)
        codi_module.CODIGPT2, looped.load_step_decoder = FakeCODI, fake_decoder

    @classmethod
    def tearDownClass(cls):
        codi_module.CODIGPT2, looped.load_step_decoder = cls.saved

    def run_script(self, script, argv):
        old, out = sys.argv, io.StringIO()
        sys.argv = [script] + [str(a) for a in argv]
        try:
            with contextlib.redirect_stdout(out):
                runpy.run_path(str(ROOT / "scripts/step_select" / script), run_name="__main__")
        finally:
            sys.argv = old
        return out.getvalue()

    def test_eval_then_report(self):
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            (folder / "base").mkdir()
            common = ["--checkpoint", folder / "base", "--dataset", "gsm8k_valid", "--limit", 3, "--device", "cpu",
                      "--chains", 5, "--candidates", 3, "--depth", 3, "--decoder-tokens", 4]
            self.run_script("eval.py", common + ["--output", folder / "zero.jsonl", "--p", 0.0])
            records = [json.loads(x) for x in (folder / "zero.jsonl").read_text().splitlines()]
            self.assertEqual(len(records), 3)
            for r in records:  # at rate 0 every sampled path is the model's own
                self.assertEqual(len(r["chains"]), 5)
                self.assertEqual([len(b) for b in r["branches"]], [2, 2, 2])
                self.assertEqual(len(r["greedy"]["steps"]), 4)
                self.assertTrue(all(c["prediction"] == r["greedy"]["prediction"] for c in r["chains"]))
                self.assertTrue(all(b["prediction"] == r["greedy"]["prediction"] for bs in r["branches"] for b in bs))
                self.assertEqual(r["rule"]["chosen"], [0, 0, 0])
                self.assertEqual(r["rule"]["text"], r["greedy"]["text"])

            self.run_script("eval.py", common + ["--output", folder / "noisy.jsonl", "--p", 0.5, "--seed", 2])
            reference = folder / "reference.jsonl"
            reference.write_text("\n".join(json.dumps({"id": r["id"], "arms": {"plain": {"text": r["greedy"]["text"]}}})
                                           for r in records))
            text = self.run_script("report.py", [folder / "zero.jsonl", "--reference", reference])
            self.assertIn("identity: greedy gives the checkpoint's own answer on 3/3 (PASS)", text)
            self.assertIn("STOP:", text)  # no room at rate 0
            text = self.run_script("report.py", [folder / "zero.jsonl", folder / "noisy.jsonl"])
            self.assertIn("Pooled over 2 run(s)", text)

    def test_report_verdicts(self):
        def feat(valid, correct=None):
            out = {"text": "", "logprob": -0.5, "parsed": valid, "valid": valid, "grounded": valid, "result": None}
            return out if correct is None else {**out, "prediction": "1" if correct else "0", "correct": correct}

        def records(informative):
            out = []
            for i in range(20):
                right = i >= 10  # greedy is wrong on the first ten; one rerun of step 1 fixes each of them
                first = [feat(informative, True), feat(False, False), feat(False, False)]
                rest = [[feat(False, right)] * 3 for _ in range(5)]
                out.append({"id": i, "gold": "1", "rationale_steps": 2,
                            "greedy": {"text": "", "prediction": "1" if right else "0", "correct": right,
                                       "steps": [feat(False)] * 7},
                            "chains": [{"prediction": "1" if right else "0", "correct": right}] * 4,
                            "branches": [first if not right else [feat(False, True)] * 3] + rest,
                            "rule": {"prediction": "", "correct": right or informative, "chosen": [0] * 6}})
            return out

        with tempfile.TemporaryDirectory() as folder:
            for informative, verdict in ((True, "GO:"), (False, "STOP:")):
                path = Path(folder) / f"{informative}.jsonl"
                path.write_text("\n".join(json.dumps(r) for r in records(informative)))
                text = self.run_script("report.py", [path])
                self.assertIn("choosing per step: greedy or any of 18 one-step branches 20 (+10 over greedy)", text)
                self.assertIn(verdict, text)
            self.assertIn("signal in 10 mixed sets: the fixed rule picks a right candidate 0 times", text)


if __name__ == "__main__":
    unittest.main()
