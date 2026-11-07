"""Checks of the inner loop (idea A, docs/brainstorm/INNER_LOOP_QUICK_REFERENCE.md) on tiny random GPT-2s: the step equals
GPT-2's own forward, zero loop parameters reproduce it at every k, the cache grows by one entry per step, the
step-decoder helpers, and idea F's KK-LSE bias."""
import math
import sys
import unittest
from pathlib import Path

import torch
from transformers import GPT2Config, GPT2LMHeadModel

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts/inner_loop"))

from kk import KKBias, load_kk, multiplicity_bias, save_kk  # noqa: E402
from looped import (LoopParams, STEP_END, STEP_STARTS, arithmetic, decode_steps, latent_step, load_loop,  # noqa: E402
                    parse_step, save_loop, step_loss, step_targets)

WIDTH, VOCAB = 32, 60


def tiny(seed=0, layers=12):
    torch.manual_seed(seed)
    config = GPT2Config(n_layer=layers, n_embd=WIDTH, n_head=4, vocab_size=VOCAB, n_positions=64,
                        resid_pdrop=0.0, embd_pdrop=0.0, attn_pdrop=0.0)
    config._attn_implementation = "eager"
    return GPT2LMHeadModel(config).eval()


def asked(model, batch=2):
    """A left-padded question batch: its cache, mask, next position and first latent-like input."""
    torch.manual_seed(1)
    lengths = [5, 3][:batch]
    width = max(lengths)
    ids = torch.randint(0, VOCAB, (batch, width))
    mask = torch.tensor([[0] * (width - n) + [1] * n for n in lengths])
    positions = (mask.cumsum(1) - 1).clamp(min=0)
    out = model(input_ids=ids, attention_mask=mask, position_ids=positions, use_cache=True, output_hidden_states=True)
    return out.past_key_values, mask, positions[:, -1:] + 1, out.hidden_states[-1][:, -1:, :]


def grown(mask):
    return torch.cat([mask, torch.ones_like(mask[:, :1])], 1)


class LatentStepTest(unittest.TestCase):
    def test_no_loop_matches_gpt2_forward(self):
        model = tiny()
        cache, mask, position, latent = asked(model)
        mask = grown(mask)
        with torch.no_grad():
            ref = model(inputs_embeds=latent, past_key_values=cache, attention_mask=mask, position_ids=position,
                        use_cache=True, output_hidden_states=True)
            hidden, presents = latent_step(model, None, latent, cache, position, mask=mask)
        torch.testing.assert_close(hidden, ref.hidden_states[-1], rtol=0, atol=1e-6)
        for (k, v), (rk, rv) in zip(presents, ref.past_key_values):
            torch.testing.assert_close(k, rk, rtol=0, atol=1e-6)
            torch.testing.assert_close(v, rv, rtol=0, atol=1e-6)

    def test_zero_loop_is_identity_at_every_k(self):
        model = tiny()
        loop = LoopParams(model, rank=4)
        cache, mask, position, latent = asked(model)
        mask = grown(mask)
        with torch.no_grad():
            ref, ref_cache = latent_step(model, None, latent, cache, position, mask=mask)
            for k in (1, 2, 4):
                hidden, presents = latent_step(model, loop, latent, cache, position, k=k, mask=mask)
                torch.testing.assert_close(hidden, ref, rtol=0, atol=1e-6)
                for (a, b), (c, d) in zip(presents, ref_cache):
                    torch.testing.assert_close(a, c, rtol=0, atol=1e-6)
                    torch.testing.assert_close(b, d, rtol=0, atol=1e-6)

    def test_open_gate_changes_output_not_cache_size(self):
        model = tiny()
        loop = LoopParams(model, rank=4).fill(0.5)
        cache, mask, position, latent = asked(model)
        mask = grown(mask)
        with torch.no_grad():
            ref, ref_cache = latent_step(model, None, latent, cache, position, mask=mask)
            hidden, first = latent_step(model, loop, latent, cache, position, k=3, mask=mask)
            _, last = latent_step(model, loop, latent, cache, position, k=3, mask=mask, kv="last")
            same_k1, _ = latent_step(model, loop, latent, cache, position, k=1, mask=mask)
        self.assertGreater(float((hidden - ref).abs().max()), 1e-4)
        torch.testing.assert_close(same_k1, ref, rtol=0, atol=1e-6)
        for (a, _), (b, _), (c, _) in zip(first, last, ref_cache):
            self.assertEqual(a.shape, c.shape)  # one new entry per layer, whatever k is
            self.assertEqual(b.shape, c.shape)
        torch.testing.assert_close(first[4][0], ref_cache[4][0], rtol=0, atol=1e-6)  # pass 1's keys are kept
        self.assertGreater(float((last[4][0] - ref_cache[4][0]).abs().max()), 1e-4)
        torch.testing.assert_close(first[3][0], ref_cache[3][0], rtol=0, atol=1e-6)  # layers before the block untouched

    def test_lora_only_on_extra_passes_unless_matched(self):
        model = tiny()
        cache, mask, position, latent = asked(model)
        mask = grown(mask)
        plain = LoopParams(model, rank=4)
        matched = LoopParams(model, rank=4, lora_on_first=True)
        with torch.no_grad():
            for b in list(plain.lora_b) + list(matched.lora_b):
                b.normal_()
            ref, _ = latent_step(model, None, latent, cache, position, mask=mask)
            loop_k1, _ = latent_step(model, plain, latent, cache, position, k=1, mask=mask)
            matched_k1, _ = latent_step(model, matched, latent, cache, position, k=1, mask=mask)
            after, _ = latent_step(model, None, latent, cache, position, mask=mask)
        torch.testing.assert_close(loop_k1, ref, rtol=0, atol=1e-6)
        self.assertGreater(float((matched_k1 - ref).abs().max()), 1e-4)
        torch.testing.assert_close(after, ref, rtol=0, atol=1e-6)  # the LoRA switches off after the step

    def test_gradient_reaches_gate_from_zero(self):
        model = tiny()
        loop = LoopParams(model, rank=4)
        cache, mask, position, latent = asked(model)
        hidden, _ = latent_step(model, loop, latent, cache, position, k=2, mask=grown(mask))
        hidden.pow(2).sum().backward()
        self.assertGreater(float(loop.gate.grad.abs().sum()), 0.0)

    def test_save_and_load(self):
        import tempfile
        model = tiny()
        loop = LoopParams(model, rank=4).fill(0.3)
        with tempfile.TemporaryDirectory() as folder:
            save_loop(loop, Path(folder) / "loop.pt")
            loop.detach_hooks()
            again = load_loop(model, Path(folder) / "loop.pt")
        torch.testing.assert_close(again.gate, loop.gate)
        self.assertEqual(again.config(), loop.config())


def run_steps(model, cache, mask, position, latent, n=3):
    """n latent steps with no loop, each fed the last hidden state: their hidden states [batch, n, width]."""
    out = []
    for step in range(n):
        mask = grown(mask)
        latent, cache = latent_step(model, None, latent, cache, position + step, mask=mask)
        out.append(latent)
    return torch.cat(out, 1)


class KKTest(unittest.TestCase):
    def test_copies_weigh_as_one(self):
        torch.manual_seed(0)
        a, c = torch.randn(8), torch.randn(8)
        c = c - (c @ a) / (a @ a) * a  # orthogonal to a
        keys = torch.stack([a, a, a, c])
        bias = multiplicity_bias(keys[None, None], torch.ones(1), beta=14.0)[0, 0]
        torch.testing.assert_close(bias[:3], torch.full((3,), -math.log(3)), rtol=0, atol=1e-5)
        self.assertGreater(float(bias[3]), -1e-5)  # a distinct key is left alone
        query = torch.randn(8)
        weights = (keys @ query + bias).softmax(-1)
        once = torch.stack([a @ query, c @ query]).softmax(-1)
        torch.testing.assert_close(weights[:3].sum(), once[0], rtol=0, atol=1e-5)  # three copies weigh as one

    def test_zero_alpha_is_identity(self):
        model = tiny()
        cache, mask, position, latent = asked(model)
        with torch.no_grad():
            ref = run_steps(model, cache, mask, position, latent)
            kk = KKBias(model)
            kk.start, kk.count = cache[0][0].shape[-2], 3
            same = run_steps(model, cache, mask, position, latent)
        torch.testing.assert_close(same, ref, rtol=0, atol=0)

    def test_bias_touches_only_latents(self):
        model = tiny()
        cache, mask, position, latent = asked(model)
        start = cache[0][0].shape[-2]
        with torch.no_grad():
            ref = run_steps(model, cache, mask, position, latent)
            kk = KKBias(model, beta=1.0)
            kk.alpha.fill_(1.0)
            kk.start, kk.count = start, 3
            _, _, _, question = asked(model)  # the question pass has no latents to bias
            moved = run_steps(model, cache, mask, position, latent)
            kk.detach_hooks()
            again = run_steps(model, cache, mask, position, latent)
        torch.testing.assert_close(question, latent, rtol=0, atol=0)
        torch.testing.assert_close(moved[:, :1], ref[:, :1], rtol=0, atol=0)  # one cached latent: nothing to count
        self.assertGreater(float((moved[:, 1:] - ref[:, 1:]).abs().max()), 1e-4)
        torch.testing.assert_close(again, ref, rtol=0, atol=0)  # detach_hooks restores GPT-2's attention

    def test_gradient_reaches_alpha_from_zero_and_save_load(self):
        import tempfile
        model = tiny()
        cache, mask, position, latent = asked(model)
        kk = KKBias(model, beta=1.0)
        kk.start, kk.count = cache[0][0].shape[-2], 3
        run_steps(model, cache, mask, position, latent).pow(2).sum().backward()
        self.assertGreater(float(kk.alpha.grad.abs().sum()), 0.0)
        with torch.no_grad():
            kk.alpha.fill_(0.3)
        with tempfile.TemporaryDirectory() as folder:
            save_kk(kk, Path(folder) / "kk.pt")
            kk.detach_hooks()
            again = load_kk(model, Path(folder) / "kk.pt")
        torch.testing.assert_close(again.alpha, kk.alpha.detach())
        self.assertEqual(again.beta, 1.0)


class FakeTokenizer:
    eos_token_id = 50

    def encode(self, text):
        table = {"<<": STEP_STARTS[0], " <<": STEP_STARTS[1], ">>": STEP_END}
        out = []
        for piece in text.replace(" <<", "|S").replace("<<", "|F").replace(">>", "|E").split("|"):
            if not piece:
                continue
            head, rest = piece[0], piece[1:]
            out.append({"S": table[" <<"], "F": table["<<"], "E": table[">>"]}.get(head, 7))
            out.extend(7 for _ in rest)
        return out


class StepHelpersTest(unittest.TestCase):
    def test_step_targets_merge_and_pad(self):
        tok = FakeTokenizer()
        steps = ["<<1+1=2>>", "<<2*3=6>>", "<<6-1=5>>"]
        targets = step_targets(tok, steps, slots=5)
        self.assertEqual(len(targets), 5)
        self.assertEqual(targets[0][0], STEP_STARTS[0])
        self.assertEqual(targets[1][0], STEP_STARTS[1])
        self.assertTrue(all(t[-1] == tok.eos_token_id for t in targets[:3]))
        self.assertEqual(targets[3:], [None, None])
        merged = step_targets(tok, steps, slots=2)
        self.assertEqual(len(merged), 2)
        self.assertEqual(merged[1].count(STEP_END), 2)  # steps 2 and 3 share the last slot
        self.assertEqual(merged[1][-1], tok.eos_token_id)

    def test_arithmetic_and_parse(self):
        self.assertEqual(arithmetic("2/.5"), 4.0)
        self.assertEqual(arithmetic("600*30/100"), 180.0)
        self.assertEqual(arithmetic("-3+1,000"), 997.0)
        self.assertIsNone(arithmetic("__import__('os')"))
        self.assertIsNone(arithmetic("1/0"))
        self.assertEqual(parse_step(" <<4-2=2>> junk"), ("4-2", "2"))
        self.assertIsNone(parse_step("no step"))

    def test_step_loss_and_decode_run(self):
        model, decoder = tiny(), tiny(seed=3, layers=2)
        embed = model.get_input_embeddings()
        latents = torch.randn(2, 3, WIDTH, requires_grad=True)
        targets = [[[5, 6, 50], None, [7, 50]], [None, None, None]]
        loss = step_loss(decoder, embed, latents, targets, pad=59)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertGreater(float(latents.grad[0, 0].abs().sum()), 0.0)
        self.assertEqual(float(latents.grad[1].abs().sum()), 0.0)  # rows without targets get no gradient
        self.assertIsNone(step_loss(decoder, embed, latents, [[None] * 3, [None] * 3], pad=59))
        decoded = decode_steps(decoder, embed, latents.detach()[0], eos=50, max_tokens=4)
        self.assertEqual(len(decoded), 3)
        self.assertTrue(all(len(d) <= 4 for d in decoded))


if __name__ == "__main__":
    unittest.main()
