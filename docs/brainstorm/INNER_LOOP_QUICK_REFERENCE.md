# Inner loop (A) and recursive attention (F): quick reference

A short summary from the dsa-brainstorm session, 5 November 2026. The full reasoning, numbers and citations are in
[Looped steps decoding and recursive attention](Looped%20steps%20decoding%20and%20recursive%20attention.md). The source
notes are in `research_notes/Looped steps decoding and recursive attention/`.

## A and F in brief

**A, an inner loop per latent step.**
- Keep CODI's six sequential latent steps. Inside each step, run GPT-2's middle layers (4–7) k times instead of once.
- The extra compute adds no cache entries and no positions, so it avoids both losses we measured past depth: dilution
  at GPT-2 and RoPE answer distance at 1B.
- New parameters start at zero, so every k reproduces the released checkpoint at the start.
- Novelty: the mechanism is close to LOTUS and Think-at-Hard. The open question is depth per step against more steps,
  at equal compute.
- Expected gain from a short fine-tune: 0 to +2 points. This is a judgement, not a measurement.

**F, recursive attention: a key-to-key bias.**
- A score computed from one entry at a time can't cancel duplicate entries, because identical copies get identical
  scores. This plausibly explains why the gate failed on CODI.
- KK-LSE does cancel them. Per head, b_j = −α·log Σ_l exp(β(cos(k_j, k_l) − 1)) over the latent entries. It is minus
  the log-normalizer of a key-to-key attention, so one attention pass feeds the other. As built (`kk.py`), it adds 144
  scalars (α per layer and head, β fixed at 14), and α = 0 gives the base model.
- On its own it can only hold the 6-step score. It has a real job only beside a loop that keeps one cache entry per
  iteration (the "duo-causal" cache from Think-at-Hard), because those entries become near-copies as the iterations
  converge.

**D, decode and re-inject,** is demoted to eval-only diagnostics: decoder reliability, gold-step injection, and a
calculator test that has to beat gpt2-sft given the same calculator.

## Plan

**Stage 0: eval only, about 1–2 T4-hours.**
- Identity: the loop code with its new parameters at zero reproduces 500/500 answers.
- A training-free overwrite probe: rerun a step r times at the same position.
- SIM-CoT decoder reliability, and the wrong answers split into readout, arithmetic and plan errors.
- Gold-step injection (D).
- A 100-update timing run.

**Stage 1: a four-arm pilot on SIM-CoT GPT-2, about 20 Colab units.**
- Arms: ft6; ft12, matched in compute to a 4-layer loop run 4 times; the loop, with k from 1–4; and a parameter-matched
  arm at k=1.
- All arms share the objective (answer CE plus the step-decoder loss), the examples, the learning rate and the seed.
- **Go:** loop@k=3 beats the better of ft6 and ft12 by at least 10/500, k=3 beats k=1 within the loop model by at
  least 5, and the gate moves off zero.
- **Stop:** a gain of +4 or less, a gate stuck at zero, or no rise from k=1 to k=3.
- **Result (6 November): STOP.** The loop at k=3 scored 201, tying ft12@12 (201) and below the base (204). Its gate
  settled at 0.013. Our ft6 (199) also scored 16 below the main line's answer-only ft6 (215, p=0.011).
  `results/inner_loop/`.

**Stage 1b: the recipe check and KK-LSE at 6 steps, at batch 128.**
- `ft6_step` is Stage 1's recipe. `ft6_ce` and `loop_ce` train on the answer loss alone: is the null the loop's or the
  recipe's?
- `kk_ce` is `ft6_ce` plus KK-LSE, scored with the bias on and off. Does the bias help the readout at 6 steps, where there
  are no copies to cancel?
- The thresholds are the same as Stage 1's (+10 go, +5 to +9 ambiguous). The second check is k=3 against k=1 for the
  loop, and the bias on against off for KK-LSE.

**Stage 2: confirm and extend.**
- Three seeds on the 1,319-question test set, needing a mean gain of at least +3 with a paired CI that excludes 0.
- Then CODI GPT-2, the duo-causal loop with and without KK-LSE (A+F), and the LLaMA-1B port with base weights frozen.

Publishable either way: a win is the first inner loop to beat more steps at matched compute; a loss is a matched 2×2
null plus the first decoder-reliability numbers.

## How do CODI, LOTUS and A get from the question to the answer?

**CODI (what we have now).** Thoughts are made one at a time, each in one pass through all 12 layers.
```
Q  <bot>  →  [12 layers] → h1 → proj → z1
              [12 layers] → h2 → proj → z2      each step appends 1 cache entry
              ...  6 times
           <eot> → decode answer
```
To think more you have to add steps, and every step adds an entry. That is where the dilution comes from.

**LOTUS.** A fixed block of thought slots is refined in parallel.
```
Q (encoded once)  +  [slot1 slot2 ... slotN]
   pass 1: whole model over all slots in parallel
   pass 2: whole model again, slots overwritten  (input re-added each pass)
   ...  R = 6 passes
<eot> → decode
```
It revisits every thought on every pass, and slots are replaced, not appended.

**A.** CODI's step-by-step chain, with more thinking inside each step.
```
Q  <bot>  →  layers 0–3  (once)
              layers 4–7  (loop k times, same position)
              layers 8–11 (once) → h1 → proj → z1     still 1 cache entry
              ...next step, same thing...  6 times
           <eot> → decode
```
- **Six thoughts, six cache entries,** each computed more deeply. At k=4 a thought gets 24 layers instead of 12.
- **Why the middle:** the first layers read the input and the last layers map to the space the projection expects, so
  both stay as CODI trained them. ETD and McLeish et al. found middle-block loops beat whole-model loops at equal
  compute. It is also cheaper.
- **Safe start:** extra passes enter through a gate initialized to zero, so every k gives CODI's answers at first.
- **Stable passes:** the block's input is re-added on every pass, as in LOTUS's h(t) = f(E + h(t−1)).
- **Training:** k is random, 1 to 4. At test time sweep k from 1 to 8 and look for accuracy rising with k inside one
  model.
- **New compared with LOTUS:** LOTUS revisits all thoughts in parallel, while A deepens each thought as it is made. The
  untested comparison is 6 steps × 2 passes against 12 steps × 1 pass, which both use 144 layer evaluations.
- **Where F comes in:** in this version there are no extra entries, so F has nothing to do. In the Stage 2 variant
  every pass keeps its own cache entry, the entries become near-copies, and KK-LSE removes the copies.

## What does each step feed forward in A, and where do LOTUS's slots come from?

Neither re-feeds the question. Both process it once and read it from the KV cache afterwards.

**A, like CODI, feeds one new vector per step.**
```
start:   [Q ... <bot>]  → full pass once, cached        → proj(h_bot) = z1
step 1:  feed z1 alone  → L0–3 → L4–7 ×k → L8–11 → h1   → proj(h1)   = z2
step 2:  feed z2 alone  → same                   → h2   → proj(h2)   = z3
...      6 steps
end:     feed <eot> → decode the answer token by token
```
- Each step is a single new position. It attends to everything in the cache (the question, `<bot>`, earlier latents),
  and it does so on every pass of layers 4–7.
- After 6 steps the cache holds `Q, <bot>, z1…z6`: one entry per layer per step, exactly as in CODI.

**LOTUS's slots are placeholders written into the prompt.** This is checked against `scripts/lotus.py` and
`args/gsm8k_lotus_gpt2.yaml`.
```
Q "\n" <start> <latent> × 78 <end>          (GPT-2 config: 6 blocks × 13 slots)
```
1. All 78 slots start as the same learned `<latent>` embedding. The question is processed once and cached.
2. Pass 0 is one ordinary forward pass over all slots at once, the way a prompt is read. A causal mask means slot j
   sees the question and the slots before it.
3. In each of the R = 6 loop passes, a slot's new input is the original `<latent>` embedding plus that same slot's
   last-layer hidden state from the previous pass. There is no projection MLP. All slots go through all 12 layers
   again, and their cache entries are overwritten.
4. At pass t, block t (13 slots) is trained through the LM head to decode rationale step t. That per-pass supervision
   is what makes block t become step t.
5. After the last pass, the answer is decoded after `<end>` from the final cache.

| | CODI | A | LOTUS |
|---|---|---|---|
| Where a thought comes from | the previous thought, one step at a time | same as CODI | blank placeholders, refined together |
| Compute per thought | 12 layers, once | 4 + 4k + 4 layers, once | 12 layers × R passes |
| Revisits earlier thoughts? | no | no | yes, every pass |
| Cache entries | grow with steps (dilution past 6) | grow with steps, not with k | fixed (78), whatever R is |

## Code

| File | What it does |
|---|---|
| `scripts/inner_loop/looped.py` | The looped step (`latent_step`), the zero-initialized loop parameters, and SIM-CoT's step decoder and step loss |
| `scripts/inner_loop/kk.py` | F's KK-LSE bias on the latent entries, patched into GPT-2's eager attention, with α starting at zero |
| `scripts/inner_loop/eval.py` | Evaluation arms: `plain`, `nokk`, `loopK`, `dampKxG`, `rerunR`, with `@D` for depth; `--decoder` reads every latent back |
| `scripts/inner_loop/train.py` | Arms `ft6`, `ft12`, `loop` and `loop_k1`, on answer CE plus the step loss (`--step-weight 0` for the answer alone), with `--kk` for KK-LSE; batch 128 by default |
| `scripts/inner_loop/report.py` | Stage 0's identity, decoder and error checks; the go/no-go rules of Stages 1 and 1b, with exact sign tests |
| `notebooks/DSA_Inner_Loop_Colab.ipynb` | Colab notebook with `STAGE = 0`, `1` or `"1b"` |
| `tests/test_inner_loop.py`, `tests/test_inner_loop_scripts.py` | 18 tests on tiny random GPT-2s, including end-to-end train → eval → report |

## Sources

- [PCCoT](https://arxiv.org/html/2506.18582v2): 49.48 on GPT-2. The 48.24 is its 12-latent CODI-style baseline.
- [LOTUS](https://arxiv.org/html/2606.31779): 44.1 on GPT-2, against 42.0 for CODI in the same table. The closest
  competitor.
- [Think-at-Hard](https://arxiv.org/abs/2511.08577): looping every token lost about 6 points; the duo-causal cache.
- [Penelope](https://arxiv.org/abs/2607.25915): a loop over one interval of a pretrained decoder. Read only at abstract
  level; read it before claiming novelty.
- [SpiralThinker](https://aclanthology.org/2026.findings-acl.1605.pdf): alternates latent blocks with text steps (D).
- [CODI](https://arxiv.org/html/2502.21074): decoded-text CoT scores 34.0–35.7 against 43.7 for latents.
- [SIM-CoT](https://arxiv.org/html/2509.20317): the step decoder, shipped in the GPT-2 checkpoint.
- [ToMe](https://arxiv.org/abs/2210.09461): the +log s merged-token correction, our identity in reverse.
- [InfiLoop](https://arxiv.org/abs/2610.11570): attention over loop states collapses without decay.
