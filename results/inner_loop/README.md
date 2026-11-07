# Inner loop (idea A): results

Idea A loops GPT-2 layers 4–7 inside each of SIM-CoT GPT-2's six latent steps, adding no cache entries. The plan is in
`docs/brainstorm/INNER_LOOP_QUICK_REFERENCE.md`, and the code in `scripts/inner_loop/`. Both stages ran from
`notebooks/DSA_Inner_Loop_Colab.ipynb` at commit `ddaab12`, on the 500 GSM8K validation questions with a single seed (7).
`report.txt` in each stage folder is the output of `scripts/inner_loop/report.py` on that folder.

## Stage 0 (T4, evaluation only): the identity check passes

- With zero new parameters, the loop reproduces both checkpoints on 500/500 questions with zero latent drift: SIM-CoT
  scores 204 and CODI 206 at k = 1, 2 and 4.
- Training-free loops don't help:
  - a damped re-application of the block scores −1 to −7 on SIM-CoT and −1 on CODI;
  - re-running the whole step scores −53 to −65 on SIM-CoT and −162 on CODI.
- SIM-CoT's step decoder parses every latent. Exact gold steps fall from 199/494 at slot 0 to 21/313 at slot 2, and then
  to about 0.
- Recomputing the arithmetic in a decoded step fixes 0 wrong answers.
- Timing: 148 s per 100 updates for `ft6` and 167 s for `loop`.

## Stage 1 (G4, batch 32, 3,072 updates per arm): STOP, no gain

| Arm | Correct /500 | vs SIM-CoT base 204 (paired sign test) |
|---|---|---|
| `ft6` | 199 | −5 (+17/−22, p=0.52) |
| `ft12` at 6 steps / at 12 steps | 196 / 201 | −8 (p=0.32) / −3 (p=0.75) |
| `loop` at k = 1 / 2 / 3 / 4 / 8 | 200 / 201 / 201 / 197 / 194 | k=3: −3 (+18/−21, p=0.75) |
| `loop_k1` (the loop's LoRA on pass 1, no loop) | 200 | −4 (p=0.60) |

- The preregistered rule at k = 3 required:
  - at least +10 over the better of `ft6` and `ft12`@12;
  - at least +5 over k = 1;
  - a gate clearly above zero.
- The result was +0 over `ft12`@12 (+13/−13), +1 over k = 1 and +1 over `loop_k1`, with a gate size of 0.013. The verdict
  is STOP.
- Training found no use for the loop:
  - the mean |gate| reached 0.013 by update 300 and stayed there for the remaining 2,700 updates, with no weight decay
    on it;
  - every arm saw the same batches in the same order, and `loop`'s answer loss equals `ft6`'s on matched batches at
    every k, within ±0.01 on a loss of about 0.18.
- Every fine-tune made the latents easier to read back: exact steps rose from 100 to about 118 at slot 1 and from 21 to
  about 40 at slot 2. Accuracy did not rise.
- Training at 12 steps removes the drift past 6 steps: `ft12`@12 scores 201 against the base's 185 at 12 steps
  (+30/−14, p=0.023). It does not beat 6 steps.
- Each arm trained in 7–11 minutes, about 14 s per 100 updates for `ft6`. The whole stage took about 50 minutes.

## Stage 1b (G4, batch 128, 768 updates per arm, seed 7, commit `6db4f35`): the recipe check and KK-LSE

Stage 1's `ft6` scored 16 below the main line's answer-only `ft6` (199 vs 215, p=0.011). Stage 1b asks two questions:
- whether that gap, and the loop's null, came from the step loss;
- whether idea F's KK-LSE bias helps the readout at 6 steps.

| Arm | Correct /500 | Notes |
|---|---|---|
| SIM-CoT base | 204 | |
| `ft6_step` (answer loss + step loss) | 209 | |
| `ft6_ce` (answer loss only) | 201 | vs the base −3 (p=0.71); the control for the two arms below |
| `loop_ce` at k = 1 / 2 / 3 / 4–8 | 206 / 208 / 210 / 210 | k=3 vs `ft6_ce`: +9 (+16/−7, p=0.093); k=3 vs k=1: +4 (+4/−0, p=0.125) |
| `kk_ce`, bias on / off at test | 204 / 203 | vs `ft6_ce`: +3 (p=0.58); bias on vs off: +1 (1 answer of 500 changes) |

- **Recipe:** the step loss did not cost accuracy. `ft6_step` beat `ft6_ce` by 8 (p=0.19).
  - The ft6 fine-tunes so far score 199, 201, 209 and 215 across runs.
  - A single-seed gap of about 8 between runs is therefore within run-to-run noise.
- **Loop:** AMBIGUOUS under the Stage 1 thresholds.
  - At k=3 the four answers that change from k=1 are all fixes of an arithmetic slip, and accuracy holds at 210 through
    k=8.
  - Most of the +9 is already there at k=1 (206 vs 201).
  - It is +1 over `ft6_step` (+17/−16).
  - The gate peaked at 0.011 and ended at 0.009, and the answer loss on matched batches is flat.
- **KK-LSE:** STOP. α reached a mean |α| of 0.026 (max 0.11), and switching the bias off at test changes 1 answer of 500.
- Each arm trained in about 3 minutes, and the whole run took about 30 minutes.

### Decision rule for a second seed

This rule was written on 6 November, before seed 8 ran. Seed 8 reruns Stage 1b with `SEED = 8`, which changes the
dropout and the sampled problems. The loop is **GO** only if all three hold:
1. Pooled over seeds 7 and 8, `loop_ce`@k3 beats `ft6_ce` by at least +7 per seed on average, with p < 0.05 on the
   pooled paired sign test (1,000 question-runs).
2. k3 is at least k1 within `loop_ce` in each seed.
3. Pooled, `loop_ce`@k3 is at least `ft6_step`.

Anything else is **STOP**: idea A is written up as a null.
