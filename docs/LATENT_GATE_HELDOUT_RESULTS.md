# Held-out and scale results: soft dedup fixes GPT-2, but at 1B the answer's position is the main cost

Status, 5 November 2026: Parts A and B of the held-out and scale study have run. Part C did not trigger, and the optional
3B model was not run. Amendment 1's B3, re-anchoring the answer at 1B, has also run (commit `48a461d`; see
[below](#amendment-1-b3-re-anchoring-the-answer-at-1b)). Its untriggered Part C is pending.

- **The runs.** One Colab A100 notebook ran everything at commit `837f5df`.
- **The plan.** [PLAN.md](../results/latent_gate/heldout_scale/PLAN.md) was frozen before any run. Its sha256
  (`9de26988…`) still matches.
- **The reports.**
  - [report.txt](../results/latent_gate/heldout_scale/report.txt) applies the plan's rules. It comes from
    `scripts/latent_gate/heldout_report.py`.
  - [explore.txt](../results/latent_gate/heldout_scale/explore.txt) holds the exploratory analyses. It comes from
    `scripts/latent_gate/heldout_explore.py`.
- **Background.** [Stage 1 results](LATENT_GATE_STAGE1_RESULTS.md).

The policies are defined in the plan and in the Stage 1 results doc:
- **`keep_first`:** the answer sees only latent entries 1–6.
- **`dedup_soft`:** each near-copy cluster gets the weight of one entry.
- **`logn`:** every latent entry gets −log(n/6).
- **`shift`:** 6 latent steps, with the answer placed at the positions it would have after 24 steps.

## Checks

- **E1, the batched evaluator.** It passed; the notebook stops on a failure. It also reproduces Stage 1's `gsm8k_valid`
  counts for both GPT-2 models exactly.
- **E2, the LLaMA port.** Checks 1–3 passed on both 1B models.
  - **CODI-1B:** 55.5% on GSM8K test, against the published 55.6%.
  - **SIM-CoT-1B:** 55.7%, against the published 56.1%.
  - The check-3 runs and the held-out runs agree on all 1,319 predictions.
- **E3, the tie fix.** It was unit-tested before the freeze.

## Results on the primary held-out pool

Correct answers out of 2,818: `gsm8k_test` 1,319, `gsm_hard` 1,319 and `multiarith` 180. Per-set and SVAMP rows are in
the report.

| Model | plain 6 | plain 12 | plain 24 | keep_first 24 | dedup_soft 12 | dedup_soft 24 | logn 24 | shift 6 |
|---|---|---|---|---|---|---|---|---|
| SIM-CoT GPT-2 | 799 | 806 | 757 | 802 | 801 | 801 | 771 | 802 |
| CODI GPT-2 | 854 | 810 | 718 | 849 | 840 | 843 | 773 | 849 |
| CODI-1B | **1077** | 1071 | 1027 | 1015 | 1065 | 1029 | 997 | 1015 |
| SIM-CoT-1B | **1075** | 1038 | 976 | 1008 | 1055 | 997 | 825 | 1008 |

**The planned rules.**
- Differences are in accuracy points, with 95% paired CIs.
- D2 and D3 need a win over `plain`@24 at p < 0.05, and a CI on the difference from `plain`@6 whose lower end is above −2.

| Model | D1, drift at 24 | D2, `dedup_soft`@24 | D3, `keep_first`@24 | D4, `shift`@6 | G |
|---|---|---|---|---|---|
| SIM-CoT GPT-2 | −1.5 [−2.3, −0.7], **holds** | **holds**: +1.6 over plain@24; +0.1 [−0.4, +0.6] vs plain@6 | **holds** | +0.1 | fails |
| CODI GPT-2 | −4.8 [−5.8, −3.8], **holds** | **holds**: +4.4; −0.4 [−1.0, +0.3] | **holds** | −0.2 | fails |
| CODI-1B | −1.8 [−2.5, −1.0], **holds** | **fails**: +0.1 (p = 0.92); −1.7 [−2.5, −0.9] | fails | **−2.2** [−3.0, −1.4] | fails |
| SIM-CoT-1B | −3.5 [−4.4, −2.6], **holds** | **fails**: +0.7 (p = 0.07); −2.8 [−3.6, −1.9] | fails | **−2.4** [−3.2, −1.6] | fails |

**The plan's verdict.**
- "The fix generalizes" fails: D1 holds on all four models, and D2 fails on both 1B models.
- "It holds across scale" is not shown.
- G fails everywhere on the primary pool, so no policy turns extra depth into accuracy.
- Part C's second trigger is not met either: 79% (CODI-1B) and 60% (SIM-CoT-1B) of entries 7–12 are near-copies, against a
  bar of under 50%.
- So no 1B training runs.

**SVAMP, the secondary set.** It shows the same pattern, with two exceptions.
- CODI GPT-2 does not drift on SVAMP.
- SIM-CoT GPT-2 gives the only G pass in the study: `dedup_soft`@24 is 415 against `plain`@6's 401 (+1.4 [+0.3, +2.5],
  p = 0.016). It is one secondary-set result among many comparisons, so it is not a claim.

## Why the fix fails at 1B: the answer's position

**The control splits the drop exactly.**
- `shift`@6 and `keep_first`@24 are the same computation: the answer sees latents 1–6 and starts 18 positions later than
  usual. They gave identical text on every question for all four models.
- The loss at 24 steps therefore splits exactly into two parts:
  - **Position:** `shift`@6 − `plain`@6.
  - **Content,** the effect of entries 7–24 at the same answer position: `plain`@24 − `shift`@6.

| Model, primary pool | Total | Position | Content | `dedup_soft`@24 − `shift`@6 |
|---|---|---|---|---|
| SIM-CoT GPT-2 | −1.5 | +0.1 | −1.6 | −0.0 |
| CODI GPT-2 | −4.8 | −0.2 | −4.6 | −0.2 |
| CODI-1B | −1.8 | −2.2 | +0.4 | +0.5 |
| SIM-CoT-1B | −3.5 | −2.4 | −1.1 | −0.4 |

**On GPT-2, the whole loss is content.**
- Its learned absolute position embeddings make the answer's position nearly free.
- The content part is the dilution by near-copy entries that Stage 1 found. Soft dedup and `keep_first` remove it.

**On the 1B LLaMA models, position causes most of the loss.**
- It causes all of CODI-1B's loss and two thirds of SIM-CoT-1B's.
- A policy that only reweights or evicts entries cannot touch it. `dedup_soft`@24 lands within 0.5 points of `shift`@6 on
  every model.
- SIM-CoT-1B also has a smaller content part, which `keep_first` removes: +1.1 over `plain`@24, p = 0.011.

**The attention shares point the same way.**
- These are exploratory, measured on `gsm8k_valid` at the answer's first query.

| Share on entries 1–6 | plain@6 | shift@6 | plain@24 | dedup_soft@24 |
|---|---|---|---|---|
| SIM-CoT GPT-2 | 0.099 | 0.094 | 0.036 | 0.075 |
| CODI GPT-2 | 0.158 | 0.153 | 0.033 | 0.057 |
| CODI-1B | 0.096 | 0.047 | 0.030 | 0.027 |
| SIM-CoT-1B | 0.096 | 0.053 | 0.036 | 0.038 |

- **On LLaMA, the shift alone halves the answer's attention on entries 1–6,** with no extra entries at all. On GPT-2 it
  barely moves.
- **Soft dedup restores much of that attention on GPT-2 but none on LLaMA.**
- This fits RoPE's decay with distance: an answer placed farther from the latents attends to them less, however many
  there are.
- **The latents still repeat at 1B.** 92% (CODI-1B) and 78% (SIM-CoT-1B) of entries 7–24 are near-copies, against 100% on
  GPT-2. So both mechanisms are present at 1B, and the positional one is larger.

## Amendment 1, B3: re-anchoring the answer at 1B

**What was tested.**
- Amendment 1 to PLAN.md was written after Part B and before any B3 run. The rehashed plan's sha256 is `40233c43…`.
- **`anchor`** moves each later near-copy entry to the position of its earliest copy, by rotating its RoPE key. It gives
  every other later entry the next free position, and starts the answer right after the last position used.
- **`dedup_anchor`** does the same after a `dedup_soft` loop, keeping its bias.
- **The rotation is exact.** Check 4 moves every key and the query 18 positions; it passed on both models.
- [amendment_report.txt](../results/latent_gate/heldout_scale/amendment_report.txt) applies the amendment's rules.

**Primary pool, correct out of 2,818:**

| Model | plain 6 | plain 24 | dedup_soft 24 | anchor 24 | dedup_anchor 12 | dedup_anchor 24 |
|---|---|---|---|---|---|---|
| CODI-1B | **1077** | 1027 | 1029 | 1058 | 1065 | 1055 |
| SIM-CoT-1B | **1075** | 976 | 997 | 1011 | 1065 | 1049 |

**The rules.** R1 and R2 use D2's criteria; G2 asks for a win over `plain`@6.

| Model | R1, `dedup_anchor`@24 | R2, `anchor`@24 | G2 |
|---|---|---|---|
| CODI-1B | **holds**: +1.0 [+0.2, +1.8] over plain@24, p = 0.013; −0.8 [−1.3, −0.3] vs plain@6 | **holds**: +1.1, p = 9e−5; −0.7 [−1.3, −0.1] | fails |
| SIM-CoT-1B | **holds**: +2.6 [+1.7, +3.5], p = 2e−8; −0.9 [−1.5, −0.3] | fails: +1.2; −2.3 [−3.0, −1.5] | fails |

**What it shows.**
- **Once the answer is re-anchored, the training-free fix holds at 1B.** With D2 on GPT-2, it now holds on all four
  models within the 2-point margin.
- **It is not free.** At 24 steps it is still 0.8–0.9 points below 6 steps, significant on both models (p ≈ 0.005).
- **No depth beats 6 steps,** so G2 fails. The best deeper result is `dedup_anchor`@12, at −0.4 points on both models.
- **Moving the answer is what helps.** `dedup_anchor`@24 beats `dedup_soft`@24 by +0.9 (CODI-1B) and +1.8 (SIM-CoT-1B)
  points, p ≤ 0.005.
- **The two models need different parts, as the split predicted.**
  - CODI-1B's loss was all position, so re-anchoring alone is enough.
  - SIM-CoT-1B also has a content part, so it needs soft dedup as well.
- **It recovers most of the drift:** 28 of CODI-1B's 50 lost answers and 73 of SIM-CoT-1B's 99.
- **SVAMP agrees.** R1 holds on both models, and `dedup_anchor`@24 is within 0.5 points of `plain`@6.

**Exploratory, on `gsm8k_valid`** (in explore.txt):
- **Few new positions are needed.** At 24 steps, `dedup_anchor` uses 1.4 (CODI-1B) and 4.0 (SIM-CoT-1B) positions past
  entry 6 on average, and none on 21–24% of questions.
- **Attention on entries 1–6 comes back only partly.** It is 0.049 and 0.058 under `dedup_anchor`@24, against 0.096 at 6
  steps and 0.027 / 0.038 under `dedup_soft`@24. The rest of the gap is not explained here.

## The fixed gate (A2)

**SIM-CoT GPT-2: it holds depth but fails its rule.**
- It scores 817 at 12 steps and 813 at 24 (−0.1 [−0.3, −0.0]).
- It does not beat `ft6` at 24: 813 against 806, p = 0.63.
- `ft6` itself barely drifts on held-out data: 816 at 6 steps, 806 at 24.
- The gate keeps exactly entries 1–6 on 190 of 500 `gsm8k_valid` questions. That is a learned `keep_first` again.

**CODI GPT-2: it still fails with the tie fix.**
- It scores 721 at 12 steps and 652 at 24.
- That is below `ft6` at 24 (760; −3.8 [−5.0, −2.7]) and below base `dedup_soft` at 24 (843).
- Its kept sets still follow CODI's two phases: {1, 3, 5, 7, 9, 11}, {1, 2, 4, 6, 8, 10}, and at 24 steps
  {1, 15, 17, 19, 21, 23}.
- **The tie bug did not cause Stage 1's CODI failure.** That caveat is retired: on CODI the gate learns a parity rule.

## What this means for the paper

The two planned headline claims are not supported. All of the following comes from planned runs; only the split and the
attention shares are exploratory analyses of them.

1. **Drift past the trained depth is real.** It holds on untouched data, on four checkpoints, at two scales. D1 holds
   everywhere, with p ≤ 5e-4.
2. **There are two mechanisms, and the position control separates them exactly.**
   - **Dilution by near-copy cache entries.** It dominates at GPT-2 scale, and soft dedup or a keep-first budget removes it
     with no training.
   - **The answer's growing RoPE distance from the latents.** It dominates at 1B, and no content-level policy fixes it.
     Re-anchoring the answer does (Amendment 1).
3. **A training-free repair holds depth on all four models.** That is soft dedup on GPT-2, and soft dedup plus
   re-anchoring at 1B, each within 2 points of the 6-step score.
4. **Nothing tested beats stopping at 6 steps.** At 1B the best deeper result is `dedup_anchor` at 12 steps, −0.4 points on
   both models.
5. **The learned gate is not a contribution.** With the tie bug fixed, it fails again on CODI, and on SIM-CoT it matches
   soft dedup.

The paper is a diagnosis across scale: why latent CoT models lose accuracy past their trained depth. It now comes with a
matching repair that holds depth, though it does not turn depth into accuracy. The position finding comes from the plan's
own control, D4, so it is not a post hoc rescue. The repair for it was designed after D4, so its test is weaker (see the
caveats).

## Caveats

- **SIM-CoT GPT-2 scores 39.3% on GSM8K test, against 42.6% in the SIM-CoT paper.**
  - Check 3 covered only the LLaMA models.
  - CODI GPT-2 scores 42.5%, against CODI's 43.7%.
  - Comparisons within a model are unaffected, but the gap must be reported.
- **The split and the attention shares are exploratory.**
  - D4 was planned, but the plan did not note that `shift`@6 equals `keep_first`@24.
  - The attention shares are measured on `gsm8k_valid` only.
- **Discovery and held-out data disagree on CODI-1B's content part:** −2.6 points on `gsm8k_valid`, +0.4 on the primary
  pool. The held-out pool is the reference.
- **B3 reuses the held-out sets.**
  - Amendment 1 was designed after Part B had shown the position cost on these same sets.
  - Its rules were fixed before B3 ran, but the sets are no longer untouched for this question.
  - A clean confirmation needs a set Part B did not use.
- **One training seed** for the gate arms.
- **`gsm_hard` is `gsm8k_test` with larger numbers,** so the primary pool is not three independent sets.
