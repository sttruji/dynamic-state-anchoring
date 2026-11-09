# D1-R: results

SIM-CoT GPT-2 converted into a mixture of experts at half the FFN, then fine-tuned with the dense control's recipe. The
plan is frozen in `PLAN.md` and pinned by `PLAN.sha256`; the code is in `scripts/ffn_sparsity/`. This is brainstorm
work for the MoE line, kept out of the course report.

## The run (commit `1e48763`): NEGATIVE, by the early kill

**The checks all passed:**
- **R0a:** the dense evaluation matched step selection's greedy answers on 500 of 500.
- **R0b:** untrained Prune-50 matched SPARSE-0's S50 on 100 of 100.
- **R0c:** 51 dense training updates matched Stage 1b's `ft6_step` losses exactly, a gap of 0.0 at every logged update.
  So the copied recipe is Stage 1b's.

**The run:**
- **Session:** one Colab session.
- **Conversion:** 23 seconds.
- **Fine-tune:** MoE-50 at seed 7 took 218 seconds.

**MoE-50 at seed 7 scored 179 of 500.** That is at or below the early-kill line of 192, so the run stopped there.
Prune-50, Hash-50 and seed 8 were not run, and the verdict is NEGATIVE.

| Against MoE-50 at seed 7 (179) | Right | Difference | p |
|---|---|---|---|
| `ft6_step` at seed 7, the dense control | 209 | −30 (+14/−44) | 0.0001 |
| The base model | 204 | −25 (+15/−40) | 0.001 |
| SPARSE-0's per-token oracle at 50%, untrained | 200 | −21 (+13/−34) | 0.003 |

**Files:**
- `report.txt` is the verdict, reproduced locally from `run/` line for line apart from the path.
- `explore.txt` is the output of `scripts/ffn_sparsity/explore_d1r.py`, which is exploratory and never gating.
- `run/` holds the evaluations, the training logs, the dense check and the conversion summary. The conversion itself
  (`conversion.pt`) stays on Drive at `d1r/run/`.

## What it shows (exploratory)

- **The fine-tune recovers only part of the conversion's loss.**
  - On the first 100 questions, the untrained MoE scores 26, against the base model's 34.
  - After 768 updates it reaches 179 of 500, against the base model's 204.
  - Its answer cross-entropy falls from 0.40 at the first update to a mean of 0.26 over the last 168 updates, still
    above `ft6_step`'s 0.18 over the same updates. The step loss shows the same pattern (0.52 against 0.48).
- **The loss is spread over every problem length, and the answers stay well-formed.**
  - By step count, MoE-50 against `ft6_step`: 106 against 117 at 0–2 steps, 52 against 63 at 3, and 21 against 29 at
    4 or more.
  - No answer runs to the token cap.
- **The router does its job, and does not drift.**
  - At conversion, it picks the true top 4 experts 72–94% of the time on held-out positions, by layer and phase, where
    chance is 33%.
  - In evaluation that rate is about 0.75–0.82 before the fine-tune and the same after it.
  - Untrained, it beats the hash table (26 against 13 of 100) and the fixed mask (11).
- **The likely cost is the experts' coarseness, though this was not measured.**
  - SPARSE-0's oracle could keep any 1,536 of the 3,072 neurons per token, and lost nothing.
  - The MoE keeps a fixed 768 and can only add 4 blocks of 192 on top. Even a perfect router at that granularity is a
    coarser choice.
  - An expert-level oracle arm, needing no training, would separate this from the router's misses.

## How far this goes

- **One seed, by design.** The early kill stops at seed 7. The gap of 30 to the dense control at the same seed is far
  beyond the seed noise of about 5–9, so a second seed would not change the call.
- **A short recovery.** 768 updates, about 5.6M tokens, is about 1/170 of what D2DMoE spent sparsifying GPT-2 before
  conversion. A longer fine-tune, finer experts, or sparsifying first might do better, but the plan fixed the recipe
  to the dense control's.
- **Nothing about other settings:** it says nothing about 25%, other routers or larger models.

## What closes

Under the plan, the MoE line closes.
- **Not reached:** the routing-against-capacity comparison, since neither Prune-50 nor Hash-50 was fine-tuned.
- **Still standing:** SPARSE-0's finding. A per-token choice of half the FFN loses nothing measurable, so the room exists
  in principle, but this conversion at this budget does not reach it.
