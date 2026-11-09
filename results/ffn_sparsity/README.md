# SPARSE-0: results

How much FFN width per token SIM-CoT GPT-2 needs, in which phase, and whether per-token selection beats a fixed pruned
set. It is evaluation only: nothing was trained. The plan is frozen in `PLAN.md` and pinned by `PLAN.sha256`; the code is
in `scripts/ffn_sparsity/`. This is brainstorm work for the MoE line (`research_notes/MoE latent reasoning/`), kept out
of the course report.

## The run (commit `e577041`): GO at 50%

- **Session and identity:** one Colab session. ID matched step selection's Stage 0 greedy answers on 500 of 500, so
  B = 204.
- **Calibration:** 2,048 training records in 12 seconds. The mean answer cross-entropy was 0.146.
- **Time:** each arm took 34 seconds, except the static arms. S50 took 299 seconds and S25 566, because their answers
  ran to the 256-token cap.
- **Files:**
  - `report.txt` is the verdict, reproduced locally from `run/` line for line apart from the path.
  - `explore.txt` is the output of `scripts/ffn_sparsity/explore.py`, which is exploratory and never gating.
  - `run/` holds the per-arm answers, the status and the overlap table. The calibration file (importance and
    stand-in values, `calibration.pt`) is ignored by git as a `.pt` file and stays on Drive at `ffn-sparsity/run/`.

| Arm | Right | vs ID | p | Active non-emb params per sparsified token | FLOPs per question |
|---|---|---|---|---|---|
| ID | 204 | | | 85.06M (100%) | 100% |
| O50, oracle 50% everywhere | 200 | −4 (+10/−14) | 0.54 | 56.73M (67%) | 67% |
| O25, oracle 25% everywhere | 183 | −21 (+14/−35) | 0.004 | 42.56M (50%) | 50% |
| S50, fixed 50% | 60 | −144 | <0.001 | 56.73M (67%) | 67% |
| S25, fixed 25% | 0 | −204 | <0.001 | 42.56M (50%) | 50% |
| O25-T, oracle 25% on thinking only | 203 | −1 (+8/−9) | 1.00 | 42.56M (50%) | 95% |
| O25-Q, oracle 25% on the question only | 192 | −12 (+13/−25) | 0.07 | 42.56M (50%) | 60% |

**The rules:**
- **R1, no headroom, does not fire.** O50 = 200, above the bar of 184.
- **R2, no routing value, does not fire.** O50 − S50 = +140 and O25 − S25 = +183.
- **R3: GO, at 50%,** because O25 = 183 is under the bar of 194 for running at 25%.
- **R4: no phase asymmetry at 25%.** O25-T − O25-Q = +11 (+25/−14), but p = 0.108 misses the bar of 0.05.

## What it shows

- **Per-token selection of half the FFN costs nothing measurable.** The oracle at 50% keeps 200 of 204 answers. At
  25%, it loses 21.
- **A fixed half of the FFN breaks the model.**
  - S50's answers degenerate into repeated `:` characters: 245 of 500 run to the token cap, and all 60 right answers
    come from the rest. S25 does this on all 500.
  - This is not a stand-in-value bug. Every stand-in value is small (all under 3.3 in absolute value), and the Taylor
    importance is spread thin: the top half of each layer's neurons holds only 60–69% of it, and the top quarter 35–47%.
  - So, training-free, which neurons a token needs changes from token to token.
  - The +140 gap therefore measures how badly a fixed mask fails without fine-tuning. It is not an estimate of what a
    trained router adds over a fine-tuned pruned model. That comparison is D1-R's.
- **The thoughts tolerate 25%, and reading the question is the more fragile phase, though not significantly so.**
  - O25-T keeps 203 of 204. O25-Q loses 12 (p = 0.07).
  - On problems with 4 or more steps, where ID gets 29 right, O25-Q is down 8 (+1/−9, p = 0.021). That is
    exploratory, not a preregistered rule.
  - The thinking positions are only 10% of positions, so cutting them to 25% saves 5% of FLOPs. Any real saving has to
    come from reading.
- **Neighbouring thoughts share more neurons than neighbouring question tokens.** The Jaccard overlap of the oracle's
  kept sets between consecutive positions:
  - at 25%, 0.34–0.45 between thoughts and 0.24–0.36 between question tokens, against 0.14 for random sets;
  - at 50%, 0.50–0.63 against 0.40–0.54, with random at 0.33.

## How far this goes

- **The oracle is a ceiling, not a method.** It sees every true activation, and so computes the full FFN in order to
  choose. A learned router can only approximate it, and the run shows no wall-clock or memory saving.
- **The best case at 50% is modest.** Active non-embedding parameters fall to 67% of dense, or about 77% counting the
  embeddings.
- **Mostly short problems:** 175 of the 204 right answers come from problems with 3 steps or fewer.
- **One model:** GPT-2 with SIM-CoT only, training-free.

## Next, under the plan

GO means D1-R at 50% gets its own frozen plan. Section 7 of the review gives its arms and rules:
- **Arms:** MoE-50 (768 shared neurons plus 4 of 12 routed experts of 192, a 2-layer MLP router), Prune-50 and
  Hash-MoE-50, each fine-tuned at seeds 7 and 8.
- **Budget:** about 35 G4-minutes.
- **Hash-MoE-50** is there to tell routing from capacity. The reviewer's expected outcome is "MoE beats pruning, through
  capacity".

D2's trained phase arm does not run, because R4 did not find the thoughts to be the sensitive phase.

**Outcome:** D1-R ran (commit `1e48763`) and was NEGATIVE by its early kill. MoE-50 at seed 7 scored 179, against
`ft6_step`'s 209. The MoE line is closed. See `d1r/README.md`.
