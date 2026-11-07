# H, answer-conditioned plan search: Stage 0, the specificity test

**Written 7 November 2026 and frozen before any run.** Its sha256 is in `PLAN.sha256` beside it, and the notebook
refuses to start unless the file matches. The idea is entry H in `docs/brainstorm/ACCURACY_IDEAS.md`, from the sleep and
memory council (`research_notes/Sleep memory and latent reasoning/council_decision_final.md`). This is brainstorm work,
recorded here and in the entry, not in the course report.

## Question

On training questions SIM-CoT GPT-2 gets wrong, can a small change to the first latent z₁ make the model's own later
latent steps reach the **gold** answer? And does that work clearly more often for gold than for a wrong answer the model
finds just as likely?

- **If gold and wrong succeed about equally,** the search is writing answers into the latents, not finding plans, and
  distilling the plans (Stage 1) would teach nothing.
- **If gold succeeds far more often,** plans that reach the right answer through the model's own dynamics exist near
  greedy, and Stage 1 can try to distill them.

Stage 0 is evaluation only: nothing is trained, and nothing sampled is trained on.

## Data

- **Model:** SIM-CoT GPT-2 (`internlm/SIM_COT-GPT2-CODI`, pinned as in `scripts/latent_gate/fetch_simcot.py`).
- **Pool:** `gsm_train.json` (GSM8K-Aug train) in the order `random.Random(11).sample` gives.
- **Skipped,** and not counted: questions over 256 tokens, and golds that are not non-negative integers.
- **No validation or test question is touched.**

## Phase 1: greedy pass, at most 8 G4-minutes

- **Greedy:** each pool question runs greedy, with the six latent steps and dropout off, until 1,000 questions are greedy
  wrong or the cap is reached.
- **Basis:** every processed question's greedy z₁ is kept for the PCA.
- **Radius:** each greedy-wrong question also gets 8 MC-dropout samples of z₁, from the question pass and CODI's
  projection with dropout at p = 0.2. Torch is reseeded per question with `11 * 100003 + pool index`.
  - Its radius r is the largest distance from greedy z₁ to a dropout sample.
- **Identity:** on the first 20 processed questions, greedy computed through `latent_step` must give the same answer
  text as GPT-2's own forward. Otherwise nothing runs.

## Phase 2: search, at most 12 G4-minutes

- **Basis:** fixed after Phase 1 and before any search. The top 64 principal directions U of the greedy z₁ over every
  processed question, mean-centered, with standard deviations σ.
- **Targets for each greedy-wrong question:**
  - **gold**;
  - **wrong:** among the integers in [max(0, gold − 50), gold + 50], excluding gold and greedy's own prediction, the
    one whose base log-probability is closest to gold's. The base log-probability is log p("The answer is: X" + end of
    text | the question and greedy's latent steps). Ties go to the smaller |X − gold|, then the smaller X.
- **Search:**
  - z₁ = greedy z₁ + U(σ ⊙ c), with c starting at 0;
  - the objective is to maximize the log-probability of the target's answer tokens after the model's **own** latent
    steps 1–6 from that z₁, with dropout off;
  - 20 Adam steps on c at learning rate 0.1;
  - after each step, if ‖U(σ ⊙ c)‖ > r, c is rescaled onto the radius-r ball, so the plan stays within the question's
    own dropout spread;
  - gold and wrong are searched side by side, from the same start, in the same way.
- **Batching:** 16 questions are searched together. The batch is left-padded as in `scripts/inner_loop/train.py`, and
  every row's search is independent of the others.
  - The base log-probabilities and the accepting rollout run one question at a time, unpadded.
  - At the first step, before any move, the padded batch must reproduce the unpadded base log-probabilities to within
    0.01, or the run stops.
- **Acceptance:** after the 20 steps, a deterministic rollout from the final z₁ (latent steps with dropout off, then
  greedy answer decoding) must give **exactly** the target.

## Gate

- **KILL** if accepted gold plans number less than 1.5 × accepted wrong plans, **or** fewer than 150 gold plans are
  accepted.
- **GO** otherwise. Stage 1 then runs: distilling the accepted plans, at 2 seeds, against ft6 and a gold-duplicate
  control, under the shared rule in `ACCURACY_IDEAS.md`. Stage 1 is built only after a GO.
- If a cap stops a phase early, the gate is applied to the questions searched, with the same absolute floor of 150.

## Reported, never gating

- Acceptance rates for gold and for wrong.
- The questions where both are accepted, and where only one is.
- How often the plan ends on the radius boundary.
- The base log-probability gap between the two targets, and their final log-probabilities.
- Distances as a fraction of r.
- Greedy accuracy on the processed pool.

## Code

- `scripts/plan_search/stage0.py` runs both phases and can resume Phase 2.
- `scripts/plan_search/report.py` applies the gate.
- `notebooks/DSA_Plan_Search_Stage0_Colab.ipynb` runs them.

The compute is at most 20 G4-minutes.
