# Step selection, Run 1: a test-set check of the steps 1–3 vote and a hidden-state probe

**Written 6 November 2026 after Stage 0 and E0, and frozen before any run.** Its sha256 is in `PLAN.sha256` beside it. The
notebook refuses to start unless the file matches.

**Where this comes from.** Stage 0 stopped under its preregistered rule (`../README.md`). E0 found one positive: a
training-free vote over greedy plus the step 1–3 reruns, +7/−1 at the question level on validation (p = 0.07). A
STOP/PROCEED review agreed on this plan, and it is the last test of step-level selection in this project. No RL runs, and
every route not named here is closed (section 5).

## Run

- **Data:** SIM-CoT GPT-2 (`internlm/SIM_COT-GPT2-CODI`, pinned as in `scripts/latent_gate/fetch_simcot.py`) on all
  1,319 `gsm8k_test` questions, with MC dropout at p = 0.2.
- **Seed:** torch is reseeded per question with `3 * 100003 + question_index`, so a resume does not change any draw.
- **Per question:**
  - greedy, the six latent steps with dropout off;
  - 7 dropout reruns of step s from greedy's state, for each s in 1, 2 and 3, each finished with dropout off to an
    answer;
  - no decoder reads and no chains.
- **Saved per candidate** (candidate 0 being greedy's own step): the prediction and whether it is correct, and in fp16
  the last-layer hidden state at the step position (after `ln_f`) and the projected next latent.
- **Code:** `scripts/step_select/run1.py`, then `scripts/step_select/run1_report.py` for every number below.
- **Cap:** 20 G4-minutes of evaluation, in one session that may be resumed at most once with the same settings. A run that
  stops at the cap before 1,319 questions cannot declare CONFIRM or GO. It may still declare KILL on the completed
  questions if they hold at least 300 sets.
- **Test-set use:** after Run 1, `gsm8k_test` is spent for step-level selection. No later step-selection result may be
  tuned or evaluated on it.

## Preflight, on the first 50 questions

1. **Identity:** greedy must match the text of `results/latent_gate/heldout_scale/simcot-gpt2/gsm8k_test.jsonl` (plain,
   6 steps) on 50 of 50 questions, or nothing runs.
2. **Projected time:** the 50 questions' elapsed time × 1319/50 must be 15 minutes or less.
   - If it is more, the full run uses 3 reruns per step and **B does not run**.
   - Stage 0b, a training-split probe scored on validation, then becomes Run 2, judged by section 4's criteria.
   - Stage 0b is not built yet.

**Full-run identity:** if fewer than 1,319 of 1,319 greedy texts match the reference, the run cannot declare CONFIRM or GO,
but may declare KILL.

## A. The steps 1–3 vote (confirmation)

- **The vote:** V1 as in `scripts/step_select/e0.py`.
  - Greedy weighs 1.01, and **the first 3 reruns** at each of steps 1, 2 and 3 weigh 1 each.
  - Right answers count as one answer, wrong ones are grouped by prediction, and the plurality wins.
- **CONFIRM:** a net of at least +8 out of 1,319 against greedy (questions the vote gets right minus questions it newly
  gets wrong), **and** a two-sided question-level sign test p < 0.05.
- **KILL** otherwise.
- No further compute either way.

## B. A hidden-state probe, cross-fitted on the test questions

- **Sets:**
  - Each (question, step) for steps 1–3 is one set of 8 candidates: candidate 0 plus 7 reruns.
  - A set is *mixed* if its candidates include both right and wrong answers.
  - **Fewer than 300 mixed sets is a KILL.**
- **Features:** the concatenated [hidden state; latent] (1,536 values), standardized, then reduced to a 64-dimensional
  PCA.
- **Folds:**
  - outer fold = `int(sha256("run1-outer:" + id), 16) % 5`;
  - inner fold = `int(sha256("run1-inner:" + id), 16) % 4`;
  - both are fixed before any label is read.
- **Model:**
  - **The scorer:** a conditional logit. The score is w·x, the softmax runs over each set's candidates, and the loss is
    −log of the probability mass on the set's right candidates. It is fitted on mixed sets.
  - **Fitted only on each outer fold's training questions:** standardization and PCA (on all candidates of those
    questions), the L2 weight (from {0.01, 0.1, 1, 10}, chosen by mean inner-fold within-set AUC), and the candidate-0
    margin.
- **Out-of-fold metrics:** every question's candidates are scored by the model of the fold that held it out.
  - **Within-set AUC:** over the mixed sets, the chance that a right candidate outscores a wrong one in the same set, ties
    counting half.
  - **The one-deviation selector, at K = 4** (candidate 0 and the first 3 reruns at steps 1–3):
    - Take the candidate (s, k ≥ 1) with the largest score margin over its own set's candidate 0.
    - If that margin exceeds the candidate-0 margin m, use that branch's answer; otherwise keep greedy.
    - m comes from {0, 0.25, 0.5, 1}, chosen by the inner folds' net gain.
  - **Net against greedy,** with a two-sided question-level sign test.
- **Outcomes:**

| Outcome | Condition | Consequence |
|---|---|---|
| **GO** | within-set AUC ≥ 0.65 **and** net ≥ +27 (2% of 1,319) **and** p < 0.05 | Run 2, Stage 0b, runs |
| **REPORT** | within-set AUC ≥ 0.65, but a gain condition fails | write up the AUC and the gain; no more compute |
| **KILL** | within-set AUC < 0.65, or fewer than 300 mixed sets | every selector route ends |

- **Reported, never gating:**
  - the within-set AUC of a candidate-0 detector (1 for candidate 0, 0 otherwise; 0.526 on validation);
  - the AUC with the probe trained on 40% and 80% of each outer fold's training questions, chosen by question hash.
- **No reopening:** after a KILL, a near-miss AUC or a learning curve still rising does not reopen the line.

## 4. Run 2: Stage 0b (only on GO, or in B's place)

- **Data:** N_train = the smaller of 4,000 and however many GSM8K-Aug training questions fit in 18 G4-minutes.
- **Stops without a result,** counted as KILL, if N_train < 2,000 or fewer than 400 mixed sets are projected. The
  training-set greedy accuracy and set rate are recorded.
- **Probe:** as in B, with every fit on the training split.
- **Evaluation:** scored once on the 500 validation questions, with seed 4 and K = 4. The cap is 30 G4-minutes.
- **CONFIRM:** within-set AUC ≥ 0.65, a net of at least +10/500 and question-level p < 0.05. That would be written up as a
  method result. Anything else is REPORT if the AUC is at least 0.65, and KILL otherwise.

## 5. Closed for good, whatever happens

- decoder-feature selectors, meaning the fixed rule, learned rankers and keep-gates;
- Stage 1 as designed, and RL in this project;
- every other sampler setting;
- votes over dropout chains;
- further variants of the steps 1–3 vote;
- any rerun of A or B with new K, steps, seeds, inputs or thresholds;
- step-selection use of `gsm8k_test` after Run 1;
- after a KILL in B or Run 2, hidden-state single-select and hidden-state keep-gates.

## 6. Write-up

The write-up goes in a section of `paper/main.tex`, "Test-time selection over latent steps", and in `../README.md`. It
covers:
- the Stage 0 table;
- the diagnosis that the room is stable but unreadable: 24 questions rescued in both seeds against 3.4 by chance, gold
  oracle step AUCs of 0.50–0.54, and learned decoder rankers scoring below candidate 0;
- the curve from AUC to gain;
- E0, at its question-level p;
- Run 1, and Run 2 if it runs;
- a disclosure of how the test set was used, with test greedy (519/1,319, 39.3%) beside validation greedy (204/500,
  40.8%);
- a ledger of every variant examined, each with its question-level test.
