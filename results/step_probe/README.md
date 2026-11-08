# Where long problems break: results

A probe diagnosis of SIM-CoT GPT-2. Linear probes read intermediate values and the answer from the model's hidden
states, to tell whether wrong questions fail at readout or in the computation, and where wrong chains break. The plan
is frozen in `PLAN.md` and pinned by `PLAN.sha256`. The code is in `scripts/step_probe/`. This is brainstorm work, kept
out of the course report.

## The run (commit `b01230d`): INCONCLUSIVE

- **Session and identity:** one Colab session. Greedy matched step selection's Stage 0 answers on 500 of 500 validation
  questions.
- **Training extraction:** 9,000 questions, 1,500 for each step count from 1 to 6, in 343 seconds, under the cap.
- **Probes:** fitted in 16 seconds.
- **Files:** `report.txt` is the verdict, reproduced locally to the byte from `run/`. `explore.txt` is the output of
  `scripts/step_probe/explore.py`, which is exploratory and never gating. The hidden states (about 0.5 GB) stay on
  Drive at `step-probe/run/`.

**The verdict is INCONCLUSIVE on both splits.** The answer probe reads gold on only 8.7% of greedy-right training
questions and 5.5% of right validation questions, against the bar of 50%. No layer or position does better: the best
is layer 12 at the last latent step. So, under the plan, the diagnosis cannot say whether wrong questions fail at
readout or in the computation.

| | Train | Validation |
|---|---|---|
| Sensitivity s (gold read on right questions) | 0.087 | 0.055 |
| Wrong questions in scope | 2,650 | 241 |
| Final latents read as gold (g) / as the model's own answer (w) | 0.023 / 0.026 | 0.037 / 0.021 |
| Break profile label | EARLY | EARLY |

**The EARLY label cannot be taken at face value.** The probes read steps 3 and later on at most about 30% of right
questions, so late breaks are nearly invisible for both groups. The label only says that wrong chains fail the first two
readable steps more often.

## What the probes can read (exploratory)

- **The first two steps are linearly readable, and later ones are not.** Each step is read best at layer 3 of the
  position that takes in its latent.
  - On right training questions, step 1 of a 2–4-step problem reads exactly 73–80% of the time, and step 2 of a
    3–4-step problem 56–63%.
  - Step 3 falls to about 30%, and the final answer to under 9% anywhere.
  - SIM-CoT's own step decoder showed the same fall in step selection's Stage 0: exact steps went from 199 at slot 0 to
    21 at slot 2.
- **Final answers are hard to read even when they are step 1.** On 1-step problems, step 1 is the answer, and it reads
  exactly only 20% of the time, against 73% for step 1 of 2-step problems. The answer probe gets the units digit right
  53% of the time but the whole number 9%. So the answer seems to be held differently from intermediate values, or
  assembled only while the answer tokens are decoded.
- **Wrong chains hold the gold first steps less often.**
  - On 2–4-step training problems, step 1 reads as gold on 35–58% of wrong questions, against 73–80% of right ones.
  - On 3–4-step problems, step 2 reads as gold on 22–27% of wrong questions, against 56–63%.
  - Validation shows the same pattern: step 1 at 22–51% against 65–76%, and step 2 at 11–18% against 45–61%.
- **The gap holds within each value size,** for example 0.56 against 0.83 for first steps between 10 and 99. It is not
  a different step done first: a misread first step equals another gold step's value 11% of the time on wrong questions
  and 13% on right ones.
- **Reading step 1 predicts the answer.** On 4-step training problems, greedy is right 74% of the time when step 1
  reads as gold, against 49% when it does not.

**How far this goes.** These reads fit errors that begin at the first steps, which agrees with the earlier decoder-based
finding that 188 of 296 validation errors were plan errors. There are two caveats.
- The probes were trained on right questions only. On wrong questions, which are harder, they may transfer worse, which
  would also lower readability.
- None of this was a preregistered rule, and the plan's verdict stays INCONCLUSIVE.
