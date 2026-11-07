# Step-level selection: results

Step-level selection chooses, at every latent step of SIM-CoT GPT-2, among the model's own step (candidate 0) and
MC-dropout reruns of it. The plan and decision rule are in `docs/brainstorm/STEP_SELECTION.md`, and the code is in
`scripts/step_select/`. `stage0/report.txt` is the output of `scripts/step_select/report.py` on both seeds, with Stage 0
of the inner loop as the identity reference.

## Stage 0 (evaluation only, commit `cc9c43a`): STOP

The run used the 500 validation questions, MC dropout at p = 0.2, 4 candidates per step and 18 chains, with sampling seeds
1 and 2.

| | Seed 1 | Seed 2 | Pooled |
|---|---|---|---|
| Identity: greedy reproduces the checkpoint | 500/500 PASS | 500/500 PASS | |
| Greedy | 204 | 204 | |
| Greedy or any of 18 one-step branches right | 238 (+34) | 234 (+30) | +6.4%, under the 8% GO bar |
| Greedy or any of 18 dropout chains right | 262 (+58) | 259 (+55) | +11.3% |
| One dropout chain, expected correct | 189.2 | 189.8 | below greedy |
| Fixed decoder rule right in mixed sets, against uniform chance | 90 vs 88.0 | 99 vs 88.0 | 189 vs 176.0, p = 0.071 |
| Untrained selector | 203 | 201 | −4 vs greedy (+9/−13, p = 0.52) |
| Majority of 4 chains / of 18 chains | 191 / 199 | 193 / 196 | both below greedy |

- **The verdict is STOP.** The per-step branches cover under 8% more than greedy, the decoder rule shows no signal at
  p < 0.05, and the untrained selector does no better than greedy.
- **The decoder cannot tell candidates apart within a question.**
  - The AUCs of valid arithmetic, grounded operands and the rule's rank are about 0.50, and decoder confidence is
    0.51–0.54.
  - The rule's small edge over uniform chance comes from keeping candidate 0. Always keeping candidate 0 would be right
    in 187 of the 348 mixed sets, against the rule's 189.
  - Where the rule moved off candidate 0 (66 sets), its pick was right 34 times and candidate 0 32.
  - The earlier gap, decode-clean greedy chains right 63% of the time against 36%, separates easy questions from hard
    ones, not right chains from wrong ones.
- **Branching breaks as much as it rescues.** At steps 1–3, one rerun of a step rescues about 17–20 questions and breaks
  about as many.
- **Dropout chains have more room but are individually weaker than greedy,** so voting over them lands below greedy.

## Run 1 (test set, commit `060e834`): A CONFIRMS, B KILLS, and step-level selection ends

**The plan.** It is frozen in `run1/PLAN.md`, pinned by `run1/PLAN.sha256`, and every criterion was fixed before the run.
It is evaluation only and the last test of step-level selection.
- **Data:** the 1,319 GSM8K test questions, with seed 3 and dropout at p = 0.2.
- **Candidates:** greedy plus 7 dropout reruns at each of steps 1–3, each rerun finished greedily.
- **Saved:** every candidate's hidden state and latent.

**The run.**
- **Preflight:** greedy matched the held-out study's test answers on 50 of 50 questions. The run projected to 3.4 minutes,
  so it kept 7 reruns.
- **Full run:** 1,319 of 1,319 questions, matching the held-out reference on all 1,319. One Colab session and 195
  seconds of evaluation.
- **Files:** `run1/report.txt` is the verdict, reproduced locally to the byte from `run1/records.jsonl` and the features.
  The features (about 97 MB) stay out of git, on Drive at `step-select/run1/run/features/`.

| Test | Result | Criterion | Verdict |
|---|---|---|---|
| A. Vote over greedy and the first 3 reruns at steps 1–3 | 528 vs greedy 519: **net +9 (+12/−3), p = 0.035** | net ≥ +8 and p < 0.05 | **CONFIRM** |
| B. Hidden-state probe, cross-fitted over 5 question folds | within-set AUC **0.495** over 400 mixed sets; one-deviation selector −1 (+14/−15) | AUC ≥ 0.65 | **KILL** |

**A.** The training-free vote from E0 replicates on questions it had never seen: +9 of 1,319, about +0.7 points, at about
9× greedy's compute. It is a real but small effect, consistent with the validation result (+7/−1). Under the plan, nothing
further runs on it.

**B.** Hidden states carry no within-set signal either:
- the probe's AUC is 0.495, against 0.499 for a candidate-0 detector;
- the learning curve is flat, at 0.499 with 40% of the training questions and 0.498 with 80%;
- the inner-fold AUCs run from 0.486 to 0.515.

Together with Stage 0's decoder features (AUC about 0.50), neither a readable step nor the hidden state tells which
dropout rerun will end right. Run 2 does not run.

**What closes, per PLAN.md section 5:**
- every selector route: decoder features, hidden-state single-select and keep-gates;
- RL in this project;
- other sampler settings;
- votes over dropout chains;
- further variants of the vote.

The test set is now spent for step-level selection.
