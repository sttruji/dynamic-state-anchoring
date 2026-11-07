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

## Run 1 (planned, frozen before running): the steps 1–3 vote and a hidden-state probe, on the test set

The plan is in `run1/PLAN.md`, pinned by `run1/PLAN.sha256`. It is evaluation only and is the last test of step-level
selection. It runs greedy plus 7 dropout reruns at each of steps 1–3 on the 1,319 GSM8K test questions, with a 20 G4-minute
cap. The code is `scripts/step_select/run1.py` and `run1_report.py`, and the notebook is
`notebooks/DSA_Step_Select_Run1_Colab.ipynb`, opted in with `RUN_TEST`.

| Test | Pass | Otherwise |
|---|---|---|
| A. Vote over greedy and the first 3 reruns at steps 1–3 | CONFIRM: net ≥ +8 and question-level p < 0.05 | KILL |
| B. A hidden-state probe, cross-fitted over 5 question folds | GO: within-set AUC ≥ 0.65, net ≥ +27 and p < 0.05 | REPORT if only the AUC passes; KILL if AUC < 0.65 |
