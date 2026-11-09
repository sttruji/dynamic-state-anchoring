# SPARSE-0: how much FFN width a latent reasoner needs per token

**Written 8 November 2026 and frozen before any run.** Its sha256 is in `PLAN.sha256` beside it, and the notebook
refuses to start unless the file matches. This is brainstorm work, not in the course report. It is evaluation only:
nothing is trained.

## Why

The MoE round (`research_notes/MoE latent reasoning/`) asked whether SIM-CoT GPT-2 could run as a mixture of experts:
fewer active parameters per token at the same accuracy. The review found that a trained MoE pilot would mostly measure
total capacity, and proposed this cheap test first. On the existing checkpoint, with nothing trained, it asks three
questions:

1. **Headroom:** how much accuracy survives when each FFN keeps only 50% or 25% of its neurons per token, chosen with the
   true activations in view?
2. **Routing value:** does choosing neurons per token beat one fixed pruned set of the same size?
3. **Phase:** is cutting FFN width more harmful in the six latent thoughts or in reading the question?

Nobody has published the third measurement for a continuous-thought model.

## Model and data

- **Model:** SIM-CoT GPT-2 (`internlm/SIM_COT-GPT2-CODI` at `8685b42`, as `scripts/latent_gate/fetch_simcot.py` pins
  it), fp32, dropout off, greedy, 6 latent steps, run exactly as `dsa/bench/models/codi.py` runs it.
- **Evaluation:** the 500 `gsm8k_valid` questions (the locked bench file). Step counts come from the rationale's `<<…>>`
  steps.
- **Calibration:** 2,048 records of `gsm_train.json` (GSM8K-Aug train), the first in `random.Random(11).sample` order
  with at most 256 question tokens and a non-empty answer. No validation or test question is used.

## The mask

A hook on every block's MLP computes the post-GELU activations a = GELU(c_fc(x)), keeps a set of neurons, and replaces
every dropped neuron i by a representative value μᵢ (G-MoEfication's fix for GELU). The output is c_proj(a ⊙ m + μ ⊙
(1 − m)).

- **Phases, set per model call:**
  - **Q:** question tokens;
  - **T:** `<bot>` and the 6 latent positions;
  - **A:** `<eot>` and the generated answer tokens.
- **Oracle selection:** per token and per layer, the k neurons with the largest |aᵢ|·‖w_out,i‖, scored in fp32, ties to
  the lower index. It sees the true activations, so a learned router can only approximate it. It is the natural ceiling
  for a router, though not a proven bound.
- **Static selection:** per layer, a fixed set of the k neurons with the highest first-order Taylor importance, the same
  at every position. This is training-free pruning.
- **k:** 1,536 neurons at 50% and 768 at 25%, out of 3,072.

## Calibration

The dense model runs in latent mode on the calibration records with the answer "The answer is: …" teacher-forced, as
`scripts/inner_loop/train.py` builds it, in batches of 64.

- **Importance:** for each example, |Σ over its positions of aᵢ·∂L/∂aᵢ|, summed over examples (MoEBERT's form). L is
  the example's mean answer cross-entropy.
- **μ, per layer and phase:**
  - for the oracle at each fraction, the mean of aᵢ over the positions of that phase where the oracle drops i;
  - for the static masks, the mean of aᵢ over all positions of that phase, since a statically dropped neuron is dropped
    everywhere;
  - where the oracle never drops i in a phase, the static value.

**Three changes from the review's sketch (Section 9 of `review.md`), all made before any run.**
- **L is the answer loss alone, not answer plus step loss.** Every arm is scored on answers only, and this keeps
  SIM-CoT's step decoder out of the run.
- **μ is kept per phase.** The phase is known at every call and costs one constant vector per layer and phase.
  Otherwise the phase arms would use constants fitted mostly to question tokens, which are about three quarters
  of calibration positions.
- **The gathered-FFN test runs on a tiny random GPT-2,** in the unit tests, not on 20 real questions. The local machine
  does not run real models.

## Arms

| Arm | Sparsified positions | Selection | Kept |
|---|---|---|---|
| ID | none | the hook with nothing dropped | all |
| O50 / O25 | Q, T and A | oracle | 50% / 25% |
| S50 / S25 | Q, T and A | static | 50% / 25% |
| O25-T | T only; Q and A dense | oracle | 25% |
| O25-Q | Q only; T and A dense | oracle | 25% |

**Also logged in ID:** per layer, the Jaccard overlap of the oracle's kept sets between consecutive question tokens and
between consecutive T positions, at both fractions. Random sets give f/(2 − f).

## The rules

B is ID's score. Every comparison is paired on the same 500 questions and uses the exact two-sided sign test. There is
no seed noise, because every arm is the same deterministic model.

- **R0, validity.** ID's predictions must match the greedy predictions in
  `results/step_select/stage0/evals/seed1.jsonl` on all 500 questions, which gives B = 204. Otherwise no other arm runs
  and there is no verdict.
- **R1, STOP, no headroom:** O50 ≤ B − 20 = 184. Even oracle selection at half the FFN costs 20 or more answers.
- **R2, STOP, no routing value:** O50 − S50 < +10 **and** O25 − S25 < +10. Per-token selection beats a fixed mask by
  less than 10 at both fractions.
  - If also S50 ≥ B − 10 = 194, record that static pruning to 50% of the FFN costs at most 10 answers training-free.
    That is a pruning note, not an MoE result.
- **R3, GO:** neither stop fires. The follow-up, D1-R (Section 7 of `review.md`), runs at 25% if O25 ≥ B − 10 = 194,
  else at 50%.
- **R4, phase,** reported in every case:
  - **thinking is the sensitive phase** if O25-T ≤ O25-Q − 10 with p < 0.05;
  - **reading is the sensitive phase** if O25-T ≥ O25-Q + 10 with p < 0.05;
  - otherwise **no phase asymmetry at 25%**.
  - Cutting T touches 7 positions per question and cutting Q about 55. So "thinking is the sensitive phase" means that a
    cut to 7 positions costs at least 10 more answers than a cut to all the question tokens.

The rules are checked in the order R0, R1, R2, R3. R4 is reported whatever the verdict.

## What each result points to

These are pointers, not commitments. Any trained follow-up gets its own frozen plan first.
- **STOP (R1):** a GELU GPT-2 latent reasoner does not tolerate per-token FFN sparsity at 50%, even with oracle
  selection. The MoE line closes.
- **STOP (R2):** per-token selection adds little over a fixed pruned set, so a router has nothing to offer here. The MoE
  line closes.
- **GO:** write D1-R's plan at the fraction R3 picks: a two-seed fine-tune with a learned router, a hash-router control
  and a width-pruned control, about 35 G4-minutes.
- **Thinking is the sensitive phase:** a phase-aware MoE (dense thoughts, sparse reading) is worth a trained arm, and
  only then (Design 2 of the review).

## What it cannot show

- That a learned router reaches the oracle, or that a fine-tuned MoE matches the dense model.
- Any wall-clock or memory saving: the masked simulation computes every neuron. The oracle also needs every activation
  to choose, so it is a ceiling, not a method.
- Anything about distilling latent reasoning into an MoE, since the weights are the latent model's own.
- Anything beyond GPT-2 with SIM-CoT.
- **Mostly short problems:** 175 of the 204 correct answers come from problems with 3 steps or fewer, so "no loss"
  mostly means short problems were preserved.

## Reported, never gating

- Each arm's score, its paired difference from ID, and its scores on problems with at most 3 steps and with 4 or more.
- How many predictions each arm shares with ID.
- **Active non-embedding parameters at a sparsified position:** 85.06M dense, 56.73M at 50% and 42.56M at 25%. The
  dropped neurons' constant folds into c_proj's bias.
- Non-embedding forward FLOPs per question, as a share of dense, at ID's positions.
- The overlap table.

## Code and compute

- `scripts/ffn_sparsity/sparsity.py` holds the hook, the selections and the parameter counter.
- `scripts/ffn_sparsity/run.py` runs the calibration, then the arms in the order of the table, resumable by arm. ID's
  identity check comes before any other arm.
- `scripts/ffn_sparsity/report.py` applies the rules.
- `tests/test_ffn_sparsity.py` checks the following:
  - the hook with nothing dropped equals GPT-2;
  - the masked FFN equals a gathered sparse FFN to 1e-5;
  - ties go to the lower index;
  - arms touch only their phases;
  - the parameter counts;
  - the run, the identity guard, resuming, and every verdict.
- `notebooks/DSA_FFN_Sparsity_Colab.ipynb` runs them in one session, plus at most one resume.

The compute is about 15 G4-minutes, with nothing trained.
