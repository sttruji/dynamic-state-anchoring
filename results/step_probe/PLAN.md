# Where long problems break: a probe diagnosis of SIM-CoT GPT-2

**Written 8 November 2026 and frozen before any run.** Its sha256 is in `PLAN.sha256` beside it, and the notebook
refuses to start unless the file matches. This is brainstorm work, recorded here and in
`docs/brainstorm/ACCURACY_IDEAS.md`, not in the course report. It is evaluation only: nothing is trained except linear
probes on saved hidden states.

## Why

SIM-CoT GPT-2's accuracy falls with problem length, even on training problems it saw. Greedy is right on 93% of 1-step
training problems and 30% of 6-step ones; on validation the fall is from 50% to 10%. Every method tried so far was aimed
without knowing where wrong chains go wrong. This diagnosis asks two questions:

1. **Readout or compute?** On problems the model gets wrong, do its final latent states hold the gold answer (it is
   computed but read out wrong) or the model's own wrong answer (the computation itself went wrong)?
2. **Where does the chain break?** For wrong chains, is the first intermediate result that cannot be read early or late
   in the problem?

Each answer points to a different kind of fix, listed under **What each result points to**.

## Data

- **Model:** SIM-CoT GPT-2 (`internlm/SIM_COT-GPT2-CODI`, pinned as in `scripts/latent_gate/fetch_simcot.py`), greedy,
  dropout off, run exactly as `dsa/bench/models/codi.py` runs it.
- **Validation:** the 500 `gsm8k_valid` questions (the locked bench file). Their step counts come from the rationale's
  `<<…>>` steps.
- **Training pool:** `gsm_train.json` (GSM8K-Aug train) in the order `random.Random(11).sample` gives.
  - Kept: questions with 1–6 steps, a gold answer that is a non-negative integer, and at most 256 tokens.
  - Up to 1,500 questions per step count, taken in pool order, then processed round-robin across step counts so a cap
    cuts every count evenly.
- **No test question is touched.**

## Identity

Greedy answers on all 500 validation questions must match the greedy predictions in
`results/step_select/stage0/evals/seed1.jsonl` exactly. Otherwise nothing else runs.

## Hidden states

For every question, the hidden states of layers 3, 6, 9 and 12 (12 is after the final layer norm) at 8 positions:
- **position 0:** `<bot>`, the last position of the question pass;
- **positions 1–6:** latent steps 1–6;
- **position 7:** `<eot>`, the first answer-readout position.

SIM-CoT trains the latent made from position p to decode as step p + 1, so step j's own position is j − 1.

## Targets

- **Step values** c₁…c₆: the number after the last `=` in each gold step.
- **The gold answer.**
- A target is valid only if it is an integer in [0, 9999].

## Probes

- **Form:** one linear probe per target, layer and position, from the standardized hidden state to 4 digits (units,
  tens, hundreds, thousands), each a 10-way softmax. The probe's reading is the 4-digit number it predicts.
- **Training:** on greedy-right training questions with a valid target, by Adam at learning rate 0.01 and weight decay
  0.001, for 300 full-batch steps. Standardization uses the training rows' means and standard deviations.
- **Cross-fitting:** 5 folds by `int(sha256(str(pool index)), 16) % 5`. Every training question is read by probes
  trained on the other folds. Validation questions are read by probes trained on all greedy-right training questions.
- **Exact:** a reading is exact when it equals the target.

## The rules

**Answer reading point.** Let (ℓ\*, p\*) be the layer and position, among layers 3, 6, 9 and 12 and positions 6–7, with
the highest exact rate for the gold answer on greedy-right training questions. Let s be that rate (its
**sensitivity**), measured separately on each split's greedy-right questions.

**Wrong questions in scope.** Greedy wrong, 1–6 steps, a valid gold answer, and a prediction that is an integer in
[0, 9999] and differs from gold. Then:
- g is the fraction whose reading at (ℓ\*, p\*) equals **gold**;
- w is the fraction whose reading equals the model's own **prediction**;
- g₀ and w₀ are the same at (ℓ\*, position 0), before any latent step, as the control for answers readable from the
  question alone.

**Verdict, per split; the headline is validation.**
- **INCONCLUSIVE** if s < 0.5: the probes are too weak to say anything.
- **READOUT-LIMITED** if g/s ≥ 0.4 and g ≥ 2·g₀: the final latents often hold gold while the model says something else.
- **COMPUTE-LIMITED** if w/s ≥ 0.4 and g/s < 0.2: the final latents hold the model's own wrong answer.
- **MIXED** otherwise.

**Break profile, per split.**
- **Reading points:** each step j is read at its own reading point. This is the layer and position with the highest exact
  rate for c_j on greedy-right training questions with at least j steps.
- **Questions:** those with 3–6 steps whose step values are all valid.
- **The break j\*:** the first step whose reading is not exact, or none. Its relative position is
  r = (j\* − 1)/(n − 1).
- **Excess breaks:** for right and wrong questions separately, take the fraction with an early break (r ≤ 0.5) and with a
  late break (r > 0.5). The excess is wrong minus right, floored at 0: E_early and E_late.
- **The label,** checked in this order:
  - **NONE** if both are 0;
  - **EARLY** if E_early ≥ 2·E_late;
  - **LATE** if E_late ≥ 2·E_early;
  - **SPREAD** otherwise.

## What each result points to

These are pointers for choosing the next idea, not commitments.
- **READOUT-LIMITED:** test a readout-side change, such as an answer head trained to read the latents. It needs no new
  reasoning.
- **COMPUTE-LIMITED and EARLY:** chains go wrong at their first steps. Look at how the question becomes the first latents.
- **COMPUTE-LIMITED and LATE:** chains start right and go wrong later. Look at how state is carried from step to step.
- **MIXED, or SPREAD:** there is no single place to aim a fix, which argues for a different direction.
- **INCONCLUSIVE:** linear probes cannot read SIM-CoT GPT-2's values, and the diagnosis says nothing either way.

## Reported, never gating

- The exact-rate matrices by target, layer and position, for right and wrong questions on each split.
- Greedy accuracy, g and w by step count.
- The share of step values that appear literally in the question.
- Exact rates by value size.

## Code and compute

- `scripts/step_probe/run.py` runs the identity check, the extraction (resumable, training capped at 12 G4-minutes)
  and the probes.
- `scripts/step_probe/report.py` applies the rules.
- `notebooks/DSA_Step_Probe_Colab.ipynb` runs them, in one session plus at most one resume.

The compute is at most 15 G4-minutes.
