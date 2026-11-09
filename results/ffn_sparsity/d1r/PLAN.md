# D1-R: SIM-CoT GPT-2 as a mixture of experts at half the FFN

**Written 8 November 2026 and frozen before any run.** Its sha256 is in `PLAN.sha256` beside it, and the notebook
refuses to start unless the file matches. This is brainstorm work, not in the course report. It extends the MoE line
after SPARSE-0's GO (`../README.md`). The design is the review's D1-R (Section 7 of
`research_notes/MoE latent reasoning/review.md`), at the 50% that SPARSE-0's rule R3 picked.

## Why

SPARSE-0 showed that, choosing per token with the true activations in view, half of each FFN keeps 200 of 204 answers,
while a fixed half collapses without fine-tuning. D1-R asks whether that survives a real router and a short
fine-tune.

1. **Equal accuracy:** after the same fine-tune as the dense control, does an MoE using half the FFN per token match the
   dense model?
2. **Is an MoE needed?** Does it beat a model pruned to the same width and fine-tuned the same way?
3. **Is it the routing?** Does a learned router beat a fixed hash on the same experts? MoEBERT found that a hash router
   does as well, so without this control any MoE gain could be total capacity alone.

## Starting point, data and recipe

- **Model:** SIM-CoT GPT-2 (`internlm/SIM_COT-GPT2-CODI` at `8685b42`, as `scripts/latent_gate/fetch_simcot.py` pins
  it). It scores 204 of 500 on validation.
- **Fine-tune:** Stage 1b's `ft6_step` recipe from `scripts/inner_loop/train.py`, copied unchanged into
  `scripts/ffn_sparsity/train_moe.py`:
  - answer cross-entropy plus SIM-CoT's step loss, with the step decoder frozen;
  - the same 98,304 GSM8K-Aug records in the same order for each seed;
  - batch 128, learning rate 4e-5 (2e-5 · √(128/32)), AdamW with weight decay 0.01, 3% warmup, then linear decay;
  - gradient clipping at 2, bf16 autocast, 768 updates, 6 latent steps.
- **Seeds:** 7 and 8.
- **Dense control:** Stage 1b's `ft6_step` at seeds 7 and 8, reused as it stands. It scored 209 and 200, and its
  per-question files are in `results/inner_loop/stage1b/evals/ft6_step/` and
  `results/inner_loop/stage1b-seed8/evals/ft6_step/`.
- **Evaluation:** the 500 `gsm8k_valid` questions, greedy, fp32, 6 steps, run as `dsa/bench/models/codi.py` runs it.

## The conversion

`scripts/ffn_sparsity/convert.py` builds it from SPARSE-0's calibration file, `ffn-sparsity/run/calibration.pt` on
Drive, with sha256 `2dd1d19ec3ddb1099177ac801c88f40364031c9cf882aebf1b195cf0d263b75f`. That file holds the per-neuron
Taylor importance, the per-phase mean activations and its 2,048 training records.

1. **FFN inputs.** The dense model runs in latent mode on those 2,048 records with the answer teacher-forced.
   - It keeps the FFN input at every T and A position and at a seeded quarter of the Q positions.
   - The phases are SPARSE-0's: Q (question tokens), T (`<bot>` and the 6 latents), A (`<eot>` and the answer).
2. **Experts, per layer.**
   - The 768 neurons with the highest importance are **shared** and always on.
   - The other 2,304 are split into **12 experts of 192** by balanced k-means. The features are each neuron's
     standardized activation over 8,192 kept positions, and there are 25 passes.
3. **Router, per layer.**
   - A 2-layer MLP (768 → 128 → 12, ReLU) reads the FFN input. It regresses each expert's contribution norm
     ‖W_out,eᵀ(a_e − μ̄_e)‖, where μ̄ is SPARSE-0's per-phase mean (D2DMoE's target, centred because dropped neurons
     stand in at their means).
   - Fit with Adam at 1e-3, 2,000 steps of 4,096 positions, on the records whose index is not divisible by 10. Its
     recall of the true top 4 is scored on the rest.
   - Router scores are computed in fp32 and ties go to the lower index.
4. **Stand-in values, per phase.** A dropped neuron i is replaced by μᵢ, as in SPARSE-0.
   - For MoE and Hash, μᵢ is the mean activation over the kept positions where the router, or the hash, leaves i's
     expert off.
   - For Prune, μᵢ is SPARSE-0's per-phase mean.
   - The constant enters through c_proj, so it tracks c_proj as c_proj trains.

## Arms

Each arm uses 1,536 of 3,072 neurons per position: 56.73M active non-embedding parameters, against 85.06M for the
dense model. MoE adds its routers, 1.20M.

| Arm | Neurons on at each position |
|---|---|
| **MoE-50** | the 768 shared, plus the 4 of 12 experts the router picks |
| **Hash-50** | the 768 shared, plus 4 experts picked by a fixed random table: of the token id at Q and A positions, of the step index at T positions (`<bot>` is step 0) |
| **Prune-50** | a fixed set: the 1,536 neurons with the highest importance. Untrained, this is SPARSE-0's S50 |

- **Fine-tuned:** every arm at seeds 7 and 8. All weights of the model and the projection train.
- **Fixed:** the router, the hash table, the expert split and the stand-in values.
- **Not fine-tuned:** MoE-50, Hash-50 and Prune-50 are also evaluated as converted, on the first 100 validation
  questions. This is reported only, apart from check R0b.

**One choice the review left open.** It offered freezing c_fc so that the router's targets stay valid. That is not taken
here. Training everything keeps the recipe identical to the dense control's, and the router's drift is measured instead:
how often its experts are the true top 4, before and after the fine-tune.

## The rules

**Validity, before any verdict.** If any of the following fails, there is no verdict until the code is fixed.
- **R0a, the evaluation path:** the dense base model, through `eval_moe.py`, matches the greedy predictions in
  `results/step_select/stage0/evals/seed1.jsonl` on all 500 questions.
- **R0b, the conversion:** untrained Prune-50 matches SPARSE-0's S50 predictions (`results/ffn_sparsity/run/evals/S50.jsonl`)
  on the first 100 questions.
- **R0c, the training path:** the dense arm of `train_moe.py` at seed 7, run for 51 updates, logs a loss within 0.02 of
  Stage 1b's `ft6_step` (`results/inner_loop/stage1b/training/ft6_step/metrics.jsonl`) at each of updates 0, 10, 20,
  30, 40 and 50.

**Early kill.** MoE-50 at seed 7 is trained and evaluated first. If it scores 192 or less, 12 below `ft6_step`'s
two-seed mean of 204.5, the run stops there and the verdict is **NEGATIVE**.

**Pooled comparisons.** As `scripts/inner_loop/pool.py` does it: every (seed, question) pair is pooled, with w wins and
l losses.
- The **mean** is (w − l)/2 per seed.
- The **p** is the exact two-sided sign test.
- The **per-seed lower bound** is (w − l − 1.645·√(w + l))/2.
- An arm **MATCHes** the dense control if, against `ft6_step`, its mean is at least −5 and its bound is above −12.

**The verdict, checked in this order.**
1. **NEGATIVE:** MoE-50's mean against `ft6_step` is below −5. The MoE line closes.
2. **INCONCLUSIVE:** MoE-50's mean is at least −5, but its bound is −12 or less. The only extension is a third seed,
   with the dense control rerun at that seed.
3. **PRUNE-SUFFICES:** MoE-50 matches, and so does Prune-50. Pruning half the FFN, then the same fine-tune, is enough,
   and the MoE is not needed.
4. **MATCH-UNRESOLVED:** MoE-50 matches and Prune-50 does not, but MoE − Prune misses +10 per seed with p < 0.05.
5. **CAPACITY-NOT-ROUTING:** MoE-50 matches and beats Prune-50, but MoE − Hash is under +5 per seed. The gain is the
   MoE's extra neurons, not its router.
6. **MoE-MATCH:** MoE-50 matches the dense control, beats Prune-50 by at least +10 per seed with p < 0.05, and beats
   Hash-50 by at least +5 per seed.

## What each result points to

These are pointers, not commitments. Any further run gets its own frozen plan first.
- **MoE-MATCH:** a continuous-thought reasoner at 68% of its active non-embedding parameters, with a router that earns
  its place. Testing 25% with a longer fine-tune would be the natural next question.
- **CAPACITY-NOT-ROUTING:** a fixed hash suffices, so a "learned MoE" claim does not hold. The line closes with that
  finding.
- **PRUNE-SUFFICES:** a positive for the efficiency aim, but not for MoE. A pruned model also saves total parameters
  and memory.
- **NEGATIVE, or MATCH-UNRESOLVED:** the line closes.
- **Phases:** D2's trained phase arm stays off whatever the verdict, since SPARSE-0's R4 did not find the thoughts to be
  the sensitive phase.

## What it cannot show

- **No wall-clock or memory saving.** The masked simulation computes every neuron.
- **Little on reasoning depth:** 175 of the base's 204 right answers come from problems with 3 steps or fewer.
- **Recovery, not distillation:** the MoE starts from the latent model's own weights, so this says nothing about
  distilling latent reasoning into an MoE.
- **Only large effects:** with two seeds, effects under about 10 answers per seed cannot be resolved.

## Reported, never gating

- The training-free arms' scores on 100 questions.
- Each arm's scores per seed, and on problems with at most 3 steps and with 4 or more.
- The router's held-out recall at conversion.
- In evaluation, how often the chosen experts are the true top 4, per phase, for MoE before and after the fine-tune and
  for Hash. Chance is 4/12.
- The training losses.
- Active parameters, with and without the embeddings.

## Code and compute

- `scripts/ffn_sparsity/moe.py` holds the expert masks, the router, the hash table and the parameter counts.
- `scripts/ffn_sparsity/convert.py` builds the conversion.
- `scripts/ffn_sparsity/train_moe.py` runs the fine-tune and the dense check.
- `scripts/ffn_sparsity/eval_moe.py` runs the evaluation, with its identity guard.
- `scripts/ffn_sparsity/report_d1r.py` applies the rules.
- `tests/test_d1r.py` checks the following:
  - the dense hook equals GPT-2;
  - the MoE mask equals a gathered sparse FFN;
  - the hash arm depends only on its keys;
  - Prune equals SPARSE-0's static arm bit for bit, and untrained Prune repeats S50 text for text on a tiny model;
  - the conversion's expert sizes and its sha guard;
  - training, the dense check, the evaluation guard and every verdict.
- `notebooks/DSA_D1R_Colab.ipynb` runs them in one session, plus at most one resume.

**Compute:** about 40 G4-minutes in all, with nothing spent past an early kill.
- six fine-tunes of about 3.5 minutes each;
- seven evaluations of 500 questions at about 35 seconds each;
- three training-free evaluations of 100 questions;
- the conversion, about 2 minutes;
- the dense check;
- setup.
