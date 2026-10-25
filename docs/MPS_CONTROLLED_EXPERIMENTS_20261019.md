# Next controlled MPS experiments — design dated 19 October 2026

Status: **design only; no training or new answer evaluation has started**.
The audit and written judgments are complete. The reviewed fixture and warm-up
support described below have not been implemented. This document does not claim
that the current trainer already accepts these settings.

## Question and rationale

Can DSA learn a usable selective write policy without sacrificing its starting
explicit-CoT ability? FLT has already improved tiny-set fitting; the unresolved
issues are controller learning, useful memory, and validation performance.

The new audit found no saved-index/tokenization error in 512 historical rows,
but found incorrect or ambiguous supervision in three of the local 32 questions
and one given-restatement checkpoint. A loop-position-only label predictor also
beats the old controller on the same diagnostic labels (AUC 0.7818 vs 0.5135).
Thus neither finite gradients nor gate AUC above 0.5 alone establishes semantic
selection. See `CHECKPOINT_LABEL_AUDIT_20261019.md`.

## Freeze the inputs before running

Machine-readable membership and design: `provenance/mps_experiment_plan_20261019.json`.

- Use the pinned `yingfanbot/gsm-cot-gpt2` revision
  `e0ef1887404adbf6c531d2cac7804694321f87bd` and the existing tokenizer plus `<SA>`.
- Create a **separately named derivative** using the 32 reviewed TRAIN source
  rows in the plan: 29 retained local questions and supported replacements
  99,935 / 186,418 / 219,649. Remove only the specified `120=120` given-restatement
  event in row 112,564, regenerating the marked text, teacher indices and K=8
  schedules consistently. Preserve all underlying source strings and gold answers.
  Expected derivative: 29 auxiliary-supervised questions, three no-event controls,
  and 58 retained checkpoint events. Assert these counts during preparation.
- This is a curated mechanism-development fixture. It changes membership and one
  annotation decision; comparison with historical runs is descriptive, not an
  isolated causal estimate of label quality. All new experimental arms use the
  identical new derivative. Preserve the original 32 and 512 derivatives.
- Use the **same first 64 validation questions** as the previous CPU follow-up for
  the initial answer/explicit-retention checks. Their identities are frozen in
  the plan; they are reused development data, not a new independent holdout.
  Do not use validation rationales to define writes, targets or training data.
- Keep all 1,319 test questions reserved. Only after a promising replicated
  configuration should development evaluation expand to all 500 validation rows.

## Common training and measurement contract

Run one arm at a time on `mps`, FP32, eager attention, existing FLT memory-query
recurrence, K=8, four workspace vectors and one anchor. Use seed 7 initially,
AdamW at 3e-5, weight decay 0.01, gradient accumulation 1, global norm clip 1,
activation checkpointing and constant learning rate. All modules, initial weights,
tokenizer, selected examples and the actual sampled sequence must match across
arms. Shuffle once with the training seed and repeat that order, as the current
trainer does. Every condition gets 200 single-example updates (6.25 exposures).

Retain the current objective:

`L = answer_CE + explicit_CoT_CE + 0.1 gate_BCE + 0.1 candidate_alignment`.

Positive gate weight stays 1, threshold stays 0.5. Candidate alignment remains
normalized MSE to detached pre-marker hidden states from the ordinary shared
teacher route. Do not add CODI's separate answer-boundary alignment loss, change
the recurrence, or retune weights during this comparison.

Record per-update component losses, writes/probabilities, unclipped global and
gate-head gradient norms, actual sample IDs, trainable parameter count, synchronized
time, MPS allocation snapshots and checkpoint hashes. Count clipping; do not
require an arbitrary clipping fraction to claim success. Frozen phases naturally
change which parameters contribute to global gradient norm.

At steps 0/64/128/200, collect question-only gate decisions on the 32 TRAIN
questions in eval mode, scoring against their saved training-only proxy labels.
Report AUC, precision/recall, write frequency and the always-no-write label
baseline. Refit a loop-position baseline on this fixture and report leave-one-
question-out scores; do not reuse the historical 0.7818 as a new-fixture target.
These are observable controller outputs, not latent-state semantic probes.
Report ordinary candidate-alignment loss only as the existing training proxy.

Answer evaluation is question-only, greedy, 64 new tokens. Explicit controls use
the ordinary language route, unmarked prompts and a fixed 192-token cap. Preserve
the pinned LOTUS numeric score as primary; additionally report full-output scalar
validity and a strict-scalar accuracy sensitivity to expose incidental extraction
credit. The two routes' token caps differ: retention comparisons are explicit
before vs explicit after, not equal-compute latent-vs-explicit superiority claims.

## E0: Establish the MPS reference without optimizer updates

Evaluate the untouched checkpoint through explicit CoT on the frozen 64 development
questions. Also record K=1 and K=8 no-write answers from a freshly initialized DSA
wrapper. Keep the two baselines separate: the former measures existing reasoning
ability; the latter measures the initially unadapted latent interface. Save decoded
outputs and prompts so formatting differences can be checked. Low untrained latent
accuracy is expected to need adaptation and is not a stopping criterion by itself.

## E1: Matched write-policy reference on the reviewed fixture

| Arm | Write rule | Updates | Backbone |
| --- | --- | ---: | --- |
| J-static | Never update the initial learned anchor | 200 | Joint training throughout |
| J-every | Write every loop | 200 | Joint training throughout |
| J-periodic | Write at loops 4 and 8 | 200 | Joint training throughout |
| J-learned | Hard threshold 0.5, straight-through training | 200 | Joint training throughout |

Run all four from verified identical initialization. All get the same auxiliary
supervision even when their actual write policy is fixed. Evaluate the final
checkpoint on all 32 training questions and the fixed 64 development questions,
and evaluate each trained backbone through explicit CoT on the same 64.

This supplies an honest local reference after the audit. Cleaner data may change
behavior; it does not by itself identify the cause of previous collapse. Do not
declare a DSA mechanism gain based on a better result than the historical run.

## E2: One controlled change — delay backbone updates

Compare J-learned with one additional arm, W-learned:

- Steps 1–64: freeze **all backbone weights**, including embeddings and LM head;
  train only initial memory, workspace normalization, candidate projection and
  controller. Keep the same total loss and dropout policy. CoT CE is recorded but
  cannot update the frozen backbone. Teacher weights are consequently fixed only
  during this phase; no separately frozen teacher is introduced.
- Steps 65–200: unfreeze the backbone and continue the original joint objective
  at the same learning rate. Keep the optimizer object and existing head moments;
  newly unfrozen weights acquire Adam state on their first update. Reset no RNG,
  data cursor or head optimizer state at the transition.

Both arms see exactly the same 200 examples in the same order. The treatment is
the training schedule; it deliberately gives the backbone 136 rather than 200
updates. It cannot isolate teacher-target drift from reduced backbone exposure.

Measure whether gate discrimination/actual writes improve, explicit capability
is better retained, and final answer performance improves. A gate score increase
without an answer benefit is an engineering result, not support for the full claim.

If warm-up is promising, run **W-static, W-every and W-periodic** with the identical
64+136 schedule before attributing an advantage to selective anchoring. Comparing
W-learned only with joint-trained fixed policies would confound schedule and policy.

If collapse persists, the next separately designed comparison can block answer-loss
gradients into the gate while retaining gate BCE and identical hard forward
decisions. That tests gradient conflict; it should not be bundled with warm-up,
new thresholds, new clock inputs or loss-weight changes in this first wave.

## E3: Test additional computation, then replicate

For matched-policy checkpoints under a fixed recipe, evaluate the same 64 development
questions at K=1/4/8/16 without retraining. Report correct-to-wrong and wrong-to-
correct pairs, latency and actual writes, not only aggregate accuracy. Training
remains K=8; K=16 is an extrapolation. K=32 (the proposal's 4x setting) is a later
bounded inference check if K=16 is useful and resources permit.

Do not supply gold schedules at validation/test time. Optional forced annotation-
schedule comparisons are TRAIN-only diagnostics and must be clearly labeled;
the token-relative schedule is not a proven optimal or oracle semantic policy.

Proceed to additional seeds 11 and 19 and the full 500 validation questions only
after controller collapse is resolved enough to test an actual selective policy
and the 64-question results are promising relative to matched controls. Freeze
the chosen recipe before these runs. Report paired differences, uncertainty,
per-seed outcomes, retained explicit capability and time/memory together. A one-
or two-question gain on the 64 does not establish a generalization improvement.
Add standalone CODI and unmodified LOTUS under a separately matched protocol
before the final project benchmark; these internal arms do not replace them.

## MPS budget and implementation work

The completed M1/16-GiB feasibility check measured 3.86–5.78 seconds per warm K=8
update over four updates per arm, with a maximum observed driver allocation of
5.06 GiB. Those are short-run observations, not sustained throughput or peak-memory
guarantees. At a planning range of 4–6 seconds/update:

- E1: 800 updates, approximately 53–80 minutes of optimizer time.
- E2: one extra 200-update arm, approximately 13–20 minutes; reuse J-learned.
- Conditional matched warm-up controls: another 600 updates, approximately
  40–60 minutes. Repeated seeds and larger data are outside this first budget.

Model loading, checkpointing, gate checks and especially explicit generation add
time. Time the first eight development generations before estimating the rest.
Use sequential arms and `save_every=200` initially; the existing trainer saves a
resumable step-zero checkpoint and final checkpoint, while its default frequent
saves would keep two large post-update states per arm. The first five arms can
use roughly 10 GB for weights and optimizer states, depending on serialization.
About 15 GiB was free during design; check actual space before execution and do
not make duplicate full checkpoint ZIPs. Future repeats need a separate retention
plan, not accumulation of every full checkpoint.

Work required before execution:

1. Implement the separately named fixture builder with source/ledger hashes,
   explicit event override and asserted 32/29/58 counts. Include its provenance
   in the preparation/training contract; never bypass the existing identity checks.
2. Implement the freeze/unfreeze schedule in trainer configuration, contract and
   resume state; verify frozen weights stay fixed and transition/resume behavior
   agrees in a bounded tiny MPS integration check. Existing flags do not do this.
3. Provide the small eval-mode gate-output summary and an MPS-safe ordinary explicit
   evaluator. The older comprehensive diagnostics script uses float64 reductions
   and has not been verified on MPS; do not run it unchanged or silently fall back
   to CPU. No intermediate-state probing is required.
4. Verify fixture membership, initialization equality and scorer/prompt identity;
   then execute E0, E1, E2 sequentially in fresh experiment directories.

No experiment has been launched and no warm-up efficacy is claimed in this design.
