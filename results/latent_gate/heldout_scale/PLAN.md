# Held-out and scale study: does the depth fix generalize?

**Written 3 November 2026 after Stage 1, and frozen before any run of this study.** Its sha256 is in `PLAN.sha256` beside it.
Later changes are amendments made before the affected run, recorded with the previous hash, as in Stage 1. The engineering
section was completed before freezing, and the notebook refuses to start without `PLAN.sha256`.

**Purpose.** Stage 1 fired its stop rule, so the gate's planned Stage 2 does not run. This study tests the diagnosis
instead, for an analysis paper: running a latent CoT model past its trained depth costs accuracy, and a budget or a
training-free soft dedup restores it. The claim is tested on untouched data and at a second model scale. One conditional
part (C) asks whether a larger model can turn extra depth into accuracy. The search for methods that raise accuracy runs as
a separate line of work and is not part of this plan.

Background:
- [Stage 1 results](../../../docs/LATENT_GATE_STAGE1_RESULTS.md)
- [idea](../../../docs/LATENT_GATE_IDEA.md)
- [Stage 1 plan](../stage1/PLAN.md)

## Models

| Scale | Model | Pinned revision | Role |
|---|---|---|---|
| GPT-2 (124M) | SIM-CoT GPT-2 `internlm/SIM_COT-GPT2-CODI` | `8685b42` | Part A |
| GPT-2 (124M) | CODI GPT-2 `zen-E/CODI-gpt2` | `fd641b3` | Part A |
| LLaMA-3.2 1B | CODI `zen-E/CODI-llama3.2-1b-Instruct` | `b2c88ba` | Part B |
| LLaMA-3.2 1B | SIM-CoT `internlm/SIM_COT-LLaMA3-CODI-1B` | `6e16fe4` | Part B |
| LLaMA-3.2 3B | SIM-CoT `internlm/SIM_COT-LLaMA3-CODI-3B` | `a2db0e5` | Part B, only if credits remain after the 1B runs |

All five were trained with 6 latent steps. 8B is out of scope. Checkpoint files are pinned by sha256 in their fetch scripts.

## Data

- **Discovery: `gsm8k_valid` (500).** It was explored for both GPT-2 models in Stages 0 and 1, and it is untouched for the
  LLaMA models.
- **Primary held-out pool: 2,818 questions.**
  - `gsm8k_test` (1,319), in distribution.
  - `gsm_hard` (1,319) and `multiarith` (180), out of distribution.
  - None has been scored past 6 latent steps or under any policy for any model here.
  - The CODI GPT-2 benchmark run scored all three at 6 steps only (`results/runs/codi/`). Those rows are not reused.
  - `gsm_hard` is `gsm8k_test` with larger numbers, so the two are correlated. Results are reported per set as well as
    pooled.
- **Secondary: `svamp` (1,000).** It is reported separately and is not in the primary pool, because the CODI trail study
  used it as a replication set for CODI GPT-2 depth probes.
- **Order.** Each held-out set is scored once per model, after the engineering checks below pass, with the policies listed
  here and no others.

## Policies

Every policy is greedy, with a budget of 6 latent entries where one applies. Each is defined as in Stage 1's
`stage1_eval.py` unless stated otherwise.

| Policy | Depths | What the answer sees |
|---|---|---|
| `plain` | 6, 12, 24 | every latent entry |
| `keep_first` | 12, 24 | latent entries 1–6. Later steps are still computed, but each new entry is evicted at once. |
| `dedup_soft` | 12, 24 | every entry, each with −log(size of its near-copy cluster, cosine ≥ 0.95 on fed latents) |
| `logn` | 12, 24 | every entry, each with −log(n/6) once there are n > 6 |
| `shift` | 6 | 6 entries, with the answer's positions as if 24 steps had run: a position-only control (new) |
| `gate` | 12, 24 | the 6 highest-scoring entries, for gated checkpoints only |

**Discovery runs only.** These run on `gsm8k_valid` and never on held-out data:
- `window` and `random`;
- `copy_last` and `copy_pair`, which run 6 steps and then append exact copies of the last entry or the last pair, as in
  the copy test;
- the answer's attention share on latents 1–6, by layer, at 6 and 24 steps.

Stage 1's `logscale` is dropped. Its key count includes the question, so it barely sharpens anything (about 3% at 24
steps). A version that counts only latent keys may be run on `gsm8k_valid`, marked exploratory.

## Engineering, with checks that must pass before any held-out run

- **E1. A batched evaluator** (`scripts/latent_gate/batched.py`, `heldout_eval.py`). It handles left-padded batches with a
  per-example bias, for GPT-2 and LLaMA. Eviction masks the entry with HIDE instead of deleting it, which gives the same
  softmax at FP32.
  - **Check** (`check_batched.py`). On `gsm8k_valid`, its GPT-2 predictions must match the Stage 1 batch-1 records on at
    least 99% of questions for every shared policy and depth, otherwise the batch-1 evaluator is used.
  - **Where it runs.** On Colab, as the first step of Part A, because the local machine lacks the memory.
  - **Before freezing.** Unit tests on tiny models showed batched and single-row runs agree for every policy. A 4-question
    run on the real SIM-CoT GPT-2 weights matched Stage 1 exactly, random eviction included.
- **E2. The LLaMA port.**
  - **Loader** (`dsa/bench/models/codi_llama.py`). It merges the CODI LoRA (r = 128, α = 32 on q, k, v, o, gate, up and
    down, per both projects' LLaMA scripts) into the LLaMA weights at FP32, and loads the projection and special tokens.
    SIM-CoT's step decoder and CODI's `dynamic_cls` are dropped.
    - Neither repo ships a usable config (SIM-CoT's `config.json` is empty), so the LLaMA-3.2 architecture is written
      into the loader and checked against the tensor shapes.
    - The tokenizer is SIM-CoT 1B's: the Llama-3.2 Instruct tokenizer with CODI's `[PAD]` at 128256. That avoids the
      gated `meta-llama` repo.
    - Every file is pinned by revision and sha256.
  - **Bias.** It reaches LLaMA as a 4D additive attention mask, which transformers 4.46 adds to the eager logits
    unchanged, so no attention code is patched.
  - **Checks** (`check_llama.py`):
    1. A zero bias reproduces transformers' own mask exactly.
    2. k exact copies of an entry, each given −log k, reproduce the single-entry logits within 1e-4 at FP32, with the
       query at the same position.
    3. `plain` at 6 steps on `gsm8k_test` is within 3 points of the published GSM8K score. The targets are 55.6% for
       CODI-1B (the CODI paper), 56.1% for SIM-CoT 1B and 62.3% for SIM-CoT 3B (the SIM-CoT paper, Tables 2 and 3).
       This run scores `plain` at 6 steps only, before anything else is scored on `gsm8k_test`.
  - **Failure.** If check 3 fails, the model is debugged or dropped, and the outcome is reported either way.
- **E3. The hard-gate tie fix.**
  - **The fix.** It is a new arm, `gate12_fixed`. The gate is computed at FP32 outside autocast, and the top-6 cut keeps
    exactly 6 entries, breaking ties in favor of older entries, as inference does.
  - **Check.** A unit test confirms exactly 6 visible entries, with ties included, and that the training cut matches the
    inference eviction order across random prefixes.
  - **Scope.** Stage 1's `gate12_hard` code and records are not changed.
- **E4. One Colab notebook for Parts A and B** (`notebooks/DSA_Latent_Gate_HeldOut_Scale_Colab.ipynb`). It is pinned to
  a commit, writes to Drive as it goes, stops at any failed check, resumes by part and model,
  and keeps per-model output folders under `results/latent_gate/heldout_scale/<model>/`.

## Part A: GPT-2 on held-out data, and the fixed gate

- **A0.** Both base GPT-2 models are scored on `gsm8k_valid` with every discovery policy and the attention shares. This
  run is check E1, and it gives the paper's tables one evaluator across all models. The set was already explored.
- **A1.** Both base GPT-2 models are scored on the held-out sets with `plain`, `keep_first`, `dedup_soft`, `logn` and
  `shift`.
- **A2.** `gate12_fixed` is trained on both GPT-2 models with Stage 1's data, order, seed (7) and recipe. The tie fix is the
  only change.
  - It is scored on `gsm8k_valid` with `plain`, `gate` and the soft gate.
  - On the held-out sets it is scored with `gate` at 12 and 24, beside Stage 1's `ft6` (`plain` at 6, 12 and 24), taken
    from the Drive backups.

## Part B: LLaMA 1B, and 3B if credits allow

- **B1. Discovery on `gsm8k_valid`.** Every policy above runs, including the discovery-only ones and the attention share.
- **B2. Held-out.** The same runs as A1.
- **3B.** It repeats B1 and B2 only after both 1B models are done and the remaining credits cover it. No training runs at
  3B.

## Hypotheses and decision rules

These are tested for each model on the primary held-out pool, with exact two-sided paired sign tests and 95% paired
confidence intervals on accuracy differences. The noninferiority margin is 2 accuracy points, about the single-seed noise
seen in Stage 1.

- **D1, the drift.** `plain` at 24 is below `plain` at 6, at p < 0.05.
- **D2, the soft dedup fix (primary).** `dedup_soft` at 24 beats `plain` at 24 at p < 0.05. It is also noninferior to
  `plain` at 6: the lower bound of the 95% CI on (`dedup_soft`@24 − `plain`@6) is above −2 points.
- **D3, the budget fix.** The same two criteria as D2, for `keep_first` at 24.
- **D4, the position control.** This one is descriptive. If `shift` costs more than 2 points against `plain` at 6, answer
  positions are a confound and must be reported beside D1.
- **G, a free gain.** `dedup_soft` at 12 or at 24 beats `plain` at 6, at p < 0.025 each (Bonferroni over the two depths).
- **Gate, from A2.** `gate12_fixed`'s `gate` at 24 is noninferior to its `gate` at 12, and it beats `ft6` `plain` at 24 at
  p < 0.05. Its comparison with base `dedup_soft` at 24 is descriptive.

**What the paper may claim:**
- **The fix generalizes.** This needs D2 to hold on every model where D1 holds. A model without D1 is reported as showing no
  drift, not as a failure of the fix.
- **The fix holds across scale.** This needs at least one 1B model to show both D1 and D2.

## Part C: training at 1B, only if Part B shows depth can pay

- **Trigger.** Part C runs if, for a 1B model, either of these holds:
  - G passes; or
  - on `gsm8k_valid`, `dedup_soft` at 12 at least matches `plain` at 6 on the 4+ step problems, while fewer than half of
    latent entries 7–12 fall in a near-copy cluster with an earlier entry. That would be a sign that the extra steps
    compute something new. A near-copy here is an entry whose fed latent has cosine ≥ 0.95 with an earlier one, which
    `dedup_soft` records as `repeats`.
- **Without the trigger.** Part C does not run, and the analysis paper reports Parts A and B.
- **Design.** It uses the triggering 1B model with arms `ft6`, `ft12` and `gate12_fixed`.
  - The recipe is Stage 1's (32,768 GSM8K-Aug problems, seed 7, one pass, lr 2e-5 and gate lr 1e-3, bf16 autocast).
  - It is a full fine-tune if it fits on the A100, otherwise LoRA with r = 128 and α = 32 on every linear layer. The
    choice is recorded before the run.
  - `stage1_train.py` trains GPT-2 only. A LLaMA trainer is built, and checked like E2, only if Part C triggers.
- **Decision rule.** Stage 1's H2 rule applies on `gsm8k_valid`: an arm must beat `ft6` at 6 on the 4+ step problems at
  p < 0.025, not be lower overall, and not drop significantly from 12 to 24. A passing arm, plus `ft6` and base
  `dedup_soft`, is then scored once on the held-out pool.

## Rough cost, A100

| Run | Estimate |
|---|---|
| A1 (two models) | under 1 h batched, 2–3 h batch-1 |
| A2 (two models) | about 1.5 h |
| B per 1B model | 2–3 h |
| B at 3B | about twice the 1B cost |
| C (three 1B arms) | 5–6 h |

Run order: A1 and A2, then B for CODI-1B and SIM-CoT-1B, then 3B if credits remain, then C if triggered.

## Not in this plan

- the Coconut and PCCoT families;
- 8B;
- inference seeds, since greedy decoding is deterministic;
- new accuracy-raising methods (a separate line of work);
- held-out scoring of any Stage 1 arm other than `ft6`.

## Known limits

- **One training seed** for A2 and C.
- **Approximate near-copy detection.** Soft dedup finds near-copies by cosine of fed latents, not by key and value error,
  which the near-copy bound in Codex's review says is what should be controlled. A key/value-based variant may be run on
  `gsm8k_valid`, marked exploratory.
- **The noninferiority margin** is a judgment, fixed here before any data.
