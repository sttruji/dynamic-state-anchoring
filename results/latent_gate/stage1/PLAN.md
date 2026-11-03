# Stage 1: training a model to use more latent steps under a cache budget

Written 2 November 2026 after Stage 0 and before any Stage 1 run, and fixed then: its sha256 is in `PLAN.sha256` beside it.
The idea and evidence are in [docs/LATENT_GATE_IDEA.md](../../../docs/LATENT_GATE_IDEA.md); the literature is in
[docs/LATENT_GATE_LITERATURE.md](../../../docs/LATENT_GATE_LITERATURE.md).

## Setup

- **Base model.** SIM-CoT GPT-2 (`internlm/SIM_COT-GPT2-CODI@8685b42`, from `scripts/latent_gate/fetch_simcot.py`). The
  replication is CODI (`zen-E/CODI-gpt2@fd641b3`). Both were trained with 6 latent steps.
- **Data.** GSM8K-Aug train, a seeded sample of 32,768 problems (seed 7) with questions of at most 256 tokens. Batch 32,
  1,024 updates, one pass. Every arm sees the same examples in the same order.
- **Optimisation.** Full fine-tune of the merged GPT-2 weights and the projection at 2e-5, and the gate at 1e-3; AdamW with
  weight decay 0.01, 3% warmup then linear decay, gradient clipping at 2. bf16 autocast on CUDA.
- **Loss.** Cross-entropy on `The answer is: N<eos>` after `<eot>`, as in CODI's student loss, with no teacher
  distillation, since the starting checkpoints are already trained.

## Arms

| Arm | Latent steps in training | What each query can see |
|---|---|---|
| `ft6` | 6 | the full cache: a control for the extra fine-tuning |
| `ft6x2` | 6, on twice the examples | the full cache: roughly the compute of a 12-step arm |
| `ft12` | 12 | the full cache |
| `ftrand` | 6–12, drawn per batch | the full cache (randomized loop counts, Kuo et al.) |
| `window12` | 12 | the 6 latent entries before it (a fixed recency budget) |
| `rand12` | 12 | above 6 entries, one random entry is hidden for good at each step: training under eviction with no learned choice |
| `gate12` | 12 | every entry j, with log g_j added to its attention logits; a penalty of 0.1 × max(0, Σg − 6) |
| `gate12_label` | 12 | as `gate12`, plus a binary cross-entropy (weight 1) pulling g_j towards the entry's label |

**Label.** 1 when the logit-lens top-1 token at the entry's position, or at the next entry's position, is one of the
problem's gold intermediate results (`<<a*b=c>>`); otherwise 0. It is computed from the model being trained, detached. The
next-entry spread keeps CODI's operator-phase entries, which never decode to a number, from all being labelled 0.

**Label check, before any training.** `scripts/latent_gate/stage0_labels.py` removes one latent entry at a time from the
frozen models at 12 steps and measures the drop in the gold answer's log-probability. If labelled entries do not matter more
than unlabelled ones, `gate12_label` is dropped from the run and H3 is not tested.

*Result, 2 November 2026* ([labels-report.txt](../stage0/labels-report.txt)):

- On SIM-CoT, the base model here, spread-1 labelled entries matter no more than unlabelled ones: removing one costs 0.023
  against 0.025 nats.
- For both models, the label's AUROC for picking the most important quarter of entries is 0.41–0.45, below chance.
- Entry 1 dominates: removing it costs 0.42–0.54 nats, while removing any later single entry costs almost nothing.
  Removing entries 7–12 slightly raises the gold answer's log-probability.
- Stage 0's in-loop label oracle also scored far below simple rules: 73 and 158 at 24 steps, against dedup's 201 and 203.

**So `gate12_label` is dropped and H3 is not tested.** The arm stays in the code for a test run outside this plan.

## Evaluation

`scripts/latent_gate/stage1_eval.py`, greedy, at 6, 12 and 24 latent steps. Policies:

- `plain`: the full cache, also run at 6 steps.
- `window`: keep the 6 most recent latent entries.
- `random`: evict a random latent entry above 6.
- `logn`: every latent entry gets −log(n/6) once there are n > 6. This restores the latent block's total mass.
- `logscale`: every logit of a query is multiplied by log K / log K₆, the multiplicative SSMax-style sharpening.
- `dedup_soft`: each entry gets −log(size of its cluster of near-copies, cosine ≥ 0.95). This exactly cancels copies.
- `gate`: hard eviction of the lowest-g entry above 6, with log g kept on the survivors.
- `gate_soft`: log g on every entry and no eviction.

The frozen base model is scored the same way, as `base`, with every policy. Its `logn`, `logscale` and `dedup_soft` rows are
the training-free rival fixes. Each trained arm is scored with `plain` plus the condition it trained in: `window` for
`window12`, `random` for `rand12`, `gate` and `gate_soft` for `gate12`, and `dedup_soft` for `ft12`. A hard gate with tied
scores evicts the newest entry, so an undecided gate behaves like `keep_first`, the best Stage 0 rule.

Discovery and selection use `gsm8k_valid` only. Report all 500, plus the 173 problems with 4 or more rationale steps. The
held-out sets `gsm_hard`, MultiArith and `gsm8k_test` are scored once, after the comparison below is settled, for the
selected arm and `ft6` only.

## Decision rules

All tests are exact two-sided paired sign tests on `gsm8k_valid`.

- **H2, the main claim.** `gate12` (hard `gate` policy, 12 steps) beats `ft6` at 6 steps on the 4+ step problems, with
  p < 0.05, and is not lower overall. It must also not fall significantly from 12 to 24 steps.
- **H2 against the rival fixes.** The same arm must also beat each of these on the 4+ step problems:
  - `ft6x2` plain at 6, the compute control;
  - `ft12` plain at 12;
  - `ftrand` plain at 12;
  - `window12` window at 12;
  - `rand12` random at 12;
  - the base model with `logn`, `logscale` and `dedup_soft` at 12.

  If any of them matches it, the claim narrows to that simpler fix.
- **H3.** Not tested, since the label check failed. It would have required `gate12_label` to beat `gate12` at 12 steps on
  all 500 (p < 0.05).
- **Stop.** If no 12-step arm beats `ft6` on the 4+ step problems, GPT-2 does not use the extra depth even when trained.
  Report that, with the CODI replication of the two best arms, and do not run Stage 2.
- The gap between `gate` and `gate_soft` is reported as the cost of hard eviction.

## Known limits, before running

- One seed per arm. A positive H2 is rerun with a second seed before it is reported.
- Not included, from the literature review's list: a terminal fixed-point objective, a state-replacing PCCoT arm, a budget
  sweep, delayed eviction, value-norm scoring and a Gumbel-annealed gate. They are Stage 2 if H2 passes.
- The gate is trained jointly with the model. Popescu et al. find that joint training can damage the trajectory, so a
  positive result should be checked with the gate fitted on a frozen model.
