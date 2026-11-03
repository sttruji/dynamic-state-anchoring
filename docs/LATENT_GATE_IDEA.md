# Dynamic State Anchoring, reframed: anchor what the answer can see

Status, 3 November 2026: Stages 0 and 1 are done; see [Stage 1 results](LATENT_GATE_STAGE1_RESULTS.md). All evidence is exploratory and comes from
`gsm8k_valid`, which has already been used for discovery. Records are under [results/latent_gate](../results/latent_gate),
and the literature is in [LATENT_GATE_LITERATURE.md](LATENT_GATE_LITERATURE.md).

## The problem

CODI-style latent reasoning feeds each latent back as an input embedding, so every latent step adds one entry to the
key/value cache. The published GPT-2 models are trained with 6 steps. Running more steps loses answers (correct of 500):

| Latent steps | 6 | 12 | 24 |
|---|---:|---:|---:|
| CODI | 206 | 192 | 173 |
| SIM-CoT (CODI with per-step supervision) | 204 | 185 | 170 |

The original DSA proposal blamed deteriorating latent states. The measurements point elsewhere:

- **Stable states drift just as much.** SIM-CoT's latents stop oscillating and keep their question content (a step-24 latent
  identifies its question among 500 in 97.6% of cases, against 67.4% for CODI), yet it loses as many answers.
- **Copies alone cause the drift.** After 6 steps, appending exact copies of the step-6 cache entry reproduces SIM-CoT's
  whole drop (172 against 170 at 24 entries) and a third of CODI's (196 against 173).
- **Attention moves off the early latents.** The answer's attention share on latents 1–6 falls from 0.158 to 0.033 between 6
  and 24 steps in CODI, and from 0.099 to 0.036 in SIM-CoT.
- **Extra steps compute nothing new.** Past step 6 the latents repeat the results of rationale steps 1–2. On 5+ step
  problems, CODI falls from 10 to 3 correct.

Duplicating a key k times adds log(k+1) to its attention logit, so each extra entry takes attention from the entries that
carry the reasoning.

## The idea

**DSA anchors which past latent states later steps and the answer can see, not the states themselves.** A small gate scores
each latent entry when it is produced. The loop keeps at most 6 entries, the trained count, and evicts the rest, so later
steps and the answer always see a bounded set, however long the loop runs.

- **Training: a soft gate.** log g_j is added to every attention logit pointing at entry j, the exact counterpart of the
  log(k+1) that copies add. A budget penalty pushes the sum of g towards 6.
- **Inference: a hard gate.** The lowest-g entry is evicted whenever the budget is exceeded.
- **The model is fine-tuned too,** at 12 steps under the budget, so that later steps can learn to compute later results.
  Any gain has to come from this.
- **Proposed label, now dropped.** Supervising the gate with each latent's logit-lens decode, matched to gold
  intermediate results, failed its check (below).

## Stage 0: eviction at fixed weights

Both frozen models were run at 12 and 24 steps with at most 6 latent entries, under six rules. All controls matched on 500
of 500 ([plan](../results/latent_gate/stage0/PLAN.md), [report](../results/latent_gate/stage0/report.txt)).

| Correct of 500 at 24 steps | Plain | Keep first 6 | Drop near-duplicates | Recency | Random | Attention score | Decode label |
|---|---:|---:|---:|---:|---:|---:|---:|
| CODI (206 at 6 steps) | 173 | 205 | 201 | 141 | 157 | 77 | 73 |
| SIM-CoT (204 at 6 steps) | 170 | 204 | 203 | 152 | 173 | 190 | 158 |

- **H1 holds.** Keeping the first six entries, or dropping near-duplicates, restores the 6-step score at 24 steps, and
  nothing beats it.
- **Early entries are what matter.** Recency hurts. Removing entry 1 alone costs 0.42–0.54 nats of gold-answer
  log-probability, while removing any later single entry costs almost nothing.
- **The decode label fails.** Keeping entries whose decode matches a gold result collapses CODI, because it discards the
  operator-phase latents. As a predictor of an entry's causal effect its AUROC is 0.41–0.45
  ([label check](../results/latent_gate/stage0/labels-report.txt)), so the label arm is out of Stage 1.

## What the literature says

- **The gate mechanism is not new.** Learned additive logit gates with hard eviction exist for text (TRIM-KV, Selective
  Attention, FoX, DMS). Bounding access to past loop states rescues depth in state-replacing looped models (InfiLoop,
  RecurTrace).
- **What looks open.** No paper runs CODI or SIM-CoT past their trained depth, and none shows that a settled latent state
  still degrades the answer once it is appended repeatedly. That contradicts the depth-safety arguments of Think Shallow,
  Solve Deep and Blayney et al., which assume a fixed context. This diagnosis stands even if training fails.
- **The main warning.** Every published stabilization past trained depth preserved peak accuracy without exceeding it.

## Hypotheses

- **H1, Stage 0:** holds, as above.
- **H2, Stage 1, the main claim.** Fine-tuned at 12 steps with the gated budget, the model beats the same model fine-tuned
  at 6 on problems with 4+ rationale steps, and holds at 24.
- **H3, the label:** dropped.
- **Stop rule.** If H2 fails, GPT-2 cannot use the depth, and the write-up is the diagnosis.

## Plan

- **Stage 1, prepared** ([plan](../results/latent_gate/stage1/PLAN.md), notebook
  `notebooks/DSA_Latent_Gate_Stage1_Colab.ipynb`). SIM-CoT GPT-2 is fine-tuned with equal data in eight arms:
  - 6 steps;
  - 6 steps on twice the data, as a compute control;
  - 12 steps with the full cache;
  - random 6–12 steps;
  - a recency window;
  - random eviction;
  - the gate, trained soft;
  - the gate, trained with the top-6 cut used at inference.

  The frozen model also gets the training-free fixes: additive log-n, multiplicative log-n, and soft deduplication. CODI is
  the replication. Held-out sets are scored once, at the end.
- **Stage 2, if H2 passes.** A second seed, the 1B checkpoints, a budget sweep and a fixed-point objective arm.

## Risks

- **GPT-2 may not use more depth.** PMPS and SIM-CoT found their best at 6–7 latents.
- **The gate may only learn "keep the first six".** That would tie with `ft6`.
- **Soft training, hard inference.** Scoring both modes measures the gap.
- **One explored set.**
