# Stage 0: latent-entry eviction at fixed weights

Written 2 November 2026, before any Stage 0 run. Exploratory: designed after the copied-entry test showed that extra latent
entries in the cache, even exact copies, cost answers past 6 steps. No training.

Models: CODI `zen-E/CODI-gpt2@fd641b3` and SIM-CoT `internlm/SIM_COT-GPT2-CODI@8685b42`, both trained with 6 latent steps.
Set: `gsm8k_valid`, 500 questions, already explored. Greedy decoding, FP32, MPS. `gsm8k_test` is not used.

**Question.** If the latent loop runs to 12 or 24 steps while the cache holds at most 6 latent entries, which rule for
choosing the entries comes closest to, or beats, stopping at 6? And does the decode label proposed for the Stage 1
controller pick better entries than simple rules?

## Setup

Steps 1–6 run as usual. From step 7 on, each new latent entry is appended, and if the cache then holds 7 latent entries one
is evicted. The question tokens and `<bot>` are never evicted. Later latent steps and the answer readout therefore see at
most 6 latent entries. Each latent keeps its natural position id (step t at position question length + t), so position is
the same as in a plain run. The answer is read out at 12 and 24 steps.

Eviction rules, all applied inside the loop:

| Rule | Evicts |
|---|---|
| `keep_first` | the newest entry, so only latents 1–6 stay (a control for position alone) |
| `recency` | the oldest entry (a sliding window) |
| `dedup` | the older entry of the most similar pair, by cosine of the latent vectors fed |
| `attention` | the entry with the lowest mean attention received per later query (layers and heads averaged); the newest is never evicted |
| `random` | a random entry, seeded by the question index |
| `oracle` | uses the gold rationale. A label marks an entry whose logit-lens top-1 token is one of the question's gold intermediate results. It evicts the oldest unlabelled entry; if all are labelled, the older of two entries with the same decoded value; otherwise the oldest |

One readout-only variant, `oracle_readout`, runs the plain loop to 12 or 24 steps and keeps 6 of the latent entries for the
answer only. It picks labelled entries with distinct values, most recent first, and fills the remaining slots with the
earliest unpicked entries.

Controls: the plain 6, 12 and 24 step texts, from an unbatched run, must match the earlier probe records on all 500;
otherwise the run is invalid.

## Expectations and readout

Expected from the copied-entry and decoding results: no rule beats plain 6 steps (206 for CODI, 204 for SIM-CoT), because the
frozen models' later latents repeat early results. Each rule's 12 and 24 step score is reported against plain 6 and plain
12/24, with exact two-sided sign tests, with no correction for multiple comparisons. These are the comparisons that matter for Stage 1:

1. *Restores*: a rule at 24 steps is not below plain 6 (p ≥ 0.05) and is above plain 24 (p < 0.05).
2. *Label beats heuristics*: `oracle` above the best of `recency`, `dedup`, `attention` and `random` at 24 steps, p < 0.05.
   If it does not, the decode label carries no usable selection signal at fixed weights. That does not rule out its use as a
   training target, but it removes the Stage 0 support for it.
3. Accuracy on questions with 4 or more rationale steps is reported per rule, descriptively.
