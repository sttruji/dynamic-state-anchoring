# Attention and copied-entry tests

Written 2 November 2026, before either test ran. Exploratory and post hoc: designed after seeing that SIM-CoT keeps a stable,
question-specific trail yet loses answers past 6 steps as CODI does. No training.

Models: CODI `zen-E/CODI-gpt2@fd641b3` and SIM-CoT `internlm/SIM_COT-GPT2-CODI@8685b42`, both trained with 6 latent steps.
Set: `gsm8k_valid`, 500 questions, the same ones used for discovery. Greedy decoding, FP32, MPS.

**Hypothesis.** Past 6 steps, answers change because the readout spreads attention over extra latent entries in its cache,
not because of what those entries contain.

## Test 1: where the readout attends

Plain runs at 6, 12 and 24 latent steps. For the `<eot>` query, and averaged over the generated answer tokens, the share of
attention (mean over 12 layers and 12 heads, also kept per layer) on: the first token, the other question tokens, `<bot>`,
latent entries 1–6, latent entries 7 and later, and the readout tokens. The hypothesis predicts the share on entries 1–6
falls as depth grows and the share on later entries rises. Descriptive, with no pass bar. Secondary: the AUROC of a question's
`<eot>` share on entries 7–24 for predicting that its answer changed between 6 and 24 steps.

## Test 2: copied entries

Run 6 steps, then append exact copies of cached key/value entries without any new forward pass, and read out at the position
a 6-step run uses. The only difference from a 6-step run is extra identical entries; duplicating a key k times is the same as
adding log(k+1) to its attention logit.

- `copy_last`: entry 6 repeated 6 or 18 times, giving 12 or 24 latent entries.
- `copy_pair`: entries 5 and 6 repeated 3 or 9 times, giving 12 or 24, which keeps both of CODI's phases.
- Control: no copies, read at the explicit 6-step position. It must reproduce the plain 6-step texts on all 500, and the
  plain 6, 12 and 24 texts must match the earlier probe records; otherwise the run is invalid.

**Decision**, per model and variant, at 24 entries against plain 6 steps, with net loss = answers lost − answers gained and
a two-sided exact sign test:

- *Copies reproduce the drift*: below 6 steps with p < 0.05, and a net loss at least half that of plain 24 steps.
- *Copies do not cause the drift*: a net loss under a quarter of plain 24 steps'.
- Otherwise *partial*.

Plain 24-step net losses: CODI 33 (14 gained, 47 lost), SIM-CoT 34 (9 gained, 43 lost). The 12-entry results are descriptive.
