# CODI depth agreement and hold test: rules fixed before any new data (2026-11-01)

Model: zen-E/CODI-gpt2@fd641b3 with no training. It trained at 6 latent steps. Decoding matches `dsa/bench/models/codi.py`.
Scoring is `dsa.bench.scoring.score`, bench-v1. `gsm8k_test` is not touched.

## Control gate
On gsm8k_valid, the instrumented 3, 6, 12 and 24 step texts must match the harness sweep on all 500 questions at all four depths.
Any mismatch stops the analysis.

## Idea 1: depth agreement. Replication set: SVAMP (n=1000)
gsm8k_valid is where the signal was found, so it is reported but does not count.
Signal A: the extracted answers at 3, 6 and 12 steps are all equal. The answer being judged is the 6-step answer.
Baselines, all from the 6-step readout or the question:
- B1 sum of generated token log-probabilities
- B2 minimum generated token probability
- B3 question length in tokens, shorter = more confident

R1 Replicates: accuracy with A true exceeds accuracy with A false, Fisher two-sided p < 0.01.
R2 Beats baselines: at A's coverage c, A's precision exceeds the precision of the top c by each baseline. The 95% bootstrap CI
   of each difference (10,000 resamples of questions) must exclude 0.
R3 Adds information: take the baseline with the best AUROC on gsm8k_valid. Within its quintiles on SVAMP, the mean of
   (accuracy with A) - (accuracy without A) has a 95% bootstrap CI excluding 0.
Pass = R1, R2 for all three baselines, and R3.

## Idea 2: hold test. Set: gsm8k_valid (n=500)
z_t is the projection output after step t; step t feeds z_(t-1). The trained 6-step model feeds z_0..z_5 and never feeds z_6.
Primary: hold-next. Steps 7+ feed z_6 unchanged.
Sensitivity: hold-fed. Steps 7+ feed z_5 unchanged.
H1 Primary test: hold-next at 24 steps vs plain 24 steps (173), exact paired sign test p < 0.05, gains > losses.
H2 Recovery: hold-next at 24 is not significantly below plain 6 steps (206) by the same test.
H3 Gain: hold-next at 12 or 24 significantly above 206. Only this counts as a benefit over stopping at 6.
Reported without a test: hold at 12 vs plain 12, and how many answers equal the 6-step answer.

## Exploratory, no pass or fail
Latents z and last hidden states h are saved at every step. Similarity across depths is checked as a predictor of 6-step correctness
and of whether the answer changes by step 12, as a first look at a similarity-triggered write.
