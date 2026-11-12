# CODI past its trained depth: what the latent trail does

**Update, 12 November 2026.** Later tests changed how section 5 reads. SIM-CoT GPT-2, whose latents settle and keep
their question content, loses as many answers past 6 steps. Appending exact copies of CODI's last cache entry or last
pair, which adds no new content, reproduces about a third to a half of CODI's loss, with the rest coming from the
changing content of its new entries ([LATENT_GATE_IDEA.md](LATENT_GATE_IDEA.md),
[copy test report](../results/latent_gate/copy_test/report.txt)). So the content decay measured here is
only part of the cause; repeated entries also dilute the answer's attention. A training-free soft dedup later held CODI
at 201 of 500 at 24 steps ([Stage 1 results](LATENT_GATE_STAGE1_RESULTS.md)). The
[held-out study](LATENT_GATE_HELDOUT_RESULTS.md) then used `gsm8k_test`, GSM-Hard and MultiArith, and confirmed that the
answer's position costs GPT-2 almost nothing, while at 1B, under RoPE, it is most of the loss.

Added 2 November 2026. These are no-training measurements of the published CODI GPT-2 weights (`zen-E/CODI-gpt2@fd641b3`, the `codi` row in the [benchmark](BENCHMARK.md)). They ask whether CODI's latent loop drifts when run past the 6 steps it was trained on, which is the premise of Dynamic State Anchoring, and what the trail carries. Decoding is greedy with 256 tokens, on Apple MPS. The sets are `gsm8k_valid` (500 questions) and SVAMP (1,000). `gsm8k_test` was not used.

Records are in [results/codi_trail](../results/codi_trail), report outputs in [results/codi_trail/reports](../results/codi_trail/reports) and scripts in [scripts/codi_trail](../scripts/codi_trail). The probe's texts at 3, 6, 12 and 24 steps match the harness sweep on all 500 validation questions. Its 6-step texts match the committed `codi` rows: 500/500 on validation and 1,000/1,000 against the A100 SVAMP row. The dynamics run's 6 and 12 step texts match the probe on all 500.

**Summary.** CODI does drift past 6 steps, slowly: 206 correct at 6, 192 at 12 and 173 at 24. Two preregistered ideas failed. Agreement between depths is a strong confidence signal but loses to the token log-probability of a single decode, and re-feeding a held latent did not stop the drift. Post hoc measurements show the drift is not a position effect. The question-specific part of each latent decays, and later latents cannot replace early ones. The trail matters on GSM8K but much less on SVAMP, where 306 questions settle on one shared vector and a constant trail answers nearly as well. For DSA this means an anchor has a real target but nothing to win at fixed weights: perfect stabilization only matches stopping at 5 or 6 steps.

## 1. Accuracy by depth

| Latent steps | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 12 | 24 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Correct of 500 | 126 | 120 | 182 | 190 | 209 | **206** | 203 | 202 | 202 | 200 | 192 | 173 |

Steps 1, 3, 6, 12 and 24 are harness rows in [sweep](../results/codi_trail/sweep), not added to the main leaderboard. The rest come from the dynamics run in section 4. Against 6 steps, 12 steps gained 9 questions and lost 23 (exploratory paired p=0.020), and 24 steps gained 14 and lost 47 (p<0.001). The closeness of wrong answers to gold falls in the same direction. At 1 step CODI writes a fragment of written reasoning instead of an answer, so that column is not a fair latent score. 5 steps against 6 is 209 against 206, p=0.45.

## 2. Preregistered test 1: depth agreement as a confidence signal

Rules were fixed in [PREREGISTRATION.md](../results/codi_trail/PREREGISTRATION.md) (sha256 `3a4b0dc6…`) before any result, with validation for discovery and SVAMP for replication. The signal is whether the 3, 6 and 12 step answers agree, scored on the 6-step answer. To pass, it had to beat three baselines at its own coverage: the summed token log-probability of the 6-step answer, its lowest token probability, and question length.

| SVAMP, n=1,000 | Precision at 593 answered | Difference from agreement, 95% CI | AUROC |
|---|---:|---:|---:|
| Depth agreement | 61.6% | | 0.814 |
| Summed token log-probability | 63.7% | −2.2 pts (−3.8 to −0.8) | 0.907 |
| Lowest token probability | 63.7% | −2.2 pts (−3.9 to −0.8) | 0.906 |
| Shorter question | 48.6% | +13.0 pts (+10.1 to +16.5) | 0.649 |

**Failed.** Agreement replicated strongly (61.6% right when the depths agree, 8.6% otherwise, p≈2e-70) but lost to both token-confidence baselines, which need one decode instead of three. On validation the gap was larger, −9.7 points. Token confidence is not a length artifact: within outputs of equal length its AUROC is 0.89–0.90. Combining the two, fitted on validation and scored on SVAMP, raises AUROC by only 0.005.

## 3. Preregistered test 2: holding the latent

From step 7 on, the same pass re-fed one fixed latent instead of the newest one. The preregistered variant re-feeds the latent produced at step 6; the sensitivity variant re-feeds the last latent the 6-step model actually consumed.

| Correct of 500 | 12 steps | 24 steps |
|---|---:|---:|
| Plain CODI | 192 | 173 |
| Hold the step-6 latent (preregistered) | 191 | 165 |
| Hold the last latent fed (sensitivity) | 195 | 189 |
| Stop at 6 | 206 | 206 |

**Failed.** The preregistered hold did not beat plain 24 (21 gained, 29 lost, p=0.32) and stayed below 6 steps. The sensitivity hold slowed the drift (34/18 against plain 24, p=0.037) but was still below 6 steps at both depths (p=0.008 at 24). This hold is an untrained stand-in for an anchor, so it does not rule out a trained one.

## 4. Post hoc: the trail's dynamics

Everything in this section was designed after seeing sections 2 and 3, so it is exploratory. The rules for the main run were fixed before it ran in [DYNAMICS_PLAN.md](../results/codi_trail/DYNAMICS_PLAN.md) (sha256 `e33677b7…`).

**The trail alternates between two states.** The cosine between neighbouring latents is about 0.25 while the cosine between latents two steps apart is 0.93–0.998. 497 of 500 validation questions show it. CODI's training puts no loss on intermediate steps, only on the answer after the last latent, so nothing targets this pattern directly. [Li et al. (PMPS)](https://arxiv.org/abs/2609.09928) report the same odd-even pattern in CODI GPT-2 within the 6 trained steps. Here it persists to 24 steps, and it does not show in the answers: the depth curve in section 1 is smooth across odd and even steps, and even depths were ahead in only 1 of 6 neighbouring pairs.

**The drift is not positional.** Reading out at the position a 6-step run would use leaves accuracy unchanged: 193 against 192 at 12 steps and 174 against 173 at 24.

**Question-specific content decays.** About a quarter of each latent's energy is specific to the question; the rest is shared across questions. Raw cosine to step 6 stays at 0.985 (step 12) and 0.943 (step 24), but after removing each step's mean over questions it falls to 0.938 and 0.796. Matching a question's step-6 latent to its own later latent among all 500 succeeds 98.8% of the time at step 8, 92.2% at 12 and 67.4% at 24.

**Early latents cannot be replaced.** Feeding only the last six latents after the question gives 165 (the six before step 12) and 143 (before step 24). Both are below the plain long trail (36 and 43 lost, p<1e-4), which keeps the early latents in its cache.

**The trail matters on GSM8K and much less on SVAMP.**

| Correct | Own trail | Mean trail over all questions | Next question's trail |
|---|---:|---:|---:|
| `gsm8k_valid`, of 500 | 206 | 63 | 28 |
| SVAMP, of 1,000 | 400 | 355 | 183 |

Another question's trail does not carry its answer over: on validation the answer matches the donor's on 8 of 500, against 9 by chance. By the plan's wording this is a mixed result rather than a portable-content one, but it rules out the trail being a generic signal on GSM8K.

**Many SVAMP questions settle instead of cycling.** On 306 SVAMP questions the latent stops changing (cosine between steps 6 and 7 at least 0.95). These fixed points are nearly the same vector for every question (mean pairwise cosine 0.97). The 6-step answer is right on 242 of them (79%), against 137 of the 649 questions that keep cycling (21%). The mean trail still gets 231 of those 242. Only one validation question settles.

**Similarity between depths detects settling, not correctness.** Neighbouring-step similarity is at chance for predicting a correct 6-step answer on validation (AUROC 0.43–0.58). On SVAMP it reaches 0.73–0.79, which reflects the fixed-point group. Similarity two steps apart, `cos(4,6)`, gives 0.677 and 0.686 on the two sets, chosen after seeing them. For predicting an answer that stays the same at 24 steps, adding `cos(4,6)` to token log-probability lowers SVAMP AUROC from 0.859 to 0.818.

**Picking a depth by token confidence did not hold up.** The rule required a gain over 6 steps with p<0.05 on both sets. Choosing among 3, 6 and 12 steps by summed log-probability gave 421 against 400 on SVAMP (25 gained, 4 lost) but 203 against 206 on validation. The best choice on validation reached 207.

## 5. What this means for DSA

The proposal's premise holds for CODI: states do deteriorate past the trained depth, and the loss is in question-specific content rather than position. But stopping at 5 or 6 steps already avoids it, the untrained hold did not help, and similarity between depths does not identify correct answers. An anchor on frozen CODI could at best recover 206–209. A benefit would need training that makes extra depth add accuracy, compared against [Kuo et al.](https://arxiv.org/abs/2606.29983), who stabilize looped models across depths by randomizing loop counts during training.

## 6. Related work read for this note

| Paper | Overlap with this note |
|---|---|
| [Li et al., PMPS (2609.09928)](https://arxiv.org/abs/2609.09928) | Odd-even similarity collapse in CODI GPT-2 across its 6 latents, 100 GSM8K-Aug test questions; also trains separate models with up to 8 latents |
| [Duraipandian et al. (2607.09698)](https://arxiv.org/abs/2607.09698) | CODI GPT-2 moves between two modes; T=6 only; ICML 2026 workshop, non-archival |
| [Chang et al. (2606.01243)](https://arxiv.org/abs/2606.01243) | Early latents are causal hubs; transplanting a whole latent sequence flips 2.5% of answers in CODI or CoCoNuT at 3B–8B (model and set not stated) |
| [Li et al. (2602.08783)](https://arxiv.org/abs/2602.08783) | Single-step interventions including a step-specific mean on CODI, T=6 |
| [Blayney et al. (2604.11791)](https://arxiv.org/abs/2604.11791) | Looped LMs settle into a cycle of fixed points, one per layer of the block, tested to 128 recurrences |
| [Kaissis et al. (2602.10097)](https://arxiv.org/abs/2602.10097) | A period-4 answer-state cycle in a looped parity model, on one probe input |
| [Fan et al., LOTUS (2606.31779)](https://arxiv.org/abs/2606.31779) | One loop past training: 70.0% at the trained 6, 69.3% at 7 |
| [Popescu et al. (2607.20519)](https://arxiv.org/abs/2607.20519) | Post-hoc confidence readouts often match or beat learned halting gates |
| [Dilgren and Wiegreffe (2604.04902)](https://arxiv.org/abs/2604.04902) | Latent tokens are often unnecessary on logical reasoning sets |
| [NoisyCoconut (2605.08221)](https://arxiv.org/abs/2605.08221) | Abstains when noise-perturbed latent paths disagree |

Not found in what was read: CODI's accuracy beyond its trained step count and why it falls, the shared fixed point on SVAMP and its accuracy split, or a constant trail nearly matching the real one. The search was not exhaustive, and several papers were read only in part.

## Limits

One published checkpoint, one model size and one decoding setup were measured, with no retraining, so none of this shows how a differently trained CODI behaves. Sections 4 and 5 are post hoc. SVAMP is a gated held-out set in the harness and has now been explored, so a claim built on it needs a fresh set such as MultiArith or GSM-Hard; `gsm8k_test` remains unused. The mean trail is computed over the same questions it is tested on. Decay and retrieval use the projected latents only. The last hidden states show the same alternation, more weakly in early steps, and the same chance-level neighbouring-step scores on validation.

## Reproduce

```bash
scripts/codi_trail/sweep.sh                       # about 35 min on MPS
python scripts/codi_trail/probe.py --dataset gsm8k_valid --output results/codi_trail/probe/gsm8k_valid.jsonl --hold
python scripts/codi_trail/probe.py --dataset svamp --output results/codi_trail/probe/svamp.jsonl
python scripts/codi_trail/probe_check.py results/codi_trail/probe/gsm8k_valid.jsonl results/codi_trail/sweep
scripts/codi_trail/dynamics.sh                    # about 25 min in three shards
```

Each probe run takes about 37 minutes on MPS and writes latents (77–154 MB, not committed) that the similarity, phase, follow-up, residual and dynamics scripts need. Each script's docstring gives its report command.
