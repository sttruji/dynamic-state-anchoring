# Brainstorm track: summary

Status, 12 November 2026. After Stage 1 of the latent gate, a separate session (dsa-brainstorm) looked for methods that
raise accuracy above the 6-step score, mostly on SIM-CoT GPT-2 and the 500 `gsm8k_valid` questions. The only confirmed
gain is small: a training-free vote over dropout reruns, +9 of 1,319 on GSM8K test (about +0.7 points) at about 9×
greedy's compute. This work is separate exploration and is kept out of the course report. The working list of ideas, with the reasons each was tried
or dropped, is [ACCURACY_IDEAS.md](brainstorm/ACCURACY_IDEAS.md). Each entry below gives the verdict as its source
records it.

## A. Inner loop: more depth inside each latent step

Loop GPT-2 layers 4–7 inside each of SIM-CoT GPT-2's six latent steps, so the extra compute adds no cache entries.

**A null.** Pooled over seeds 7 and 8, `loop_ce`@k3 beat its control `ft6_ce` by +6.0 per seed (+31/−19, p=0.119),
short of the preregistered +7 at p < 0.05. The loop's gate never opened: in all three loop trainings it peaked near
0.011. Seed noise was as large as the effects tested: the same `ft6_step` recipe scored 209 at seed 7 and 200 at seed 8.

[Results](../results/inner_loop/README.md) · [quick reference](brainstorm/INNER_LOOP_QUICK_REFERENCE.md) ·
[deep pass on A, D and F](brainstorm/Looped%20steps%20decoding%20and%20recursive%20attention.md)

## F. Recursive attention (KK-LSE)

A learned key-to-key bias over the latent entries that feeds back into the main attention logits. Soft dedup is a
hand-written special case of it. Tested at 6 steps, in the inner loop's Stage 1b.

**STOP.** The bias stayed small (mean |α| 0.026), and switching it off at test changes 1 answer of 500.

[Results](../results/inner_loop/README.md) (Stage 1b)

## B. Step-level selection over dropout reruns

At each latent step, choose among the model's own step and MC-dropout reruns of it.

- **Stage 0, validation: STOP.** One-step branches cover +6.4% over greedy, under the 8% GO bar, and an untrained
  selector using SIM-CoT's step decoder scores −4 against greedy (p = 0.52).
- **E0, offline checks on Stage 0's records** for a multi-candidate extension, found one small positive: a vote over
  greedy and the reruns at steps 1–3 (+7/−1 on validation).
- **Run 1, GSM8K test (1,319 questions).** That vote **CONFIRMS**: 528 against greedy's 519, net +9 (+12/−3),
  p = 0.035, at about 9× greedy's compute. A hidden-state probe to pick reruns is **KILLED**: within-set AUC 0.495,
  against a bar of 0.65. Step-level selection ends here, and the test set is spent for it.

[Results](../results/step_select/README.md) · [plan](brainstorm/STEP_SELECTION.md) ·
[multi-candidate design study](brainstorm/MULTI_CANDIDATE_SELECTION.md)

## H. Answer-conditioned plan search

From the sleep-and-memory idea pipeline. On training questions the model gets wrong, search the first latent z₁ by
gradient for plans whose own later steps reach the gold answer, then distil them into the loop.

**KILL at Stage 0.** Of 1,000 greedy-wrong training questions, gold plans were accepted on 676 and wrong-answer plans on
478, a ratio of 1.41, under the 1.5 bar. The search mostly writes answers into z₁ rather than finding plans, so Stage 1
did not run.

[Results](../results/plan_search/README.md)

## Step probe: where long problems break

Cross-fitted linear probes on SIM-CoT GPT-2's hidden states, to tell readout failures from compute failures.

**INCONCLUSIVE.** The answer probe reads gold on only 8.7% of greedy-right training questions, against a bar of 50%.
Exploratory reads: on right training questions, step 1 reads 73–80% of the time and step 2 56–63%; on wrong questions,
35–58% and 22–27%. That fits errors that begin at the first steps, but no preregistered rule confirms it.

[Results](../results/step_probe/README.md)

## MoE student: SPARSE-0 and D1-R

Could SIM-CoT GPT-2 run as a mixture of experts, with fewer active parameters per token at the same accuracy?

- **SPARSE-0, evaluation only: GO at 50%.** A per-token oracle that keeps half of each FFN keeps 200 of 204 answers,
  while one fixed half collapses to 60.
- **D1-R: NEGATIVE, by the early kill.** Converted to a mixture of experts at half the FFN and fine-tuned with the dense
  control's recipe, MoE-50 at seed 7 scored 179, against 209 for the dense `ft6_step`. The MoE line is closed.

[SPARSE-0](../results/ffn_sparsity/README.md) · [D1-R](../results/ffn_sparsity/d1r/README.md) ·
[idea round notes](../research_notes/MoE%20latent%20reasoning/)

## Not run

- **C**, more supervised latents per step, with its composition-data arm, and **G**, gold-filtered replay of the
  model's own samples (latent RFT), with its explicit-route arm. Both were dropped on 8 November without a run.
- **D**, decode and re-inject, was not built. In the inner loop's Stage 0, recomputing the arithmetic in a decoded step
  fixed 0 wrong answers.
- **E**, scale, was left to the main line's held-out study, where no policy significantly beat 6 steps at 1B either.

## Surveys

- [Latent reasoning limitations ranked](brainstorm/Latent%20reasoning%20limitations%20ranked.md): the limitations and
  future work stated in 42 latent-reasoning and looped-transformer papers.
- [Looped steps decoding and recursive attention](brainstorm/Looped%20steps%20decoding%20and%20recursive%20attention.md):
  the deep pass that ranked A, D and F.

The source notes behind these docs are in [`research_notes/`](../research_notes).
