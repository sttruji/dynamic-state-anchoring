# Raising accuracy past trained depth: working list

Brainstorm notes from the dsa-brainstorm session, started 3 November 2026. These are ideas, not plans. Citations marked
"(verify)" have not been checked yet.

**Update, 5 November:** the deep pass on A, D and F is in
[Looped steps decoding and recursive attention](Looped%20steps%20decoding%20and%20recursive%20attention.md). It
recommends A first, D as cheap diagnostics, and F paired with a loop that keeps one cache entry per iteration. One
correction to the list below: 48.24 is PCCoT's 12-latent baseline; PCCoT itself scores 49.48 on GPT-2.

## What our evidence says any method must overcome

- **No new work past step 6.** The latents repeat early results: CODI repeats rationale steps 1–2, and SIM-CoT's trailing
  latents repeat the answer.
- **Fine-tuning finds no use for steps 7–12.** One pass of answer-only fine-tuning at 12 steps gains nothing, and the gate
  learns "keep the early entries".
- **Depth gives little answer diversity.** Counting a question as solved if any of 6, 12 or 24 steps gets it right adds
  only 9–15 answers of 500, and majority voting across those depths never helps.
- **Every appended entry costs attention.** It dilutes the early entries unless corrected by soft dedup or a budget.
- **Recipe matters more than "GPT-2 can't".** PCCoT, training from scratch, found 12 latents much better than 6 for GPT-2.
  So the objective and recipe, not model size alone, may be the lever.

## Candidate directions

| Idea | Why the evidence points here | Closest prior work to check | Cheapest first test |
|---|---|---|---|
| **A. Deeper per step, not more steps.** Loop a block of layers inside each latent step, so extra compute adds no cache entries. | Appending entries dilutes attention and adds nothing; extra compute per step adds no entries at all | Huginn, Ouro; retrofitted recurrence for pretrained LMs (verify); InfiLoop, RecurTrace | Literature check, then a short fine-tune of CODI GPT-2 with randomized loop counts per latent step |
| **B. Test-time scaling from latent diversity.** Add noise or dropout to the latents, then take a majority vote or use a verifier. | Depth gives almost no diversity, but self-consistency gives large gains for explicit CoT | SoftCoT++; LatentSeek; a 2025 parallel test-time scaling paper for latent reasoners (verify) | Training-free: maj@k and pass@k with Gaussian noise on the projection output, on gsm8k_valid |
| **C. Give later latents a job.** Extend SIM-CoT's per-step supervision to more latents, such as 2 per rationale step or "recompute and check" targets. | Answer-only training never assigns work to steps 7–12; PCCoT says the recipe matters | SIM-CoT, PMPS, PCCoT, CoLaR | Fine-tune SIM-CoT GPT-2 at 12 steps with the auxiliary step decoder, against `ft12` |
| **D. Decode and re-inject.** Every k latents, decode the intermediate result and feed it back as a discrete token. | It breaks the attractor and gives later steps new input. It reuses the user's decode idea as an input, not a gate label | Token Assorted, hybrid latent reasoning such as HRPO (verify) | Risky: Stage 0 showed logit-lens decodes are unreliable at GPT-2, so it would need a trained decoder |
| **E. Scale.** | Larger models may compute new things past step 6 | The other session's Part B tests this (the free-gain test G) | Wait for Part B before spending training here |
| **F. Recursive attention (the user's idea, 3 Nov).** A second attention pass over the latent entries decides what the main attention may select. For example, keys attend to each other to find redundant or important entries, and the result biases the main logits. Pair it with A: the loop adds compute, and recursive attention keeps the extra entries from spreading attention thin. | It targets dilution directly and depends on the query, where our gate gives each entry one fixed score. Soft dedup is already a hand-written special case: key-to-key cosine with a hard 0.95 threshold, fed back as −log cluster size | Selective Attention (Leviathan et al., 2410.02703); Multi-Token Attention (2504.00927); Talking-Heads Attention (2003.02436); Attention on Attention (1908.06954). All from memory (verify) | "Learned dedup": replace the fixed threshold with a small trained key-to-key attention on CODI at 12–24 steps, compared against `dedup_soft` |

## Open questions

- At GPT-2 scale, CODI roughly matches explicit CoT fine-tuning, about 43–44% on GSM8K. Beating that probably needs
  test-time compute (B) or more serial compute without dilution (A, C).
- F alone is a selection method, and so far selection (gate, dedup) has only held accuracy, never raised it. Any gain
  has to come from new compute (A or C), with F letting that compute run past step 6 without dilution.
