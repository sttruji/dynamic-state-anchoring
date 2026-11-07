# Raising accuracy past trained depth: working list

Brainstorm notes from the dsa-brainstorm session, started 3 November 2026. These are ideas, not plans. Citations marked
"(verify)" have not been checked yet.

**Update, 5 November:** the deep pass on A, D and F is in
[Looped steps decoding and recursive attention](Looped%20steps%20decoding%20and%20recursive%20attention.md). It
recommends A first, D as cheap diagnostics, and F paired with a loop that keeps one cache entry per iteration. One
correction to the list below: 48.24 is PCCoT's 12-latent baseline; PCCoT itself scores 49.48 on GPT-2.

**Update, 6 November: A and F are nulls** (`results/inner_loop/README.md`).
- Over two seeds, the inner loop beats its control by +6 per seed (p=0.12), short of the preregistered bar, and its gate
  never opened.
- KK-LSE at 6 steps changes 1 answer of 500.
- Seed noise is about 9/500: the same recipe scored 209 and 200 across seeds.
- B is next, with a twist: SIM-CoT's step decoder as a free verifier for sampled latent chains. Prior work on B (Parallel
  Test-Time Scaling for Latent Reasoning Models, arXiv 2510.07745, ACL 2026) finds coverage rises but selection is the
  bottleneck. Its trained latent reward model adds about 1 point, and it states that latents cannot be scored by existing
  process reward models.
- On greedy SIM-CoT chains, those whose decoded steps are all arithmetically valid and contain the answer are right 63%
  of the time, against 36% for the rest. That gap is across questions, though; within a question it is untested.

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

## Additions from the sleep and memory pipeline (6 November)

**How these were produced.** Two specialists wrote briefs on latent reasoning and on the neuroscience of memory and REM
sleep. A generator proposed 9 ideas, a literature review grounded them, and a two-agent council debated each one to a
signed decision.

**The decisions:**
- ADD: **G**, from rescue replay, and **H**, from reverse replay.
- MERGE: REM recombination into **C**, and System-2 sleep into **G**.
- REJECT: tag and capture, hippocampal index, TMR-cued nap, dentate separation and dream unlearning.

**What carries over.**
- The shared kill rule is 2 seeds, GO at +10/seed with p < 0.05 against ft6, and a margin of at least +5/seed over a
  gold-duplicate control (the same questions with gold CoTs).
- Entries are named by mechanism, not by sleep. The sleep framing is a metaphor here, and LatentEvolve (2509.24771)
  already published the day-and-night consolidation framing.

| Idea | Why the evidence points here | Closest prior work to check | Cheapest first test |
|---|---|---|---|
| **G. Gold-filtered replay of the model's own samples (latent RFT).** *Pending the user's ruling on whether Run 1's closure of "RL in this project" covers one round of rejection-sampling fine-tuning.* On training questions, keep the MC-dropout latent chains that reach the gold answer where greedy fails. Distil them back for one cycle (L1 on latents plus answer CE), interleaved with standard data. Explicit-route arm: the model's own sampled CoTs that differ from gold and pass a Python check of every `<<a op b=c>>` become extra CODI teacher targets. | Dropout chains cover +58/+55 over greedy, but no selector picks the winners (step selection's Stage 0 STOP). Training can use the gold answer instead. GSM8K-Aug has one path per question, and RFT's gain tracks distinct verified paths. | SLPO (2607.19691): the same signal through RL, +0.46 on CODI GPT-2, with no RFT baseline. RFT (2308.01825): LLaMA-7B 35.9→41.7, about +2 at k=3. STaR; ReST-EM. LatentEvolve (2509.24771). | **Preflight** (eval, about 28 G4-min): greedy plus 8 chains (p = 0.2) on as many training questions as fit in 25 minutes (at least 4,000), then 8 more chains on the rescued questions. Stop if rescues are under 10% of greedy-wrong, or if fewer than 250 are rescued in both seeds. **Arms** (2 seeds, about 20 G4-min): shared ft6; replay of stable rescues (at most 4 replays per chain, at most 25% of each batch); gold-duplicate. **Explicit arm:** a preflight (eval, about 15 G4-min) needs at least 0.45 new verified paths per question at k = 8, then verified paths against gold-duplicate (about 12 G4-min). **Kill** under the shared rule. After a pass, run the label-removed arm at 2 seeds, and credit the filter only at ≥ +5/seed. |
| **H. Answer-conditioned plan search, then distillation.** On training questions the model gets wrong, search z₁ by gradient until the model's *own* steps 2–6 reach the gold answer. Distil the reachable plans into the loop. | It is the only route to the 238–241 validation questions that no chain solves. A gold reward moves latents far at test time (LatentSeek: Qwen2.5-1.5B 68.1→85.4). Open: do such plans lie near greedy, and do they survive distillation? | LatentSeek (2505.13308); LTPO; AMVL (2607.00461), which names "answer leakage"; STaR's rationalization (+0.6 on GSM8K); Latent Thought Flow (2606.16222). | **Stage 0** (eval, about 10 G4-min; about 10 more if G does not run): on up to 1,000 greedy-wrong training questions, take 20 Adam steps on z₁ in the top-64 PCA span, toward gold and toward the wrong integer nearest gold in base log-prob. Accept a plan if it gives the exact target and stays within that question's farthest dropout chain. Kill if gold acceptance is under 1.5× wrong, or if there are fewer than 150 plans. **Stage 1** (about 32 G4-min): 2 seeds of shared ft6, distillation and gold-duplicate, under the shared rule. After a pass, run wrong-plan distillation at 2 seeds. |

**Row C gains a data arm, from REM recombination.**
- **The arm:** train at 12 steps on exact compositions, "Let X be the answer to [A]. [B with one quantity replaced by X]",
  with B's equations re-run and non-integer or negative results rejected.
- **The evidence:**
  - only 11/500 validation questions need more than 6 steps;
  - 4+-step problems are 17.6–19% of training, against 34.6% of validation;
  - 85% of random pairs compose arithmetically, but their semantics are unchecked (Compositional GSM hand-edited 25%).
- **Prior work:** MathFusion (2503.16212), and Compositional GSM (2410.01748).
- **The test:**
  - *Gate* (eval, about 5 G4-min): run ft12 on 500 compositions of validation pairs. Stop if composed accuracy is at least
    0.9× the product of the component accuracies.
  - *Arms* (pairs with 7–12 total steps, 2 seeds, about 24 G4-min): 25% composition and 25% shuffled composition, against
    ft6 and ft12. ft12 is reused only on a matching recipe hash and seeds.
  - *Kill* if composition beats shuffled by less than +5/seed on the 173 questions with 4+ steps, or fails to beat
    ft12-long by 5/seed.

## Open questions

- At GPT-2 scale, CODI roughly matches explicit CoT fine-tuning, about 43–44% on GSM8K. Beating that probably needs
  test-time compute (B) or more serial compute without dilution (A, C).
- F alone is a selection method, and so far selection (gate, dedup) has only held accuracy, never raised it. Any gain
  has to come from new compute (A or C), with F letting that compute run past step 6 without dilution.
